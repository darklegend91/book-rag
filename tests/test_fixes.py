"""Regression tests for the audit fixes not already covered by audit/test_audit.py.

Offline: no model server, no encoder weights.
Run: python -m unittest discover -s tests -v
"""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, PropertyMock, patch

import numpy as np

from bookrag.chat.engine import ChatEngine, _cited_numbers
from bookrag.config import Config, load_config
from bookrag.evaluate.harness import faithfulness_eval, retrieval_eval
from bookrag.index.store import IndexBusy, Store, index_lock
from bookrag.paper.blueprint import Blueprint, SectionSpec, attempt_from_instructions
from bookrag.paper.export import export_paper
from bookrag.paper.generator import Paper, PaperGenerator
from bookrag.paper.pyq import PYQProfile, PYQQuestion
from bookrag.paper.verifier import verify_question
from bookrag.retrieve.pipeline import RetrievalResult
from bookrag.schemas import Chunk, Question, ScoredChunk


class ScriptedLLM:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = 0

    def complete_json(self, *a, **k):
        self.calls += 1
        return self.replies.pop(0)


def question(**kw):
    d = dict(number=1, section="A", text="What is thermal equilibrium?", marks=2,
             qtype="short", bloom="remember", topic="Thermal", answer="Same temperature.")
    d.update(kw)
    return Question(**d)


def mcq(**kw):
    return question(qtype="mcq", options=["Hot", "Cold", "Same temperature", "Moving"],
                    correct_option="Same temperature", **kw)


ALL_FALSE_BUT = lambda k: {"options": [{"index": i, "true": i == k} for i in range(4)]}


class VerifierTests(unittest.TestCase):
    def test_string_false_is_rejection(self):
        v = verify_question(ScriptedLLM({"answerable": "false", "confidence": 0.99}), question(), "ctx")
        self.assertFalse(v.answerable)

    def test_non_object_payload_is_rejection(self):
        self.assertFalse(verify_question(ScriptedLLM([]), question(), "ctx").answerable)

    def test_out_of_range_confidence_fails_closed(self):
        v = verify_question(ScriptedLLM({"answerable": True, "confidence": 95}), question(), "ctx")
        self.assertFalse(v.answerable)

    def test_mcq_without_options_rejected_before_llm(self):
        llm = ScriptedLLM()
        self.assertFalse(verify_question(llm, question(qtype="mcq"), "ctx").answerable)
        self.assertEqual(llm.calls, 0)

    def test_mcq_partial_option_check_rejected(self):
        llm = ScriptedLLM({"answerable": True, "confidence": 0.99},
                          {"options": [{"index": 2, "true": True}]})
        self.assertFalse(verify_question(llm, mcq(), "ctx").answerable)

    def test_mcq_complete_check_passes_and_letter_key_normalised(self):
        q = mcq()
        q.correct_option = "C"
        llm = ScriptedLLM({"answerable": True, "confidence": 0.9}, ALL_FALSE_BUT(2))
        self.assertTrue(verify_question(llm, q, "ctx").answerable)
        self.assertEqual(q.correct_option, "Same temperature")

    def test_mcq_two_true_options_rejected(self):
        flags = {"options": [{"index": i, "true": i in (1, 2)} for i in range(4)]}
        llm = ScriptedLLM({"answerable": True, "confidence": 0.9}, flags)
        self.assertFalse(verify_question(llm, mcq(), "ctx").answerable)


class GeneratorTests(unittest.TestCase):
    def setUp(self):
        chunk = Chunk("c1", "Thermal equilibrium means the same temperature.", "", "b1", "Book")
        rr = SimpleNamespace(context=chunk.text, results=[ScoredChunk(chunk, 1)], grounded=True)
        self.retriever = SimpleNamespace(retrieve=lambda *a, **k: rr, store=SimpleNamespace(chunks=[chunk]))

    def test_malformed_items_retry_instead_of_crash(self):
        llm = ScriptedLLM({"questions": ["bad"]}, {"questions": [None]}, {"questions": 5})
        gen = PaperGenerator(Config(), retriever=self.retriever, llm=llm)
        paper = gen.generate(Blueprint(sections=[SectionSpec("A", count=1)], topics=["T"]),
                             progress=lambda m: None)
        self.assertEqual((paper.report.accepted, paper.report.rejected), (0, 1))

    def test_verification_disabled_is_not_labelled_verified(self):
        llm = ScriptedLLM({"questions": [{"text": "Define heat.", "answer": "Energy in transit."}]})
        gen = PaperGenerator(Config({"grounding": {"verify_questions": False}}),
                             retriever=self.retriever, llm=llm)
        paper = gen.generate(Blueprint(sections=[SectionSpec("A", count=1)], topics=["T"]),
                             progress=lambda m: None)
        self.assertFalse(paper.questions[0].verified)

    def test_dedup_short_questions(self):
        ohm = question(text="State Ohm's law.")
        self.assertTrue(PaperGenerator._is_duplicate(ohm, [question(text="State Ohm's law.")]))
        self.assertFalse(PaperGenerator._is_duplicate(ohm, [question(text="State Hooke's law.")]))


class MarksAndExportTests(unittest.TestCase):
    def test_attempt_parsing(self):
        self.assertEqual(attempt_from_instructions("Answer ANY ONE."), 1)
        self.assertEqual(attempt_from_instructions("Attempt any 3 questions"), 3)
        self.assertEqual(attempt_from_instructions("Answer all questions."), 0)

    def test_optional_section_marks(self):
        bp = Blueprint(sections=[SectionSpec("A", count=3, marks_each=2, instructions="Answer ANY ONE."),
                                 SectionSpec("B", count=2, marks_each=5)])
        qs = [question(number=i) for i in range(3)] + [question(section="B", marks=5) for _ in range(2)]
        paper = Paper(bp, questions=qs)
        self.assertEqual(paper.total_marks, 12)
        self.assertEqual(bp.computed_marks, 12)

    def test_no_key_json_has_no_answers(self):
        paper = Paper(Blueprint(sections=[SectionSpec("A", count=1)]), questions=[mcq()])
        paper.report.rejections.append({"topic": "T", "reason": "answer is Same temperature"})
        with tempfile.TemporaryDirectory() as d:
            [p] = export_paper(paper, Path(d), "s", formats=("json",), include_key=False)
            data = json.loads(p.read_text())
        self.assertNotIn("answer", data["questions"][0])
        self.assertNotIn("correct_option", data["questions"][0])
        self.assertNotIn("rejections", data["report"])

    def test_pyq_structure_counted_per_paper(self):
        profile = PYQProfile(papers=["one", "two"], total_marks=4,
                             questions=[PYQQuestion(marks=2, type="short", source_paper=p)
                                        for p in ["one", "one", "two", "two"]])
        bp = profile.to_blueprint()
        self.assertEqual(bp.computed_marks, 4)


class EvaluationTests(unittest.TestCase):
    def test_missing_verdicts_count_as_unsupported(self):
        llm = ScriptedLLM({"claims": ["a fact", "another fact"]},
                          {"verdicts": [{"claim": "a fact", "supported": True}]})
        rr = SimpleNamespace(context="ctx")
        engine = SimpleNamespace(llm=llm, reset=lambda: None,
                                 ask=lambda q: SimpleNamespace(grounded=True, text="x", retrieval=rr))
        with patch("bookrag.chat.engine.ChatEngine", return_value=engine):
            res = faithfulness_eval(Config(), ["q"], progress=lambda m: None)
        self.assertEqual(res["faithfulness"], 0.5)

    def test_empty_retrieval_run_reports_nothing_measured(self):
        chunk = Chunk("c1", "text " * 200, "", "b", "B", token_count=200)
        retriever = SimpleNamespace(store=SimpleNamespace(chunks=[chunk]))
        broken = Mock()
        broken.complete.side_effect = ValueError("down")
        with patch("bookrag.evaluate.harness.Retriever", return_value=retriever), \
                patch("bookrag.evaluate.harness.client_from_config", return_value=broken):
            m = retrieval_eval(Config(), n_samples=1, progress=lambda m: None)
        self.assertEqual((m.n, m.hit_at_1), (0, None))


class ChatTests(unittest.TestCase):
    def rr(self):
        return RetrievalResult("q", ["q"], [], [], "src", ["Book p.1", "Book p.2"], 0.9, True)

    def test_grouped_citations_parsed(self):
        self.assertEqual(_cited_numbers("A [1, 2]. B [3][4]."), {1, 2, 3, 4})

    def test_sentence_citation_mode(self):
        cfg = Config({"grounding": {"require_sentence_citations": True}})
        engine = ChatEngine(cfg, retriever=SimpleNamespace(), llm=Mock())
        bad = engine._finalize("q", "q", "Heat is energy in transit [1]. "
                               "Entropy always increases in every single process.", self.rr())
        good = engine._finalize("q", "q", "Heat is energy in transit [1]. "
                                "Entropy of an isolated system never decreases [2].", self.rr())
        self.assertEqual((bad.grounded, good.grounded), (False, True))


class FakeEmbedder:
    def embed_passages(self, texts, show_progress=False):
        v = np.random.default_rng(0).normal(size=(len(texts), 8)).astype(np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    def unload(self):
        pass


class IndexConfigMixin:
    def cfg(self, d):
        raw = copy.deepcopy(load_config().raw)
        raw["paths"] = dict(index_dir=d + "/index", books_dir=d + "/books",
                            pyq_dir=d + "/pyq", export_dir=d + "/exports")
        return Config(raw=raw)


class IndexTests(IndexConfigMixin, unittest.TestCase):
    def test_append_dedupe_refresh_and_lock(self):
        from bookrag.index.builder import build_index
        body = lambda w: f"# Chapter {w}\n" + f"The {w} principle explains energy transfer. " * 60
        with tempfile.TemporaryDirectory() as d, \
                patch("bookrag.index.builder.embedder_from_config", return_value=FakeEmbedder()), \
                patch("bookrag.index.builder.arbiter_from_config", return_value=Mock()):
            cfg = self.cfg(d)
            books = Path(d) / "books"
            books.mkdir()
            (books / "alpha.md").write_text(body("alpha"))
            (books / "alpha_copy.md").write_text(body("alpha"))
            m = build_index(cfg, progress=lambda m: None)
            self.assertEqual([b["title"] for b in m["books"]], ["alpha"])

            extra = Path(d) / "beta.md"
            extra.write_text(body("beta"))
            m = build_index(cfg, [extra], progress=lambda m: None)
            self.assertEqual(sorted(b["title"] for b in m["books"]), ["alpha", "beta"])

            extra.write_text(body("beta") + " Changed. " * 50)
            m = build_index(cfg, [extra], progress=lambda m: None)
            store = Store(cfg.index_dir).load()
            self.assertEqual(len({c.book_id for c in store.chunks}), 2)
            self.assertEqual(len(store), len(store.vectors))

            with index_lock(cfg.index_dir):
                with self.assertRaises(IndexBusy):
                    build_index(cfg, progress=lambda m: None)

    def test_save_switches_generation_and_prunes(self):
        chunk = lambda t: Chunk("a::00001", t, t, "a", "A", ordinal=1)
        with tempfile.TemporaryDirectory() as d:
            s = Store(Path(d))
            for text in ("one", "two", "three"):
                s.save([chunk(text)], np.ones((1, 2)), {"books": [{"book_id": "a"}]})
            self.assertEqual(len(list(Path(d).glob("gen-*"))), 2)
            self.assertEqual(Store(Path(d)).load().chunks[0].text, "three")


class ClaimLLM:
    def __init__(self, reply=None, fail_fast=False, fail_all=False):
        self.reply, self.fail_fast, self.fail_all = reply, fail_fast, fail_all
        self.calls = []

    def complete_json(self, system, user, fast=False, temperature=None):
        self.calls.append(fast)
        if self.fail_all or (fast and self.fail_fast):
            raise RuntimeError("model down")
        return self.reply


class ClaimCheckTests(unittest.TestCase):
    CFG = Config({"grounding": {"verify_answer_claims": True}})
    ANSWER = "Heat is energy in transit [1]. Entropy never decreases in isolation [2]."
    BOTH_OK = {"verdicts": [{"id": 1, "supported": True}, {"id": 2, "supported": True}]}

    def finalize(self, llm, answer=None):
        rr = RetrievalResult("q", ["q"], [], [], "ctx", ["Book p.1", "Book p.2"], 0.9, True,
                             passages=["Heat is energy in transit.",
                                       "Entropy of an isolated system never decreases."])
        engine = ChatEngine(self.CFG, retriever=SimpleNamespace(), llm=llm)
        return engine._finalize("q", "q", answer or self.ANSWER, rr)

    def test_supported_answer_is_grounded(self):
        self.assertTrue(self.finalize(ClaimLLM(self.BOTH_OK)).grounded)

    def test_unsupported_statement_refuses(self):
        llm = ClaimLLM({"verdicts": [{"id": 1, "supported": True},
                                     {"id": 2, "supported": False, "reason": "not stated"}]})
        answer = self.finalize(llm)
        self.assertFalse(answer.grounded)
        self.assertIn("Entropy never decreases", answer.text)

    def test_missing_or_string_false_verdict_refuses(self):
        self.assertFalse(self.finalize(ClaimLLM({"verdicts": [{"id": 1, "supported": True}]})).grounded)
        llm = ClaimLLM({"verdicts": [{"id": "1", "supported": True}, {"id": "2", "supported": "false"}]})
        self.assertFalse(self.finalize(llm).grounded)

    def test_fast_model_failure_falls_back_to_primary(self):
        llm = ClaimLLM(self.BOTH_OK, fail_fast=True)
        self.assertTrue(self.finalize(llm).grounded)
        self.assertEqual(llm.calls, [True, False])

    def test_check_failure_refuses(self):
        self.assertFalse(self.finalize(ClaimLLM(fail_all=True)).grounded)

    def test_off_by_config_makes_no_call(self):
        llm = ClaimLLM(fail_all=True)
        rr = RetrievalResult("q", ["q"], [], [], "ctx", ["Book p.1", "Book p.2"], 0.9, True)
        answer = ChatEngine(Config(), retriever=SimpleNamespace(), llm=llm)._finalize("q", "q", self.ANSWER, rr)
        self.assertEqual((answer.grounded, llm.calls), (True, []))

    def test_latex_is_shown_to_the_checker_as_plain_math(self):
        from bookrag.chat.engine import _plain_math
        self.assertEqual(_plain_math("$$ \\eta = 1 - \\frac{T_{\\text{cold}}}{T_{\\text{hot}}} $$"),
                         "eta = 1 - (T_cold)/(T_hot)")
        self.assertEqual(_plain_math("$\\Delta G^{\\circ} = -RT \\ln K$"), "Delta G^circ = -RT ln K")
        self.assertEqual(_plain_math("No maths here."), "No maths here.")

        seen = {}

        class RecordingLLM(ClaimLLM):
            def complete_json(self, system, user, fast=False, temperature=None):
                seen["user"] = user
                return super().complete_json(system, user, fast, temperature)

        answer = "$\\eta = 1 - \\frac{T_{\\text{cold}}}{T_{\\text{hot}}}$ [1]."
        self.finalize(RecordingLLM({"verdicts": [{"id": 1, "supported": True}]}), answer)
        self.assertIn("1. eta = 1 - (T_cold)/(T_hot)", seen["user"])
        self.assertNotIn("\\frac", seen["user"].split("STATEMENTS", 1)[1])

    def test_formula_block_is_checked_with_its_lead_in_and_citation(self):
        from bookrag.chat.engine import _statements
        answer = ("The formula for the array gain when non-excited elements are match-terminated is given by:\n\n"
                  "$$ E_{\\text{array}} = E_a P V^+ $$\n\n"
                  "This formula follows from superposition for match-terminated elements [5].")
        self.assertEqual(_statements(answer), [
            ("The formula for the array gain when non-excited elements are match-terminated is given by: "
             "$$ E_{\\text{array}} = E_a P V^+ $$", [5]),
            ("This formula follows from superposition for match-terminated elements.", [5])])
        # An uncited paragraph with no later citation falls back to the previous one.
        self.assertEqual(_statements("Heat is energy in transit [1].\n\nIt always flows from hot to cold."),
                         [("Heat is energy in transit.", [1]), ("It always flows from hot to cold.", [1])])
        # A marker on its own line covers the statements before it.
        self.assertEqual(_statements("$$ S = k \\ln W $$\n[2]"), [("$$ S = k \\ln W $$", [2])])

    def test_statement_grouping(self):
        from bookrag.chat.engine import _statements
        got = _statements("Heat flows from hot bodies. It needs a temperature gradient [2].\n\n"
                          "## Notes\nWork is done by the gas [1][3].")
        self.assertEqual(got, [("Heat flows from hot bodies.", [2]),
                               ("It needs a temperature gradient.", [2]),
                               ("Work is done by the gas.", [1, 3])])


class HealthTests(unittest.TestCase):
    def test_openai_server_listing_no_models_is_unconfirmed(self):
        from bookrag.llm.openai_client import OpenAICompatClient
        client = OpenAICompatClient(base_url="http://x", model="m")
        client.available_models = lambda: []
        h = client.health_check()
        self.assertEqual((h["primary_ok"], h["models_listed"]), (False, False))
        client.available_models = lambda: ["m"]
        self.assertTrue(client.health_check()["primary_ok"])


class OpenAICompatTests(unittest.TestCase):
    def client(self, **kw):
        from bookrag.llm.openai_client import OpenAICompatClient
        return OpenAICompatClient(base_url="http://x/v1", model="m", **kw)

    def test_thinking_switch_and_json_mode_payload(self):
        p = self.client(think=False, json_mode=False)._payload([], None, None, None, False, True)
        self.assertEqual(p["chat_template_kwargs"], {"enable_thinking": False})
        self.assertNotIn("response_format", p)
        p = self.client()._payload([], None, None, None, False, True)
        self.assertNotIn("chat_template_kwargs", p)
        self.assertEqual(p["response_format"], {"type": "json_object"})

    def test_think_block_stripped_from_stream_even_when_tags_split(self):
        from bookrag.llm.client import ThinkStripper
        s = ThinkStripper()
        pieces = ["<thi", "nk>\nreasoning here", " more</th", "ink>\n\nFinal ", "answer [1]."]
        self.assertEqual(("".join(s.feed(p) for p in pieces) + s.flush()).strip(), "Final answer [1].")
        s = ThinkStripper()
        self.assertEqual(s.feed("a < b and x <th") + s.feed("en y") + s.flush(), "a < b and x <then y")

    def test_stream_and_error_paths_over_http(self):
        import httpx
        from bookrag.llm.client import LLMError
        pieces = ["<thi", "nk>\nhmm", "</think>", "\n\nHeat is ", "energy [1]."]
        sse = "".join(f"data: {json.dumps({'choices': [{'delta': {'content': p}}]})}\n\n"
                      for p in pieces) + "data: [DONE]\n\n"
        seen = {}

        def handler(request):
            body = json.loads(request.content)
            seen.update(body)
            if body["messages"][0]["content"] == "too long":
                return httpx.Response(400, json={"message": "maximum context length is 8192 tokens"})
            return httpx.Response(200, content=sse.encode(),
                                  headers={"content-type": "text/event-stream"})

        c = self.client(think=False)
        c._client = httpx.Client(transport=httpx.MockTransport(handler))
        self.assertEqual("".join(c.stream_chat([{"role": "user", "content": "q"}])),
                         "Heat is energy [1].")
        self.assertEqual(seen["chat_template_kwargs"], {"enable_thinking": False})
        with self.assertRaisesRegex(LLMError, "8192"):
            c.chat([{"role": "user", "content": "too long"}])
        with self.assertRaisesRegex(LLMError, "8192"):
            list(c.stream_chat([{"role": "user", "content": "too long"}]))

    def test_config_wiring(self):
        from bookrag.llm.client import _openai_from_config
        cfg = Config({"llm": {"openai": {"base_url": "http://x/v1", "model": "m",
                                         "enable_thinking": False, "json_mode": False}}})
        c = _openai_from_config(cfg)
        self.assertEqual((c.think, c.json_mode), (False, False))

    def test_no_ollama_unload_sent_to_remote_server(self):
        from bookrag.chat.engine import ChatEngine
        llm = self.client()
        engine = ChatEngine(Config(), retriever=SimpleNamespace(_llm=None), llm=llm)
        with patch("bookrag.memory.ollama_unload") as unload:
            info = engine.use_model("other")
        unload.assert_not_called()
        self.assertEqual((llm.primary, info["released_previous"]), ("other", False))


class JobTests(unittest.TestCase):
    def test_cancel_done_and_failed(self):
        import time
        from bookrag.jobs import Cancelled, Job

        def loop(progress, should_cancel):
            while not should_cancel():
                time.sleep(0.01)
            raise Cancelled("stopped")

        job = Job("loop", loop).start()
        job.cancel()
        self.assertTrue(job.wait(5))
        self.assertEqual(job.status, "cancelled")

        job = Job("ok", lambda x, progress, should_cancel: x * 2, 21).start()
        job.wait(5)
        self.assertEqual((job.status, job.result), ("done", 42))

        def boom(progress, should_cancel):
            raise ValueError("bad input")

        job = Job("boom", boom).start()
        job.wait(5)
        self.assertEqual(job.status, "failed")
        self.assertIn("bad input", job.error)

    def test_generator_cancel_keeps_accepted_questions(self):
        chunk = Chunk("c1", "Thermal equilibrium means the same temperature.", "", "b1", "Book")
        rr = SimpleNamespace(context=chunk.text, results=[ScoredChunk(chunk, 1)], grounded=True)
        retriever = SimpleNamespace(retrieve=lambda *a, **k: rr, store=SimpleNamespace(chunks=[chunk]))
        llm = ScriptedLLM(*[{"questions": [{"text": f"Define {w} energy.", "answer": "x"}]}
                            for w in ("kinetic", "potential", "internal")])
        gen = PaperGenerator(Config({"grounding": {"verify_questions": False}}),
                             retriever=retriever, llm=llm)
        paper = gen.generate(Blueprint(sections=[SectionSpec("A", count=3)], topics=["T"]),
                             progress=lambda m: None, should_cancel=lambda: llm.calls >= 1)
        self.assertTrue(paper.report.cancelled)
        self.assertEqual((len(paper.questions), paper.report.requested), (1, 1))


class ResumeTests(IndexConfigMixin, unittest.TestCase):
    def test_cancelled_build_resumes_without_reembedding(self):
        from bookrag.index.builder import build_index, cached_book_count
        from bookrag.jobs import Cancelled

        class CountingEmbedder(FakeEmbedder):
            def __init__(self):
                self.texts = []

            def embed_passages(self, texts, show_progress=False):
                self.texts.extend(texts)
                return super().embed_passages(texts)

        emb = CountingEmbedder()
        body = lambda w: f"# Chapter {w}\n" + f"The {w} principle explains energy transfer. " * 60
        with tempfile.TemporaryDirectory() as d, \
                patch("bookrag.index.builder.embedder_from_config", return_value=emb), \
                patch("bookrag.index.builder.arbiter_from_config", return_value=Mock()):
            cfg = self.cfg(d)
            books = Path(d) / "books"
            books.mkdir()
            (books / "alpha.md").write_text(body("alpha"))
            (books / "beta.md").write_text(body("beta"))

            # Cancel as soon as the first book has been embedded.
            with self.assertRaises(Cancelled):
                build_index(cfg, progress=lambda m: None, should_cancel=lambda: bool(emb.texts))
            self.assertFalse(Store.exists(cfg.index_dir))
            self.assertEqual(cached_book_count(cfg), 1)

            first_run = list(emb.texts)
            emb.texts.clear()
            m = build_index(cfg, progress=lambda m: None)
            self.assertEqual(sorted(b["title"] for b in m["books"]), ["alpha", "beta"])
            self.assertFalse(set(first_run) & set(emb.texts), "alpha was embedded twice")
            self.assertEqual(cached_book_count(cfg), 0)


class RemainingShortcomingTests(unittest.TestCase):
    """The 'Remaining shortcomings' section of audit/REPORT.md."""

    RR = RetrievalResult("q", ["q"], [], [], "ctx", ["Book p.1"], 0.9, True,
                         passages=["Heat is energy in transit."])

    def finalize(self, llm, answer):
        return ChatEngine(ClaimCheckTests.CFG, retriever=SimpleNamespace(), llm=llm)._finalize(
            "q", "q", answer, self.RR)

    def test_short_cited_claim_is_checked(self):
        llm = ClaimLLM({"verdicts": [{"id": 1, "supported": False}]})
        self.assertFalse(self.finalize(llm, "Einstein invented it [1].").grounded)
        self.assertEqual(len(llm.calls), 1)

    def test_conflicting_duplicate_verdicts_refuse_but_agreeing_ones_pass(self):
        answer = "Heat is energy in transit between bodies [1]."
        conflicting = ClaimLLM({"verdicts": [{"id": 1, "supported": False}, {"id": 1, "supported": True}]})
        self.assertFalse(self.finalize(conflicting, answer).grounded)
        agreeing = ClaimLLM({"verdicts": [{"id": 1, "supported": True}, {"id": 1, "supported": True}]})
        self.assertTrue(self.finalize(agreeing, answer).grounded)

    def test_resolve_sources_accepts_odd_shapes(self):
        from bookrag.paper.generator import _resolve_sources
        chunks = [Chunk(f"c{i}", "t", "", "b", "B") for i in range(1, 5)]
        rr = SimpleNamespace(results=[ScoredChunk(c, 1) for c in chunks])
        ids = lambda v: [c.id for c in _resolve_sources(v, rr)]
        self.assertEqual(ids(2), ["c2"])
        self.assertEqual(ids("1, 3"), ["c1", "c3"])
        self.assertEqual(ids({"x": 1}), ["c1", "c2", "c3"])
        self.assertEqual(ids([True, None, "4", 9, 4]), ["c4"])

    def test_blueprint_problems_block_generation(self):
        bp = Blueprint(sections=[SectionSpec("A", count=2), SectionSpec("a", count=1)])
        self.assertTrue(any("two sections" in p for p in bp.problems()))
        self.assertTrue(Blueprint(sections=[SectionSpec("A", count=2, attempt=3)]).problems())
        self.assertTrue(Blueprint().problems())
        self.assertEqual(Blueprint(sections=[SectionSpec("A"), SectionSpec("B")]).problems(), [])
        with self.assertRaises(ValueError):
            PaperGenerator(Config(), retriever=SimpleNamespace(), llm=ScriptedLLM()).generate(
                bp, progress=lambda m: None)

    def test_export_title_with_slash_and_docx_attempt_rule(self):
        import docx
        from bookrag.paper.export import safe_stem
        self.assertEqual(safe_stem("physics i/ii: midterm"), "physics_i_ii_midterm")
        self.assertEqual(safe_stem("../../etc"), "etc")
        self.assertEqual(safe_stem("///"), "paper")
        self.assertEqual(safe_stem("CON"), "CON_")
        bp = Blueprint(sections=[SectionSpec("A", count=3, marks_each=2, attempt=1)])
        paper = Paper(bp, questions=[question(number=i) for i in (1, 2, 3)])
        with tempfile.TemporaryDirectory() as d:
            files = export_paper(paper, Path(d), "paper_Physics I/II", include_key=False)
            self.assertEqual(len(files), 3)
            self.assertTrue(all(f.parent == Path(d) for f in files))
            text = " ".join(p.text for p in docx.Document(
                str(next(f for f in files if f.suffix == ".docx"))).paragraphs)
        self.assertIn("answer 1 of 3", text)

    def test_router_model_switch_follows_active_backend(self):
        from bookrag.llm.router import FallbackClient
        p = SimpleNamespace(primary="p1", fast="p1", host="h")
        f = SimpleNamespace(primary="f1", fast="f1", host="h")
        router = FallbackClient(p, f, label_primary="openai", label_fallback="ollama")
        router.primary = "p2"
        self.assertEqual((p.primary, router.primary), ("p2", "p2"))
        router.last_used = "ollama"
        router.primary = "f2"
        self.assertEqual((f.primary, router.primary, p.primary), ("f2", "f2", "p2"))

    def test_connect_timeout_is_short_and_separate(self):
        from bookrag.llm.client import OllamaClient, client_from_config
        self.assertEqual(OllamaClient(connect_timeout_s=5)._client.timeout.connect, 5)
        client = client_from_config(Config({"llm": {
            "provider": "openai", "connect_timeout_s": 3, "timeout_s": 600,
            "fallback": {"enabled": False},
            "openai": {"base_url": "http://x/v1", "model": "m"}}}))
        self.assertEqual((client._client.timeout.connect, client._client.timeout.read), (3, 600))

    def test_expansion_failure_reaches_the_result(self):
        from bookrag.retrieve.pipeline import Retriever

        class Emb:
            def embed_queries(self, variants):
                return np.ones((len(variants), 2), np.float32)

        with tempfile.TemporaryDirectory() as d:
            store = Store(Path(d))
            c = Chunk("a::00001", "Heat is energy.", "Heat is energy.", "a", "A", ordinal=1)
            store.save([c], np.ones((1, 2)), {"books": [{"book_id": "a"}]})
            llm = Mock()
            llm.complete_json.side_effect = RuntimeError("server down")
            llm.complete.side_effect = RuntimeError("server down")
            retriever = Retriever(Config({"retrieval": {"multi_query": True, "hyde": True}}),
                                  store=store, embedder=Emb(), reranker=False, llm=llm,
                                  arbiter=Mock())
            rr = retriever.retrieve("heat", expand=True)
        self.assertEqual(len(rr.warnings), 2)
        self.assertTrue(rr.grounded)

    def test_sources_list_only_cited_and_merged(self):
        from bookrag.chat.engine import Answer
        a = Answer("q", "q", "x [1] y [3] z [4]",
                   citations=["B > Ch1, p.1", "B > Ch1, p.2", "B > Prologue, pp.12-14",
                              "B > Prologue, pp.12-14"],
                   used_citations=[1, 3, 4])
        self.assertEqual(a.sources(), ["[1] B > Ch1, p.1", "[3][4] B > Prologue, pp.12-14"])
        self.assertIn("[3][4] B > Prologue", a.formatted())
        self.assertEqual(Answer("q", "q", "refused", grounded=False).sources(), [])


class QualityToolingTests(unittest.TestCase):
    def test_mcq_options_differing_only_in_spacing_are_duplicates(self):
        q = question(qtype="mcq", options=["dS = dQ/T", "dS=dQ/T", "dS = T/dQ", "dS = 0"],
                     correct_option="dS = 0")
        llm = ScriptedLLM()
        self.assertFalse(verify_question(llm, q, "ctx").answerable)
        self.assertEqual(llm.calls, 0)

    def test_mcq_distractor_that_is_the_key_minus_a_subscript_is_rejected(self):
        q = question(qtype="mcq", options=["dS = dQ / T", "dS = dQ_rev / T", "dS = T / dQ", "dS = dW / T"],
                     correct_option="dS = dQ_rev / T")
        llm = ScriptedLLM()
        verdict = verify_question(llm, q, "ctx")
        self.assertFalse(verdict.answerable)
        self.assertIn("subscript", verdict.reason)
        self.assertEqual(llm.calls, 0)
        unicode_q = question(qtype="mcq", options=["T₁ + T₂", "T + T", "T₁ - T₂", "2T₁"], correct_option="T₁ + T₂")
        self.assertFalse(verify_question(ScriptedLLM(), unicode_q, "ctx").answerable)
        # Swapping subscripts is a legitimate distractor and must still reach the model.
        swapped = question(qtype="mcq",
                           options=["η = 1 - T_cold/T_hot", "η = 1 - T_hot/T_cold", "η = T_cold/T_hot", "η = 1"],
                           correct_option="η = 1 - T_cold/T_hot")
        llm = ScriptedLLM({"answerable": True, "confidence": 0.9},
                          {"options": [{"index": i, "true": i == 0} for i in range(4)]})
        self.assertTrue(verify_question(llm, swapped, "ctx").answerable)
        self.assertEqual(llm.calls, 2)

    def test_keyword_matching_ignores_notation_and_punctuation(self):
        from bookrag.evaluate.gold import keyword_present
        answer = ("The efficiency is $$ \\eta = 1 - \\frac{T_{\\text{cold}}}{T_{\\text{hot}}} $$, "
                  "found with the Generalized Scattering Matrix (GSM) method.")
        for keyword in ["1 - T_cold / T_hot", "T_cold", "eta", "GSM method"]:
            self.assertTrue(keyword_present(keyword, answer), keyword)
        for keyword in ["T_warm", "Etchu Seto-yaki", "x", ""]:
            self.assertFalse(keyword_present(keyword, answer), keyword)
        self.assertTrue(keyword_present("P", "E_array = E_a P V^+"))
        self.assertFalse(keyword_present("P", "Pressure rises"))
        # Short symbols survive notation differences but still need a whole token.
        for keyword in ["Ea", "V+", "P"]:
            self.assertTrue(keyword_present(keyword, "$$ E_{\\text{array}} = E_a P V^+ $$"), keyword)
        self.assertFalse(keyword_present("Ea", "Earray is large"))

    def test_config_overrides(self):
        from bookrag.config import parse_overrides
        base = Config({"retrieval": {"max_context_tokens": 4500}})
        overrides = parse_overrides(["retrieval.max_context_tokens=12000", "retrieval.hyde=true",
                                     "llm.openai.model=Qwen/Qwen3"])
        cfg = base.with_overrides(overrides)
        self.assertEqual((cfg.get("retrieval.max_context_tokens"), cfg.get("retrieval.hyde"),
                          cfg.get("llm.openai.model")), (12000, True, "Qwen/Qwen3"))
        self.assertEqual(base.get("retrieval.max_context_tokens"), 4500)
        for bad in ["novalue", "=3", "a..b=1"]:
            with self.assertRaises(ValueError):
                parse_overrides([bad])

    def test_gold_file_validation(self):
        from bookrag.evaluate.gold import load_gold
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "gold.jsonl"

            def write(*lines):
                path.write_text("\n".join(l if isinstance(l, str) else json.dumps(l) for l in lines))
                return path

            cases = load_gold(write("# comment", {"question": "What is heat?", "pages": [1],
                                                  "keywords": ["energy"], "reviewed": True},
                                    "", {"question": "Cake?", "answerable": False}))
            self.assertEqual([c.answerable for c in cases], [True, False])
            for bad in [{"question": ""}, {"question": "q", "pages": ["1"]},
                        {"question": "q", "answerable": "yes"}, {"question": "q", "extra": 1},
                        {"question": "q", "reviewed": "false"}, "not json", "# only a comment"]:
                with self.assertRaises(ValueError):
                    load_gold(write(bad))

    def test_gold_run_scores_hits_refusals_and_errors(self):
        from bookrag.chat.engine import Answer
        from bookrag.evaluate.gold import GoldCase, run_gold_eval, summary_text
        chunk = Chunk("b::00001", "Heat is energy in transit.", "", "b", "Thermal",
                      page_start=2, page_end=3, ordinal=1)
        rr = RetrievalResult("q", ["q"], [ScoredChunk(chunk, 1)], ["b::00001"], "ctx",
                             [chunk.citation()], 0.9, True)

        def ask(q, book_ids=None):
            if "cake" in q:
                return Answer(q, q, "I can't answer that.", grounded=False, retrieval=rr)
            if "boom" in q:
                raise RuntimeError("server down")
            return Answer(q, q, "Heat is energy [1].", [chunk.citation()], [1], True, 0.9, rr)

        store = SimpleNamespace(books=lambda: [{"book_id": "b", "title": "Thermal"}])
        engine = SimpleNamespace(ask=ask, reset=lambda: None, retriever=SimpleNamespace(store=store))
        cases = [GoldCase("What is heat?", pages=[3], keywords=["energy", "entropy"], book="thermal"),
                 GoldCase("How to bake a cake?", answerable=False),
                 GoldCase("boom?"),
                 GoldCase("What is heat?", book="nonexistent")]
        res = run_gold_eval(Config(), cases, progress=lambda m: None, engine=engine)
        self.assertEqual((res["answer_rate"], res["cited_page_hit"], res["context_page_hit"],
                          res["keyword_recall"], res["correct_refusal_rate"], res["errors"]),
                         (1.0, 1.0, 1.0, 0.5, 1.0, 2))
        self.assertIn("correctly refused 100%", summary_text(res))

    def test_draft_round_robins_books_and_adds_controls(self):
        from bookrag.evaluate.gold import OFF_TOPIC, case_from_dict, draft_gold
        chunks = [Chunk(f"{b}::{i:05d}", f"The {b} principle number {i} explains energy transfer. " * 20,
                        "", b, b.title(), page_start=i, page_end=i, ordinal=i, token_count=200)
                  for b in ("alpha", "beta") for i in range(1, 6)]
        store = SimpleNamespace(chunks=chunks, books=lambda: [{"book_id": "alpha", "title": "Alpha"},
                                                              {"book_id": "beta", "title": "Beta"}])
        llm = ScriptedLLM(*[{"question": f"What does principle {i} explain?",
                             "keywords": ["energy", "Energy", "not in passage"]} for i in range(4)])
        cases = draft_gold(Config(), n=4, progress=lambda m: None, llm=llm, store=store)
        answerable = [c for c in cases if c["answerable"]]
        self.assertEqual(sorted(c["book"] for c in answerable), ["Alpha", "Alpha", "Beta", "Beta"])
        self.assertTrue(all(c["keywords"] == ["energy"] for c in answerable))
        self.assertEqual(len(cases) - len(answerable), len(OFF_TOPIC))
        for c in cases:
            case_from_dict(c, "draft")          # every draft line is a valid gold case


    def test_draft_strips_or_drops_references_to_the_source(self):
        from bookrag.evaluate.gold import draft_gold
        chunks = [Chunk(f"a::{i:05d}", "The alpha principle explains energy transfer. " * 20, "", "a",
                        "Alpha", page_start=i, page_end=i, ordinal=i, token_count=200) for i in range(1, 4)]
        store = SimpleNamespace(chunks=chunks, books=lambda: [{"book_id": "a", "title": "Alpha"}])
        llm = ScriptedLLM(
            {"question": "What does the principle explain, as described in the passage?", "keywords": ["energy"]},
            {"question": "According to the text, what is transferred?", "keywords": ["energy"]},
            {"question": "What does this passage say?", "keywords": ["energy"]})
        cases = [c for c in draft_gold(Config(), n=3, progress=lambda m: None, llm=llm, store=store)
                 if c["answerable"]]
        self.assertEqual([c["question"] for c in cases],
                         ["What does the principle explain?", "What is transferred?"])


class JsonRepairTests(unittest.TestCase):
    def test_latex_inside_json_strings_is_kept_literally(self):
        from bookrag.llm.client import parse_json
        # Verbatim shape of a live claim-check reply that used to fail to parse.
        raw = ('```json\n{"verdicts": [{"id": 1, "supported": true, "reason": '
               '"states $ \\eta = 1 - \\frac{T_{\\text{cold}}}{T_{\\text{hot}}} $"}]}\n```')
        parsed = parse_json(raw)
        self.assertEqual(parsed["verdicts"][0]["reason"],
                         "states $ \\eta = 1 - \\frac{T_{\\text{cold}}}{T_{\\text{hot}}} $")

    def test_genuine_escapes_are_preserved(self):
        from bookrag.llm.client import parse_json
        self.assertEqual(parse_json('{"a": "x \\\\frac y", "b": "line1\\nline2"}'),
                         {"a": "x \\frac y", "b": "line1\nline2"})
        self.assertEqual(parse_json('{"a": "\\Delta G = -RT \\ln K", "b": "quote \\" ok", "c": "tab\\t1"}'),
                         {"a": "\\Delta G = -RT \\ln K", "b": 'quote " ok', "c": "tab\t1"})
        self.assertEqual(parse_json('{"a": "caf\\u00e9 \\beta"}'), {"a": "café \\beta"})
        self.assertIsNone(parse_json("not JSON"))

    def test_latex_that_is_valid_json_is_not_turned_into_control_characters(self):
        from bookrag.llm.client import parse_json
        # Valid JSON as written: \f, \t and \r are legal escapes.
        parsed = parse_json('{"a": "$\\frac{1}{2}$", "b": "$\\theta$ and $\\rho$", "c": "one.\\nTwo"}')
        self.assertEqual(parsed, {"a": "$\\frac{1}{2}$", "b": "$\\theta$ and $\\rho$", "c": "one.\nTwo"})
        self.assertFalse(any(ch in value for value in parsed.values() for ch in "\x08\x0c\r\t"))

    def test_openai_complete_json_accepts_a_latex_reply(self):
        import httpx
        from bookrag.llm.openai_client import OpenAICompatClient
        content = '```json\n{"verdicts": [{"id": 1, "supported": true, "reason": "$\\eta = 1 - \\frac{a}{b}$"}]}\n```'
        client = OpenAICompatClient(base_url="http://x/v1", model="m")
        client._client = httpx.Client(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"choices": [{"message": {"content": content}}]})))
        self.assertEqual(client.complete_json("s", "u")["verdicts"][0]["reason"],
                         "$\\eta = 1 - \\frac{a}{b}$")


class FailoverTests(unittest.TestCase):
    @staticmethod
    def with_cause(exc, cause):
        exc.__cause__ = cause
        return exc

    def test_fails_over_on_outages_only_and_names_both_errors(self):
        import httpx
        from bookrag.llm.client import LLMError, LLMReplyError
        from bookrag.llm.router import FallbackClient
        request = httpx.Request("POST", "http://x")
        primary, fallback = Mock(), Mock()
        fallback.complete_json.return_value = {"ok": True}
        router = FallbackClient(primary, fallback, label_primary="openai", label_fallback="ollama")

        primary.complete_json.side_effect = LLMReplyError("Model did not return valid JSON")
        with self.assertRaises(LLMReplyError):
            router.complete_json("s", "u")
        primary.complete_json.side_effect = self.with_cause(
            LLMError("400 prompt too long"),
            httpx.HTTPStatusError("400", request=request, response=httpx.Response(400, request=request)))
        with self.assertRaises(LLMError):
            router.complete_json("s", "u")
        fallback.complete_json.assert_not_called()

        for outage in (self.with_cause(LLMError("down"), httpx.ConnectError("refused")),
                       self.with_cause(LLMError("busy"), httpx.HTTPStatusError(
                           "503", request=request, response=httpx.Response(503, request=request))),
                       RuntimeError("offline")):
            primary.complete_json.side_effect = outage
            self.assertEqual(router.complete_json("s", "u"), {"ok": True})

        primary.complete_json.side_effect = self.with_cause(LLMError("down"), httpx.ConnectError("refused"))
        fallback.complete_json.side_effect = LLMError("Ollama request failed: connection refused")
        with self.assertRaisesRegex(LLMError, r"openai failed \(down\), and the fallback \(ollama\) also failed"):
            router.complete_json("s", "u")

    def test_stream_status_errors_are_classified(self):
        from bookrag.llm.client import LLMError, should_fail_over
        bad_request, overloaded = LLMError("400"), LLMError("503")
        bad_request.status_code, overloaded.status_code = 400, 503
        self.assertEqual((should_fail_over(bad_request), should_fail_over(overloaded)), (False, True))


class SectionCitationTests(unittest.TestCase):
    @staticmethod
    def line(text, section, heading=False, chapter="Chapter 2", page=2):
        return dict(text=text, chapter=chapter, section=section, page=page, is_heading=heading)

    def test_chunk_spanning_sections_cites_the_whole_range(self):
        from bookrag.ingest.chunker import chunk_book
        s1, s2, s3 = "2.1 Definition of Entropy", "2.2 The Second Law", "2.3 The Carnot Cycle"
        lines = [self.line(s1, s1, True), self.line("Entropy S is defined by dS = dQ_rev / T. " * 5, s1),
                 self.line(s2, s2, True), self.line("Entropy of an isolated system never decreases. " * 5, s2),
                 self.line(s3, s3, True), self.line("The Carnot cycle has four reversible steps. " * 5, s3)]
        [chunk] = chunk_book(lines, "b", "Book", chunk_tokens=700, overlap_tokens=0, min_tokens=5)
        self.assertEqual(chunk.section, f"{s1} – {s3}")
        self.assertIn(s3, chunk.citation())
        self.assertIn(s3, chunk.embed_text)

    def test_single_section_and_split_sections_keep_their_own_labels(self):
        from bookrag.ingest.chunker import chunk_book
        s1, s2 = "1.1 Zeroth Law", "1.2 First Law"
        lines = [self.line(s1, s1, True, "Chapter 1", 1),
                 self.line("Thermal equilibrium is transitive between bodies. " * 40, s1, chapter="Chapter 1", page=1),
                 self.line(s2, s2, True, "Chapter 1", 2),
                 self.line("Energy is conserved in every closed process. " * 40, s2, chapter="Chapter 1", page=2)]
        chunks = chunk_book(lines, "b", "Book", chunk_tokens=150, overlap_tokens=0, min_tokens=5)
        self.assertEqual(chunks[0].section, s1)
        self.assertEqual(chunks[-1].section, s2)
        self.assertTrue(all(c.token_count <= 150 for c in chunks))


class OcrTests(IndexConfigMixin, unittest.TestCase):
    def test_unreadable_pdf_is_ocrd_when_tool_present_and_explained_when_not(self):
        from bookrag.index import builder
        from bookrag.schemas import Page
        readable = [Page(i, "", [dict(text="A readable sentence about heat engines and entropy. " * 3,
                                      size=11.0, y=float(j), page=i) for j in range(40)])
                    for i in range(1, 4)]
        unreadable = [Page(1, "", [dict(text="x", size=11.0, y=0.0, page=1)])]

        def fake_load(p, drop_furniture=True):
            meta = {"title": "scan", "author": "", "n_pages": 3, "toc": [], "path": str(p),
                    "book_id": "scan-" + Path(p).stem[:6]}
            return (readable, meta) if builder.OCR_DIR in str(p) else (unreadable, meta)

        def fake_run(cmd, **kwargs):
            Path(cmd[-1]).write_bytes(b"%PDF-ocr")
            return SimpleNamespace(returncode=0)

        with tempfile.TemporaryDirectory() as d, \
                patch("bookrag.index.builder.embedder_from_config", return_value=FakeEmbedder()), \
                patch("bookrag.index.builder.arbiter_from_config", return_value=Mock()):
            cfg = self.cfg(d)
            books = Path(d) / "books"
            books.mkdir()
            (books / "scan.pdf").write_bytes(b"%PDF-fake")
            with patch.object(builder, "load_document", side_effect=fake_load), \
                    patch.object(builder.shutil, "which", return_value="/opt/bin/ocrmypdf"), \
                    patch.object(builder.subprocess, "run", side_effect=fake_run) as run:
                m = builder.build_index(cfg, progress=lambda s: None)
            self.assertEqual(run.call_count, 1)
            self.assertEqual([(b["title"], b["path"], b["book_id"]) for b in m["books"]],
                             [("scan", str(books / "scan.pdf"), "scan-scan")])

            log: list[str] = []
            with patch.object(builder, "load_document", side_effect=fake_load), \
                    patch.object(builder.shutil, "which", return_value=None):
                with self.assertRaises(ValueError):
                    builder.build_index(cfg, progress=log.append)
            self.assertTrue(any("isn't installed" in line for line in log))


class AppAccessTests(unittest.TestCase):
    APP = str(Path(__file__).resolve().parent.parent / "app.py")

    def app(self, address, password=None, headers=None, ip="203.0.113.7"):
        import os
        import streamlit
        from contextlib import ExitStack
        from streamlit.runtime.context import ContextProxy
        from streamlit.testing.v1 import AppTest
        real_get_option = streamlit.get_option
        streamlit.cache_resource.clear()
        env = {k: v for k, v in os.environ.items() if k != "BOOKRAG_APP_PASSWORD"}
        if password:
            env["BOOKRAG_APP_PASSWORD"] = password
        stack = ExitStack()
        stack.enter_context(patch.dict(os.environ, env, clear=True))
        stack.enter_context(patch("streamlit.get_option", side_effect=lambda key: (
            address if key == "server.address" else real_get_option(key))))
        stack.enter_context(patch.object(ContextProxy, "headers", new_callable=PropertyMock,
                                         return_value=headers or {}))
        stack.enter_context(patch.object(ContextProxy, "ip_address", new_callable=PropertyMock,
                                         return_value=ip))
        self.addCleanup(stack.close)
        return AppTest.from_file(self.APP)

    def test_made_up_forwarding_headers_do_not_escape_the_lockout(self):
        # A direct connection sends a new X-Forwarded-For with every guess.
        import itertools
        from streamlit.runtime.context import ContextProxy
        at = self.app("0.0.0.0", password="s3cret", ip="203.0.113.7").run(timeout=60)
        fake = itertools.count(1)
        with patch.object(ContextProxy, "headers", new_callable=PropertyMock,
                          side_effect=lambda: {"x-forwarded-for": f"10.0.0.{next(fake)}"}):
            for _ in range(5):
                at = self.sign_in(at, "wrong")
            at = self.sign_in(at, "s3cret")
        self.assertTrue(any("Too many wrong passwords" in e.value for e in at.error))

    @staticmethod
    def gate_messages(at):
        # Past the gate the app may still report other things, e.g. an unreachable LLM.
        return [e.value for e in at.error if "password" in e.value.lower() or "proxy" in e.value]

    def sign_in(self, at, password):
        at.text_input[0].set_value(password)
        return at.button[0].click().run(timeout=60)

    def test_local_address_behind_a_tunnel_still_needs_a_password(self):
        # cloudflared/nginx on this machine connect from 127.0.0.1; only their headers tell.
        at = self.app("127.0.0.1").run(timeout=60)
        self.assertFalse(self.gate_messages(at))
        self.assertFalse(any(t.label == "Password" for t in at.text_input))
        at = self.app("127.0.0.1", headers={"Cf-Connecting-Ip": "198.51.100.4"}).run(timeout=60)
        self.assertTrue(any("proxy or tunnel" in e.value for e in at.error), [e.value for e in at.error])
        self.assertEqual(len(at.tabs), 0)
        at = self.app("127.0.0.1", headers={"X-Forwarded-For": "198.51.100.4"}).run(timeout=60)
        self.assertTrue(any("proxy or tunnel" in e.value for e in at.error))

    def test_repeated_wrong_passwords_lock_that_client_out(self):
        at = self.app("127.0.0.1", password="s3cret", ip="127.0.0.1",      # via the tunnel
                      headers={"cf-connecting-ip": "198.51.100.4"}).run(timeout=60)
        for _ in range(5):
            at = self.sign_in(at, "wrong")
            self.assertTrue(any("Wrong password" in e.value for e in at.error))
        at = self.sign_in(at, "s3cret")                       # right password, still locked
        self.assertTrue(any("Too many wrong passwords" in e.value for e in at.error))
        self.assertEqual(len(at.tabs), 0)

        # Another visitor through the same tunnel is not affected.
        from streamlit.testing.v1 import AppTest
        from streamlit.runtime.context import ContextProxy
        with patch.object(ContextProxy, "headers", new_callable=PropertyMock,
                          return_value={"cf-connecting-ip": "192.0.2.9"}):
            other = AppTest.from_file(self.APP).run(timeout=60)
            other = self.sign_in(other, "s3cret")
        self.assertFalse(self.gate_messages(other))
        self.assertTrue(other.session_state["authenticated"])

    def test_network_address_without_password_is_refused(self):
        at = self.app("0.0.0.0").run(timeout=60)
        self.assertTrue(any("no password is set" in e.value for e in at.error))
        self.assertEqual(len(at.tabs), 0)

    def test_password_gate_rejects_wrong_password(self):
        at = self.app("0.0.0.0", password="s3cret").run(timeout=60)
        self.assertEqual(len(at.tabs), 0)
        self.assertEqual(len(at.text_input), 1)
        at.text_input[0].set_value("wrong")
        at.button[0].click().run(timeout=60)
        self.assertTrue(any("Wrong password" in e.value for e in at.error))
        self.assertEqual(len(at.tabs), 0)
        self.assertFalse(at.session_state["authenticated"] if "authenticated" in at.session_state else False)


class SecurityHardeningTests(IndexConfigMixin, unittest.TestCase):
    EVIL = "evil.test"

    def render(self, text):
        from markdown_it import MarkdownIt
        return MarkdownIt("commonmark", {"html": False}).render(text)

    def test_client_address_trusts_forwarding_only_from_a_local_proxy(self):
        from bookrag.websafe import client_address
        spoof = {"x-forwarded-for": "1.2.3.4", "cf-connecting-ip": "5.6.7.8", "x-real-ip": "9.9.9.9"}
        self.assertEqual(client_address("203.0.113.7", spoof), "203.0.113.7")
        self.assertEqual(client_address("127.0.0.1", {"cf-connecting-ip": "198.51.100.4"}), "198.51.100.4")
        self.assertEqual(client_address("127.0.0.1", {"x-real-ip": "198.51.100.5"}), "198.51.100.5")
        # nginx appends the real peer; whatever the client put in front is ignored.
        self.assertEqual(client_address("127.0.0.1", {"x-forwarded-for": "6.6.6.6, 198.51.100.6"}),
                         "198.51.100.6")
        self.assertEqual(client_address("::1", {}), "::1")

    def test_answers_cannot_load_images_or_carry_links(self):
        from bookrag.websafe import safe_markdown
        u = f"https://{self.EVIL}/x.png?q=secret"
        attacks = [
            f"Heat is energy [1].\n\n![x]({u})",
            f"![a [nested] b]({u})",
            f"![a](https&#58;//{self.EVIL}/x.png)",
            f"![a](<{u}>)",
            f"![a](\n{u})",
            f"[click here]({u})",
            f"Source: <{u}>",
            f"![logo][r]\n\n[r]: {u}",
            f"![r]\n\n   [r]: {u}",
            f"> [r]: {u}\n\n![a][r]",
            f"- [r]: {u}\n\n![a][r]",
            f"1. [r]: {u}\n\n![a][r]",
            f"[r\nx]: {u}\n\n![a][r x]",
            f"[a\\]b]: {u}\n\n![z][a\\]b]",
            f"```a`b\n![x]({u})\n",                     # not a real fence opener
            f"````\ncode\n```\n![x]({u})\n````\n",     # ``` does not close ````
        ]
        for text in attacks:
            raw = self.render(text)
            self.assertTrue("<img" in raw or "<a " in raw or text.startswith("````"), text)
            safe = self.render(safe_markdown(text))
            self.assertNotIn("<img", safe, text)
            self.assertNotIn("<a ", safe, text)

    def test_ordinary_answers_render_exactly_as_before(self):
        from bookrag.websafe import safe_markdown
        benign = [
            "Energy is conserved [1][2]. The efficiency is $\\eta = 1 - T_c/T_h$ [3].",
            "[1] Heat flows from hot to cold.\n\n- item one [2]\n- item two",
            "**Carnot cycle** (see [4]): two isotherms and two adiabats.",
            "```python\nx = a[1](2)\n[r]: https://example.com\n```\nAfter the code [5].",
        ]
        for text in benign:
            self.assertEqual(self.render(safe_markdown(text)), self.render(text), text)
        code = "```python\nx = a[1](2)\n```"
        self.assertEqual(safe_markdown(code), code)

    def test_uploads_are_validated_and_never_replace_a_different_book(self):
        from bookrag.websafe import save_uploads
        up = lambda name, data: SimpleNamespace(name=name, getbuffer=lambda: memoryview(data))
        with tempfile.TemporaryDirectory() as d:
            dest = Path(d)
            saved, skipped = save_uploads([up("../../escape.pdf", b"one"), up(".env", b"x"),
                                           up("run.sh", b"x"), up("notes.PDF", b"two")],
                                          dest, {".pdf"})
            self.assertEqual(saved, ["escape.pdf", "notes.PDF"])
            self.assertEqual(len(skipped), 2)
            self.assertEqual(sorted(p.name for p in dest.iterdir()), ["escape.pdf", "notes.PDF"])

            saved, skipped = save_uploads([up("escape.pdf", b"one")], dest, {".pdf"})
            self.assertEqual((saved, skipped), ([], []))              # same file again: no-op
            saved, skipped = save_uploads([up("escape.pdf", b"other")], dest, {".pdf"})
            self.assertEqual(saved, [])
            self.assertIn("already exists", skipped[0])
            self.assertEqual((dest / "escape.pdf").read_bytes(), b"one")

    def test_index_never_unpickles(self):
        import pickle
        with tempfile.TemporaryDirectory() as d:
            store = Store(Path(d) / "index")
            chunks = [Chunk(f"b::{i:05d}", t, t, "b", "Book", ordinal=i)
                      for i, t in enumerate(["entropy and heat", "carnot engine efficiency",
                                             "second law of thermodynamics"])]
            store.save(chunks, np.eye(3, dtype=np.float32), {"books": []})
            gen = store._data_dir
            self.assertFalse((gen / "bm25.pkl").exists())

            marker = Path(d) / "pwned"

            class Payload:
                def __reduce__(self):
                    return (Path.write_text, (marker, "code ran"))

            (gen / "bm25.pkl").write_bytes(pickle.dumps(Payload()))
            loaded = Store(Path(d) / "index").load()
            hits = loaded.bm25_search("carnot efficiency", top_k=1)
            self.assertFalse(marker.exists())
            self.assertEqual(loaded.chunks[hits[0][0]].text, "carnot engine efficiency")

    def test_zip_bombs_are_refused_before_parsing(self):
        import zipfile
        from bookrag.ingest import loaders
        with tempfile.TemporaryDirectory() as d:
            bomb = Path(d) / "book.docx"
            with zipfile.ZipFile(bomb, "w", zipfile.ZIP_DEFLATED) as zf:
                for i in range(10):
                    zf.writestr(f"word/part{i}.xml", "0" * 100_000)
            self.assertLess(bomb.stat().st_size, 5_000)
            with self.assertRaisesRegex(ValueError, "unpacks to"):
                loaders.check_archive(bomb, max_bytes=500_000)
            with self.assertRaisesRegex(ValueError, "archive members"):
                loaders.check_archive(bomb, max_members=5)
            with patch.object(loaders, "MAX_UNZIPPED_BYTES", 500_000), \
                    patch("docx.Document", side_effect=AssertionError("parsed a bomb")):
                with self.assertRaisesRegex(ValueError, "unpacks to"):
                    loaders.load_document(bomb)
            fake = Path(d) / "book.epub"
            fake.write_bytes(b"not a zip at all")
            with self.assertRaisesRegex(ValueError, "not a valid"):
                loaders.load_document(fake)

    def test_one_unreadable_book_does_not_stop_the_build(self):
        from bookrag.index.builder import build_index
        with tempfile.TemporaryDirectory() as d, \
                patch("bookrag.index.builder.embedder_from_config", return_value=FakeEmbedder()), \
                patch("bookrag.index.builder.arbiter_from_config", return_value=Mock()):
            cfg = self.cfg(d)
            books = Path(d) / "books"
            books.mkdir()
            (books / "good.md").write_text("# Chapter 1\n" + "Heat is energy in transit. " * 80)
            (books / "broken.epub").write_bytes(b"\x00garbage")
            log = []
            m = build_index(cfg, progress=log.append)
            self.assertEqual([b["title"] for b in m["books"]], ["good"])
            self.assertTrue(any("broken.epub could not be read" in line for line in log), log)


class WeightsTests(unittest.TestCase):
    def test_offline_ok_modes(self):
        from bookrag.weights import offline_ok
        with patch("bookrag.weights.weights_cached", return_value=True):
            self.assertTrue(offline_ok("repo"))                 # auto + cached -> offline
            self.assertFalse(offline_ok("repo", "false"))       # explicitly disabled
        with patch("bookrag.weights.weights_cached", return_value=False):
            self.assertFalse(offline_ok("repo"))                # auto + missing -> may download
            self.assertTrue(offline_ok("repo", True))
        with patch("bookrag.weights.weights_cached", return_value=None):
            self.assertFalse(offline_ok("repo"))                # unknown -> don't force offline

    def test_load_local_first_retries_online_when_cache_is_incomplete(self):
        from bookrag.weights import load_local_first
        calls = []

        class Model:
            def __init__(self, name, **kwargs):
                calls.append(kwargs)
                if kwargs.get("local_files_only"):
                    raise OSError("cache incomplete")
                self.name = name

        with patch("bookrag.weights.weights_cached", return_value=True):
            model = load_local_first(Model, "repo", "auto", device="cpu")
        self.assertEqual([c.get("local_files_only") for c in calls], [True, None])
        self.assertEqual(model.name, "repo")

        calls.clear()
        with patch("bookrag.weights.weights_cached", return_value=False):
            load_local_first(Model, "repo", "auto", device="cpu")
        self.assertEqual([c.get("local_files_only") for c in calls], [None])

    def test_config_wires_offline_loading_into_both_encoders(self):
        from bookrag.index.embedder import embedder_from_config
        from bookrag.retrieve.rerank import reranker_from_config
        cfg = Config({"embedding": {"local_files_only": True}, "rerank": {"local_files_only": "false"}})
        self.assertIs(embedder_from_config(cfg).local_files_only, True)
        self.assertEqual(reranker_from_config(cfg).local_files_only, "false")


class DeviceSelectionTests(unittest.TestCase):
    """A GPU shared with vLLM: 47.65 GiB total, 3.9 MiB free, every query OOM'd."""

    def torch_env(self, cuda=True, free_gb=1.0, mps=False):
        import torch
        from contextlib import ExitStack
        stack = ExitStack()
        stack.enter_context(patch.object(torch.backends.mps, "is_available", return_value=mps))
        stack.enter_context(patch.object(torch.cuda, "is_available", return_value=cuda))
        stack.enter_context(patch.object(torch.cuda, "mem_get_info",
                                         return_value=(int(free_gb * 1e9), int(48e9))))
        return stack

    def test_auto_picks_cpu_when_the_gpu_is_full_and_cuda_when_it_has_room(self):
        from bookrag.index.embedder import resolve_device
        with self.torch_env(free_gb=2.3):
            self.assertEqual(resolve_device("auto", 6.0), "cpu")
            self.assertEqual(resolve_device("cuda", 6.0), "cuda")      # explicit always wins
        with self.torch_env(free_gb=20.0):
            self.assertEqual(resolve_device("auto", 6.0), "cuda")
        with self.torch_env(cuda=False, mps=True):
            self.assertEqual(resolve_device("auto", 6.0), "mps")
        with self.torch_env(cuda=False):
            self.assertEqual(resolve_device("auto", 6.0), "cpu")

    def test_reranker_still_gets_the_gpu_after_the_embedder_takes_its_share(self):
        # vLLM at 0.82 on a 48 GB card leaves ~7 GB; the embedder then uses ~1.6 GB.
        from bookrag.index.embedder import DEFAULT_MIN_FREE_GPU_GB, resolve_device
        with self.torch_env(free_gb=7.0):
            self.assertEqual(resolve_device("auto", DEFAULT_MIN_FREE_GPU_GB), "cuda")
        with self.torch_env(free_gb=5.4):
            self.assertEqual(resolve_device("auto", DEFAULT_MIN_FREE_GPU_GB), "cuda")
        with self.torch_env(free_gb=2.3):                         # vLLM at 0.92
            self.assertEqual(resolve_device("auto", DEFAULT_MIN_FREE_GPU_GB), "cpu")

    def test_oom_detection(self):
        from bookrag.index.embedder import is_gpu_oom
        self.assertTrue(is_gpu_oom(RuntimeError("CUDA out of memory. Tried to allocate 20.00 MiB")))
        self.assertFalse(is_gpu_oom(RuntimeError("shape mismatch")))

    def test_embedder_and_reranker_finish_on_cpu_after_gpu_oom(self):
        from bookrag.index import embedder as emb_mod
        from bookrag.retrieve import rerank as rerank_mod
        devices = []

        class FakeModel:
            def __init__(self, name, device="cpu", **kwargs):
                self.device = device
                devices.append(device)

            def _check(self):
                if self.device == "cuda":
                    raise RuntimeError("CUDA out of memory. Tried to allocate 20.00 MiB")

            def encode(self, payload, **kwargs):
                self._check()
                return np.ones((len(payload), 4), np.float32)

            def predict(self, pairs, **kwargs):
                self._check()
                return [0.9 for _ in pairs]

        fake_loader = lambda cls, name, local_files_only, **kwargs: FakeModel(name, **kwargs)
        try:
            with patch("bookrag.weights.load_local_first", side_effect=fake_loader):
                embedder = emb_mod.Embedder("fake/embedder", device="cuda", fp16=False)
                vectors = embedder.embed_queries(["q1", "q2"])
                reranker = rerank_mod.Reranker("fake/reranker", device="cuda", fp16=False)
                scores = reranker.score("q", ["p1", "p2", "p3"])
        finally:
            for cache in (emb_mod._MODEL_CACHE, rerank_mod._RERANKER_CACHE):
                for key in [k for k in cache if k.startswith("fake/")]:
                    cache.pop(key)
        self.assertEqual(vectors.shape, (2, 4))
        self.assertEqual(scores, [0.9, 0.9, 0.9])
        self.assertEqual((embedder.device, reranker.device), ("cpu", "cpu"))
        self.assertEqual(devices, ["cuda", "cpu", "cuda", "cpu"])

    def test_other_errors_are_not_swallowed(self):
        from bookrag.index import embedder as emb_mod

        class Broken:
            def __init__(self, name, **kwargs):
                pass

            def encode(self, payload, **kwargs):
                raise ValueError("bad input")

        try:
            with patch("bookrag.weights.load_local_first", side_effect=lambda c, n, l, **k: Broken(n)):
                embedder = emb_mod.Embedder("fake/broken", device="cuda", fp16=False)
                with self.assertRaises(ValueError):
                    embedder.embed_queries(["q"])
        finally:
            for key in [k for k in emb_mod._MODEL_CACHE if k.startswith("fake/")]:
                emb_mod._MODEL_CACHE.pop(key)


class AppChatErrorTests(IndexConfigMixin, unittest.TestCase):
    """The chat tab's error message must point at the real cause."""

    def test_gpu_oom_and_backend_failures_get_the_right_hint(self):
        import os
        import streamlit
        import streamlit as st
        from contextlib import ExitStack
        from streamlit.testing.v1 import AppTest
        app_path = str(Path(__file__).resolve().parent.parent / "app.py")
        real_get_option = streamlit.get_option
        with tempfile.TemporaryDirectory() as d, ExitStack() as stack:
            cfg = self.cfg(d)
            store = Store(cfg.index_dir)
            chunk = Chunk("b::00001", "Heat is energy.", "Heat is energy.", "b", "Book", ordinal=1)
            store.save([chunk], np.ones((1, 2)), {"books": [{"book_id": "b", "title": "Book",
                                                              "n_pages": 1, "n_chunks": 1}]})
            llm = Mock()
            llm.primary, llm.host = "m", "h"
            llm.health_check.return_value = {"primary_ok": True, "host": "h"}
            llm.model_catalog.return_value = []
            env = {k: v for k, v in os.environ.items() if k != "BOOKRAG_APP_PASSWORD"}
            for p in [patch.dict(os.environ, env, clear=True),
                      patch("streamlit.get_option", side_effect=lambda k: (
                          "127.0.0.1" if k == "server.address" else real_get_option(k))),
                      patch("bookrag.config.load_config", return_value=cfg),
                      patch("bookrag.llm.client.client_from_config", return_value=llm),
                      patch("bookrag.retrieve.pipeline.Retriever",
                            side_effect=lambda c: SimpleNamespace(_llm=None, reranker=None, store=store)),
                      patch("bookrag.memory.ollama_loaded", return_value=[]),
                      patch("bookrag.memory.system_memory_gb", return_value=(16, 8))]:
                stack.enter_context(p)
            st.cache_resource.clear()
            st.cache_data.clear()
            at = AppTest.from_file(app_path).run(timeout=60)
            self.assertFalse(at.exception, [e.message for e in at.exception])
            for message, hint in [
                ("CUDA out of memory. Tried to allocate 20.00 MiB", "GPU ran out of memory"),
                ("connection refused", "Check the model server"),
            ]:
                with patch("bookrag.chat.engine.ChatEngine.ask", side_effect=RuntimeError(message)):
                    at.chat_input[0].set_value("What is heat?").run(timeout=60)
                self.assertFalse(at.exception, [e.message for e in at.exception])
                self.assertTrue(any(hint in e.value for e in at.error), [e.value for e in at.error])

            # A streamed answer carrying an injected image is rendered disarmed,
            # both live and when the chat history is redrawn.
            evil = "Heat is energy [1].\n\n![x](https://evil.test/?q=secret)"
            pieces = [evil[:25], evil[25:40], evil[40:]]
            with patch("bookrag.chat.engine.ChatEngine.ask", return_value=iter(pieces)):
                at.chat_input[0].set_value("What is heat?").run(timeout=60)
            self.assertFalse(at.exception, [e.message for e in at.exception])
            shown = [m.value for m in at.markdown if "Heat is energy" in m.value]
            self.assertTrue(shown)
            self.assertTrue(all("](" not in v for v in shown), shown)
            self.assertIn("] (https://evil.test", shown[-1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
