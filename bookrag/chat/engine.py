"""Grounded chat over the indexed books.

Three properties this enforces that a plain RAG loop does not:

1. **Refusal.** If nothing clears the reranker floor, the LLM is never called.
   The system says the books do not cover it, instead of paraphrasing a weak hit.
2. **Citations.** Every answer carries [n] markers resolved to real
   book/chapter/page strings from chunk metadata, not model-generated ones.
3. **History condensation.** Follow-ups like "and its derivation?" are rewritten
   into standalone queries before retrieval, so pronouns don't destroy recall.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from bookrag.config import Config
from bookrag.llm.client import OllamaClient, client_from_config
from bookrag.llm.prompts import CHAT_SYSTEM, CHAT_USER, CLAIM_CHECK_SYSTEM, CLAIM_CHECK_USER
from bookrag.paper.verifier import _as_bool
from bookrag.retrieve.pipeline import Retriever, RetrievalResult

CONDENSE_SYSTEM = """Rewrite the user's latest message into a standalone search query,
resolving pronouns and references using the conversation history. Keep technical
terms verbatim. If it is already standalone, return it unchanged.
Output the query only, nothing else."""

NOT_IN_SOURCE = re.compile(r"^\s*NOT_IN_SOURCE\s*:?\s*(.*)", re.IGNORECASE)

# A follow-up only needs the history-condensing LLM call when it actually leans
# on the history. Most questions are already standalone, and paying an 8B
# round-trip to have the model hand the query back unchanged was the cheapest
# latency in the pipeline to delete.
_CONTEXT_DEPENDENT = re.compile(
    r"\b(it|its|it's|this|that|these|those|they|them|their|he|she|his|her|"
    r"the same|above|previous|former|latter|one)\b|^\s*(and|but|so|also|what about|how about|why|ok)\b",
    re.IGNORECASE)


def _local_ollama(llm) -> bool:
    """Whether the answering backend is an Ollama we can evict models from.
    Unload requests sent to a remote OpenAI-compatible server are meaningless."""
    return isinstance(getattr(llm, "_primary", llm), OllamaClient)


def _needs_condensing(question: str) -> bool:
    """True when the question cannot stand on its own without the history."""
    q = question.strip()
    return bool(len(q.split()) <= 3 or _CONTEXT_DEPENDENT.search(q))


# "[3]", "[1, 4]", "[2][5]" -- every number inside a bracketed marker.
_CITE_GROUP = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")


def _cited_numbers(text: str) -> set[int]:
    return {int(n) for g in _CITE_GROUP.findall(text) for n in re.findall(r"\d+", g)}


def _uncited_sentences(text: str, min_words: int = 6) -> list[str]:
    """Substantive sentences carrying no citation marker.

    Short sentences, list lead-ins ending in ':' and headings are exempt; they
    rarely carry a factual claim of their own.
    """
    out = []
    for sent in re.split(r"(?<=[.!?])\s+|\n+", text):
        s = sent.strip().lstrip("-*• ").strip()
        if len(s.split()) < min_words or s.endswith(":") or s.startswith("#"):
            continue
        if not _CITE_GROUP.search(s):
            out.append(s)
    return out


def _plain_math(text: str) -> str:
    """Render LaTeX as plain text for the claim check.

    The answering model often rewrites a book's "eta = 1 - T_cold / T_hot" as
    "$\\eta = 1 - \\frac{T_{\\text{cold}}}{T_{\\text{hot}}}$" despite being told
    not to, and the checker then judged the correct formula "not stated".
    Measured: 2 of 3 correct formula answers were refused that way.
    """
    t = re.sub(r"\${1,2}", " ", text)
    for pattern, repl in ((r"\\(?:text|mathrm|mathit|mathbf|operatorname|rm)\{([^{}]*)\}", r"\1"),
                          (r"_\{([^{}]*)\}", r"_\1"),
                          (r"\^\{([^{}]*)\}", r"^\1"),
                          (r"\\[dt]?frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)")):
        for _ in range(4):                      # unwrap nesting from the inside out
            t, n = re.subn(pattern, repl, t)
            if not n:
                break
    t = re.sub(r"\\(?:left|right)\b", "", t)
    t = re.sub(r"\\(?:cdot|times)\b", "*", t)
    t = re.sub(r"\\([A-Za-z]+)", r"\1", t)      # \eta -> eta, \Delta -> Delta, \ln -> ln
    return re.sub(r"\s+", " ", t).strip()


def _statements(text: str, min_words: int = 4) -> list[tuple[str, list[int]]]:
    """Split an answer into checkable statements, each with the citations that
    cover it.

    A marker covers its own sentence and any uncited sentences just before it
    in the same paragraph ("A. B [2]." attributes both to [2]); an uncited
    paragraph tail inherits the paragraph's last marker. A lead-in ending in ":"
    is joined to what follows it, and an uncited paragraph (typically a display
    formula) takes the next citation in a later paragraph, else the previous
    one. Measured: "…is given by:\\n\\n$$ E = E_a P V^+ $$\\n\\nThis follows … [5]."
    was checked as a bare, uncited "E = E_a P V^+" -- stripped of the condition
    it depends on -- and intermittently refused although the book states it.
    """
    out: list[tuple[str, list[int]]] = []
    orphans: list[str] = []          # uncited, waiting for a citation in a later paragraph
    previous: list[int] = []         # the most recent citation seen
    lead = ""                        # "The formula is given by:" joins the next statement
    for para in re.split(r"\n\s*\n", text.strip()):
        pending: list[str] = []
        last: list[int] = []
        for sent in re.split(r"(?<=[.!?])\s+|\n+", para):
            nums = sorted(_cited_numbers(sent))
            body = re.sub(r"\s+([.,;:!?])", r"\1", _CITE_GROUP.sub("", sent))
            body = body.strip().lstrip("-*•#> ").strip()
            if not body:
                if nums:                     # a bare marker covers what came before it
                    out.extend((p, nums) for p in orphans + pending)
                    orphans, pending, last = [], [], nums
                continue
            if not nums and body.endswith(":"):
                lead = f"{lead} {body}".strip()
                continue
            if lead:
                body, lead = f"{lead} {body}", ""
            is_math = "$" in body or "\\" in body
            if nums:
                out.extend((p, nums) for p in orphans + pending)
                orphans, pending = [], []
                # A cited statement is checked however short it is: a
                # three-word claim can be just as false as a long one.
                if re.search(r"[^\W_]", body):
                    out.append((body, nums))
                last = nums
            elif is_math or len(body.split()) >= min_words:
                pending.append(body)
        if pending:
            if last:
                out.extend((p, last) for p in pending)
            else:
                orphans.extend(pending)
        if last:
            previous = last
    out.extend((p, previous) for p in orphans)
    return out


@dataclass
class Answer:
    question: str
    search_query: str
    text: str
    citations: list[str] = field(default_factory=list)
    used_citations: list[int] = field(default_factory=list)
    grounded: bool = True
    top_score: float = 0.0
    retrieval: RetrievalResult | None = None

    def sources(self) -> list[str]:
        """One line per distinct source the answer cites, e.g. "[2][5] Book > Ch 1, p.4".

        Only cited passages are listed -- neighbours pulled in for context are
        not sources -- and passages that resolve to the same citation label are
        merged instead of repeated.
        """
        shown = self.used_citations or list(range(1, len(self.citations) + 1))
        merged: dict[str, list[int]] = {}
        for n in shown:
            if 1 <= n <= len(self.citations):
                merged.setdefault(self.citations[n - 1], []).append(n)
        return [f"{''.join(f'[{n}]' for n in ns)} {label}" for label, ns in merged.items()]

    def formatted(self) -> str:
        sources = self.sources()
        if not sources:
            return self.text
        return "\n".join([self.text, "", "Sources:"] + [f"  {s}" for s in sources])


class ChatEngine:
    def __init__(self, cfg: Config, retriever: Retriever | None = None,
                 llm: OllamaClient | None = None):
        self.cfg = cfg
        self.retriever = retriever or Retriever(cfg)
        self.llm = llm or client_from_config(cfg)
        self.history: list[dict] = []
        self.last_answer: Answer | None = None

    def reset(self) -> None:
        self.history = []

    # ------------------------------------------------------------- models
    def available_models(self) -> list[str]:
        """Models Ollama can serve, best-effort."""
        try:
            return self.llm.available_models()
        except Exception:
            return [self.llm.primary]

    def active_model(self) -> str:
        return self.llm.primary

    def use_model(self, name: str, release_previous: bool = True) -> dict:
        """Switch the answering model, freeing the previous one's memory.

        The client reads `primary` per request, so this is a field assignment
        rather than a reconnect -- the encoders and the index stay loaded. The
        previous model is evicted from Ollama by default, which is the whole
        point of switching on a machine that is short of RAM.
        """
        from bookrag.memory import ollama_unload, system_memory_gb
        previous = self.llm.primary
        freed = False
        if release_previous and previous and previous != name and _local_ollama(self.llm):
            freed = ollama_unload(previous, self.llm.host)
        self.llm.primary = name
        # The retriever may hold its own client (query condensing, expansion).
        # A retriever shared between UI sessions must not have one user's
        # choice imposed on everyone else's.
        if (getattr(self.retriever, "_llm", None) is not None
                and not getattr(self.retriever, "shared", False)):
            self.retriever._llm.primary = name
        return {"previous": previous, "active": name, "released_previous": freed,
                "available_gb": system_memory_gb()[1]}

    def release_model(self) -> dict:
        """Evict the active model from Ollama now, without changing the choice.

        The next question reloads it. Use when handing the machine to something
        else -- an ingest, another app, a different model.
        """
        from bookrag.memory import ollama_unload, system_memory_gb
        before = system_memory_gb()[1]
        ok = ollama_unload(self.llm.primary, self.llm.host) if _local_ollama(self.llm) else False
        after = system_memory_gb()[1]
        return {"model": self.llm.primary, "released": ok,
                "freed_gb": round(max(after - before, 0.0), 1), "available_gb": after}

    def _condense(self, question: str) -> str:
        if not self.history or not _needs_condensing(question):
            return question
        recent = self.history[-6:]
        convo = "\n".join(f"{m['role'].upper()}: {m['content'][:400]}" for m in recent)
        try:
            q = self.llm.complete(CONDENSE_SYSTEM,
                                  f"HISTORY\n{convo}\n\nLATEST: {question}",
                                  fast=True, temperature=0.0, max_tokens=120)
            q = q.strip().strip('"')
            return q if 3 < len(q) < 400 else question
        except Exception:
            return question

    def ask(self, question: str, book_ids: list[str] | None = None,
            stream: bool = False):
        search_query = self._condense(question)
        # The threshold lives in the reranker's probability space. With
        # reranking off, top_score is an RRF score on an unrelated scale, so
        # comparing the two would refuse everything.
        threshold = (float(self.cfg.get("grounding.answer_threshold", 0.05))
                     if self.retriever.reranker else float("-inf"))
        rr = self.retriever.retrieve(search_query, book_ids=book_ids)

        # An empty context is a refusal whatever the score says: the model
        # would otherwise be answering from its own weights.
        if not rr.grounded or not rr.context.strip() or rr.top_score < threshold:
            msg = ("I can't answer that from the indexed book(s) — no passage in them "
                   "covers it. Try rephrasing with the book's own terminology, or "
                   "check whether the relevant chapter was ingested.")
            answer = Answer(question, search_query, msg, [], [], False, rr.top_score, rr)
            self.history += [{"role": "user", "content": question},
                             {"role": "assistant", "content": msg}]
            self.last_answer = answer
            # stream=True is a promise about the return type, and refusing is a
            # normal outcome — not an excuse to hand the caller a bare Answer
            # where it is iterating.
            return self._refusal_stream(msg) if stream else answer

        # The encoders are done for this query; the LLM is about to want ~6 GB.
        self.retriever.arbiter.before("generate")

        user_msg = CHAT_USER.format(context=rr.context, question=search_query)
        messages = [{"role": "system", "content": CHAT_SYSTEM}]
        for m in self.history[-2:]:
            messages.append({"role": m["role"], "content": m["content"][:1200]})
        messages.append({"role": "user", "content": user_msg})

        if stream:
            return self._stream(question, search_query, messages, rr)

        raw = self.llm.chat(messages, temperature=float(self.cfg.get("llm.temperature", 0.1)))
        self.last_answer = self._finalize(question, search_query, raw, rr)
        return self.last_answer

    @staticmethod
    def _refusal_stream(msg: str):
        yield msg

    def _stream(self, question: str, search_query: str, messages: list[dict],
                rr: RetrievalResult):
        buf: list[str] = []
        for piece in self.llm.stream_chat(messages):
            buf.append(piece)
            yield piece
        # Streaming callers read the text as it arrives; the finalised Answer
        # (citations, grounding flag) is left here for them to pick up after.
        self.last_answer = self._finalize(question, search_query, "".join(buf), rr)

    def _finalize(self, question: str, search_query: str, raw: str,
                  rr: RetrievalResult) -> Answer:
        m = NOT_IN_SOURCE.match(raw.strip())
        if m:
            text = (f"Not covered by the indexed book(s): {m.group(1).strip()}"
                    if m.group(1).strip() else
                    "The indexed book(s) don't cover this.")
            answer = Answer(question, search_query, text, [], [], False, rr.top_score, rr)
        else:
            cited = _cited_numbers(raw)
            invalid = sorted(n for n in cited if not 1 <= n <= len(rr.citations))
            used = sorted(n for n in cited if 1 <= n <= len(rr.citations))
            uncited = (_uncited_sentences(raw)
                       if bool(self.cfg.get("grounding.require_sentence_citations", False))
                       else [])
            if invalid:
                # A marker pointing at no passage means the model invented a
                # source. One real citation elsewhere doesn't make up for that.
                text = ("The answer cited "
                        + ", ".join(f"[{n}]" for n in invalid[:5])
                        + " but no such passage was retrieved, so I can't vouch "
                          "that it came from the books. Try asking again.")
                answer = Answer(question, search_query, text, [], [], False,
                                rr.top_score, rr)
            elif uncited:
                text = ("Part of the answer made claims without citing the books "
                        f"(e.g. \"{uncited[0][:120]}\"), so I can't vouch for it. "
                        "Try asking again or narrowing the question.")
                answer = Answer(question, search_query, text, [], [], False,
                                rr.top_score, rr)
            elif not used and bool(self.cfg.get("grounding.require_citations", True)):
                # An uncited answer is indistinguishable from the model
                # answering out of its own weights, which is the one thing this
                # system promises not to do. Refuse rather than let it through.
                text = ("The books were searched but the answer came back without "
                        "citations to them, so I can't vouch that it came from the "
                        "books rather than the model's own knowledge. Try rephrasing "
                        "with the book's own terminology.")
                answer = Answer(question, search_query, text, [], [], False,
                                rr.top_score, rr)
            else:
                problem = (self._check_claims(raw, rr)
                           if bool(self.cfg.get("grounding.verify_answer_claims", False)) else "")
                if problem:
                    answer = Answer(question, search_query, problem, [], [], False,
                                    rr.top_score, rr)
                else:
                    answer = Answer(question, search_query, raw.strip(), rr.citations,
                                    used, True, rr.top_score, rr)
        self.history += [{"role": "user", "content": question},
                         {"role": "assistant", "content": answer.text}]
        return answer

    def _check_claims(self, raw: str, rr: RetrievalResult) -> str:
        """Check each cited statement against the passages it cites.

        A valid [n] marker only proves the model pointed at a real passage, not
        that the passage says what the sentence claims. A separate temperature-0
        pass judges that. Returns a refusal message, or "" when every statement
        is supported. Anything short of an explicit "supported" -- a missing
        verdict, malformed JSON, a failed call -- is a refusal.
        """
        statements = _statements(raw)
        if not statements:
            return ""
        use_fast = str(self.cfg.get("grounding.claim_check_model", "fast")).lower() != "primary"
        have_passages = bool(rr.passages) and len(rr.passages) == len(rr.citations)
        batch_size = 12
        for start in range(0, len(statements), batch_size):
            batch = statements[start:start + batch_size]
            if have_passages:
                cited = sorted({n for _, ns in batch for n in ns if 1 <= n <= len(rr.citations)})
                shown = cited or list(range(1, len(rr.citations) + 1))
                excerpts = "\n\n".join(f"[{n}] {rr.citations[n - 1]}\n{rr.passages[n - 1]}"
                                       for n in shown)
            else:
                excerpts = rr.context
            listing = "\n".join(
                f"{i}. {_plain_math(text)} (cites {''.join(f'[{n}]' for n in ns) or 'none'})"
                for i, (text, ns) in enumerate(batch, start=1))
            user = CLAIM_CHECK_USER.format(excerpts=excerpts, statements=listing)

            data, error = None, None
            # The fast model keeps the added latency small; if it is missing or
            # fails, the answering model is certainly available.
            for fast in ([True, False] if use_fast else [False]):
                try:
                    data = self.llm.complete_json(CLAIM_CHECK_SYSTEM, user, fast=fast,
                                                  temperature=0.0)
                    break
                except Exception as exc:
                    error = exc
            if data is None:
                return ("The answer couldn't be checked against the books "
                        f"({str(error)[:120]}), so I can't vouch for it. Try again.")

            verdicts = data.get("verdicts") if isinstance(data, dict) else None
            by_id: dict[int, list[dict]] = {}
            for v in verdicts if isinstance(verdicts, list) else []:
                if not isinstance(v, dict) or isinstance(v.get("id"), bool):
                    continue
                try:
                    by_id.setdefault(int(v.get("id")), []).append(v)
                except (TypeError, ValueError):
                    continue
            for i, (text, _) in enumerate(batch, start=1):
                found = by_id.get(i, [])
                # A judge that says both "supported" and "not supported" about
                # one statement has decided nothing; neither verdict is a pass.
                if len({_as_bool(x.get("supported")) for x in found}) > 1:
                    return ("The answer couldn't be verified: the check returned conflicting "
                            f"verdicts for \"{text[:140]}\". Try again.")
                v = found[0] if found else None
                if v is None or not _as_bool(v.get("supported")):
                    why = (str(v.get("reason") or "").strip() if v
                           else "the check returned no verdict for it")
                    return ("Part of the answer isn't supported by the passages it cites "
                            f"(\"{text[:140]}\" — {why[:140]}), so I can't vouch for it. "
                            "Try rephrasing the question.")
        return ""
