"""Question-paper generation.

Per question slot:
  1. pick a topic (PYQ-weighted if a profile is loaded, else round-robin over
     chapters, with used topics penalised so the paper spans the syllabus),
  2. retrieve evidence for that topic from the index,
  3. generate the question against that evidence, in the PYQ style if supplied,
  4. verify it independently (verifier.py), retrying with feedback on failure,
  5. deduplicate against questions already accepted for this paper.

Slot-by-slot rather than one giant call: a per-question retrieval means each
question is grounded in evidence chosen *for it*, and one bad question can be
retried without regenerating the paper.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from bookrag.config import Config
from bookrag.llm.client import OllamaClient, client_from_config
from bookrag.llm.prompts import QGEN_SYSTEM, QGEN_USER
from bookrag.paper.blueprint import Blueprint, assign_blooms, bloom_for_marks, discover_topics
from bookrag.paper.pyq import PYQProfile
from bookrag.paper.verifier import verify_question
from bookrag.retrieve.pipeline import Retriever
from bookrag.schemas import Question

_CANCELLED = object()       # _generate_slot's "stopped on request" result


@dataclass
class GenerationReport:
    requested: int = 0
    accepted: int = 0
    rejected: int = 0
    retried: int = 0
    duplicates: int = 0
    cancelled: bool = False        # stopped early; questions accepted so far are kept
    ungrounded_topics: list[str] = field(default_factory=list)
    rejections: list[dict] = field(default_factory=list)

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.requested if self.requested else 0.0


@dataclass
class Paper:
    blueprint: Blueprint
    questions: list[Question] = field(default_factory=list)
    report: GenerationReport = field(default_factory=GenerationReport)

    def section_spec(self, name: str):
        for spec in self.blueprint.sections:
            if spec.name == name:
                return spec
        return None

    def section_marks(self, name: str) -> int:
        """Marks a candidate can score in a section. "Answer ANY ONE" of three
        2-mark questions is worth 2, not 6."""
        marks = sorted((q.marks for q in self.questions if q.section == name), reverse=True)
        spec = self.section_spec(name)
        return sum(marks[:spec.answer_count] if spec else marks)

    @property
    def total_marks(self) -> int:
        return sum(self.section_marks(n) for n in dict.fromkeys(q.section for q in self.questions))


class PaperGenerator:
    def __init__(self, cfg: Config, retriever: Retriever | None = None,
                 llm: OllamaClient | None = None, pyq: PYQProfile | None = None):
        self.cfg = cfg
        self.retriever = retriever or Retriever(cfg)
        self.llm = llm or client_from_config(cfg)
        self.pyq = pyq

    # ------------------------------------------------------------ topics
    def _topic_pool(self, blueprint: Blueprint, book_ids: list[str] | None) -> list[str]:
        if blueprint.topics:
            return list(blueprint.topics)
        if self.pyq and self.pyq.priority_topics():
            # PYQ topics first (what this examiner actually asks), book chapters
            # after them as filler so the paper can still span the syllabus.
            pyq_topics = self.pyq.priority_topics()
            chapters = discover_topics(self.retriever.store, book_ids)
            seen = {t.lower() for t in pyq_topics}
            return pyq_topics + [c for c in chapters if c.lower() not in seen]
        return discover_topics(self.retriever.store, book_ids)

    def _pick_topic(self, pool: list[str], used: dict[str, int]) -> str:
        if not pool:
            return "the book's core material"
        return min(pool, key=lambda t: (used.get(t, 0), pool.index(t)))

    # -------------------------------------------------------- generation
    def generate(self, blueprint: Blueprint, book_ids: list[str] | None = None,
                 progress=print, should_cancel=None) -> Paper:
        """Build a paper slot by slot.

        `should_cancel` is polled before each question and each retry. On
        cancel the paper built so far is returned with `report.cancelled` set,
        rather than discarding questions that took minutes to verify.
        """
        stop = should_cancel or (lambda: False)
        blueprint.validate()
        paper = Paper(blueprint=blueprint)
        pool = self._topic_pool(blueprint, book_ids)
        used: dict[str, int] = {}
        number = 0
        verify_on = bool(self.cfg.get("grounding.verify_questions", True))
        max_retries = int(self.cfg.get("grounding.verification_retries", 2))
        min_conf = float(self.cfg.get("grounding.min_verify_confidence", 0.7))

        for section in blueprint.sections:
            section_pool = section.topics or pool
            blooms = assign_blooms(section.count,
                                   section.bloom_mix or blueprint.bloom_mix
                                   or bloom_for_marks(section.marks_each))
            progress(f"\n{section.name}: {section.count} x {section.marks_each} marks")

            for i in range(section.count):
                if stop():
                    return self._cancelled(paper, progress)
                paper.report.requested += 1
                bloom = blooms[i] if i < len(blooms) else "understand"
                topic = self._pick_topic(section_pool, used)
                used[topic] = used.get(topic, 0) + 1

                q = self._generate_slot(topic, section, bloom, book_ids, paper,
                                        verify_on, max_retries, min_conf, progress, stop)
                if q is _CANCELLED:
                    paper.report.requested -= 1      # the slot was abandoned, not attempted
                    return self._cancelled(paper, progress)
                if q is None:
                    continue
                number += 1
                q.number = number
                paper.questions.append(q)
                paper.report.accepted += 1
                progress(f"  [{number}] ok ({topic[:45]}, {bloom}, conf {q.verify_confidence:.2f})")

        return paper

    @staticmethod
    def _cancelled(paper: Paper, progress) -> Paper:
        paper.report.cancelled = True
        progress(f"Cancelled: keeping the {len(paper.questions)} question(s) accepted so far.")
        return paper

    def _generate_slot(self, topic: str, section, bloom: str,
                       book_ids: list[str] | None, paper: Paper,
                       verify_on: bool, max_retries: int, min_conf: float,
                       progress, stop=lambda: False):
        """An accepted Question, None if the slot failed, or _CANCELLED."""
        feedback = ""
        for attempt in range(max_retries + 1):
            if stop():
                return _CANCELLED
            rr = self.retriever.retrieve(
                f"{topic} — key concepts, definitions, worked examples and derivations",
                book_ids=book_ids,
            )
            if not rr.grounded or not rr.results:
                paper.report.ungrounded_topics.append(topic)
                progress(f"  - skipped '{topic[:45]}': no grounded passages")
                return None

            candidates = self._call_generator(topic, section, bloom, rr, feedback, count=1)
            if not candidates:
                feedback = "Your previous output was unusable. Return valid JSON."
                paper.report.retried += 1
                continue

            q = candidates[0]
            if self._is_duplicate(q, paper.questions):
                paper.report.duplicates += 1
                feedback = ("This paper already contains a question on that exact point. "
                            "Ask about a different aspect of the topic.")
                paper.report.retried += 1
                continue

            if not verify_on:
                # Nothing checked it, so it must not be labelled as checked.
                q.verified, q.verify_confidence = False, 0.0
                q.verify_notes = "verification disabled"
                return q

            verdict = verify_question(self.llm, q, rr.context, min_confidence=min_conf)
            if verdict.answerable:
                q.verified = True
                q.verify_confidence = verdict.confidence
                q.verify_notes = verdict.reason
                return q

            paper.report.rejections.append({
                "topic": topic, "attempt": attempt + 1, "question": q.text[:200],
                "reason": verdict.reason, "confidence": verdict.confidence,
                "unsupported": verdict.unsupported_claims,
            })
            paper.report.retried += 1
            feedback = (
                f"A moderator REJECTED your previous attempt: {verdict.reason}. "
                f"Unsupported: {'; '.join(verdict.unsupported_claims) or 'n/a'}. "
                f"{verdict.suggested_fix} Write a question that is beyond doubt "
                f"answerable from the excerpts alone."
            )

        paper.report.rejected += 1
        progress(f"  - dropped '{topic[:45]}': failed verification {max_retries + 1}x")
        return None

    def _call_generator(self, topic, section, bloom, rr, feedback: str,
                        count: int = 1) -> list[Question]:
        style_note = self._style_note(section, feedback)
        user = QGEN_USER.format(context=rr.context, topic=topic, count=count,
                                qtype=section.type, marks=section.marks_each,
                                bloom=bloom, style_note=style_note)
        try:
            data = self.llm.complete_json(
                QGEN_SYSTEM, user, fast=False,
                temperature=float(self.cfg.get("llm.generation_temperature", 0.35)),
            )
        except Exception:
            return []

        raw_list = data.get("questions") if isinstance(data, dict) else data
        if isinstance(raw_list, dict):
            raw_list = [raw_list]
        if not isinstance(raw_list, list):
            return []
        out: list[Question] = []
        # Bare strings or nulls in the list are unusable drafts to retry, not a
        # reason to crash the whole paper.
        for item in [i for i in raw_list if isinstance(i, dict)][:count]:
            text = _strip_source_references(str(item.get("text") or "").strip())
            if not text:
                continue
            options = item.get("options")
            src_ids = _resolve_sources(item.get("source_ids"), rr)
            out.append(Question(
                number=0, section=section.name, text=text,
                marks=section.marks_each, qtype=section.type,
                bloom=str(item.get("bloom") or bloom).lower(),
                topic=str(item.get("topic") or topic),
                answer=str(item.get("answer") or "").strip(),
                options=([str(o).strip() for o in options if str(o).strip()]
                         if isinstance(options, list) else []),
                correct_option=str(item.get("correct_option") or ""),
                source_chunk_ids=[c.id for c in src_ids],
                citations=[c.citation() for c in src_ids],
            ))
        return out

    def _style_note(self, section, feedback: str) -> str:
        bits: list[str] = []
        if section.instructions:
            bits.append(f"Section instructions: {section.instructions}")
        if self.pyq and self.pyq.style_guide:
            bits.append("Match this examiner's house style:\n" + self.pyq.style_guide)
        if self.pyq:
            examples = self.pyq.example_questions(section.type, section.marks_each)
            if examples:
                sample = "\n".join(f"  - {e}" for e in examples)
                bits.append("Previous-year questions from this course — mirror their "
                            "phrasing and structure, but ask about DIFFERENT content "
                            "drawn from the excerpts:\n" + sample)
        if feedback:
            bits.append("IMPORTANT FEEDBACK ON YOUR LAST ATTEMPT: " + feedback)
        return "\n\n".join(bits)

    @staticmethod
    def _is_duplicate(q: Question, existing: list[Question], threshold: float = 0.45) -> bool:
        """Token-overlap dedup — cheap, and catches the near-restatements an
        LLM produces when two chapters cover the same concept.

        The threshold is deliberately low. Measured on real output, "Define the
        term 'heat capacity' and state its two forms" vs "Define the term 'heat
        capacity' for a substance" scores 0.571, and two Zeroth-Law questions
        scored 0.50 — both obvious duplicates a 0.6 cutoff let through. Command
        verbs are stripped first so "State X" and "Define X" collide.
        """
        def norm(t: str) -> str:
            return " ".join(re.findall(r"[a-z0-9]+", t.lower()))

        def toks(t: str) -> set[str]:
            words = set(re.findall(r"[a-z]{4,}", t.lower()))
            # Command verbs and filler carry no topical signal; keeping them
            # inflates similarity between unrelated questions and deflates it
            # between two askings of the same thing.
            words -= _STOPVERBS
            # A short question ("State Ohm's law.") may have no 4+ letter
            # topical word at all; compare its shorter words instead of
            # declaring it unique by default.
            return words or set(re.findall(r"[a-z0-9]{2,}", t.lower())) - _STOPVERBS

        a, a_norm = toks(q.text), norm(q.text)
        for other in existing:
            if a_norm and a_norm == norm(other.text):
                return True
            b = toks(other.text)
            if a and b and len(a & b) / len(a | b) >= threshold:
                return True
        return False


# "Create an expression..." / "Remember the definition..." are taxonomy labels
# leaking into exam language. Analyse/Evaluate are legitimate and left alone.
# Matches the taxonomy verb whether it opens the sentence or follows a leading
# subordinate clause ("Using the definition of G, create an expression ...").
_BLOOM_VERB = re.compile(r"(^|,\s*)(create|remember)\b\s*", re.IGNORECASE)
_BLOOM_REPLACEMENT = {"create": "derive", "remember": "state"}
# "Remember the statement of X" -> "State X" rather than "State the statement of X".
_REDUNDANT = re.compile(
    r"\b(state|derive)\s+the\s+(statement|definition|expression)\s+of\s+",
    re.IGNORECASE,
)


def _rewrite_bloom_verb(m: "re.Match") -> str:
    lead, verb = m.group(1), _BLOOM_REPLACEMENT[m.group(2).lower()]
    # Sentence-initial keeps its capital; mid-sentence stays lowercase.
    return (verb.capitalize() + " ") if lead == "" else (lead + verb + " ")


_STOPVERBS = {
    "define", "state", "explain", "describe", "discuss", "write", "give",
    "derive", "compare", "distinguish", "analyze", "analyse", "evaluate",
    "create", "list", "outline", "show", "prove", "term", "terms", "following",
    "with", "your", "answer", "briefly", "using", "what", "which", "that",
    "this", "from", "into", "also", "these", "those", "such", "each", "both",
}

# Phrases that betray the excerpts to a student. The prompt forbids them, but a
# 4-8B model still leaks one occasionally, so strip them deterministically.
_SOURCE_REF = re.compile(
    r"\s*(,\s*)?\b(as\s+)?(given|stated|provided|described|discussed|mentioned|"
    r"defined|presented|shown|outlined)\s+(in|by)\s+the\s+"
    r"(text|passage|excerpt|chapter|section|book|material)s?\b",
    re.IGNORECASE,
)
# "the given relation dP/dT = ..." / "the following equation"
_GIVEN = re.compile(
    r"\s*(,\s*)?\b(from|using|for)?\s*the\s+(given|following|above|stated)\s+"
    r"(relation|equation|formula|expression|result|statement)\b",
    re.IGNORECASE,
)
# "as described in Chapter 1", "discussed in Section 2.3"
_IN_CHAPTER = re.compile(
    r"\s*(,\s*)?\b(as\s+)?(described|explained|given|discussed|mentioned|"
    r"presented|shown|defined|stated|covered)\s+in\s+"
    r"(the\s+)?(chapter|section|unit|part)\s*[\dIVXivx.]*\b",
    re.IGNORECASE,
)
_ACCORDING = re.compile(
    r"\s*\b(according\s+to|based\s+on|from)\s+the\s+"
    r"(text|passage|excerpt|chapter|section|book|material)s?\b\s*,?\s*",
    re.IGNORECASE,
)


def _strip_source_references(text: str) -> str:
    text = _SOURCE_REF.sub("", text)
    text = _GIVEN.sub("", text)
    text = _IN_CHAPTER.sub("", text)
    text = _BLOOM_VERB.sub(_rewrite_bloom_verb, text)
    text = _REDUNDANT.sub(lambda m: m.group(1).capitalize() + " ", text)
    text = _ACCORDING.sub(" ", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    text = re.sub(r"\s+([,.?!])", r"\1", text)
    # A removal at the head of the sentence can leave stranded punctuation
    # ("Using the following equation, compute..." -> ", compute...").
    text = re.sub(r"^[\s,;:.\-]+", "", text)
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    return text


def _resolve_sources(source_ids, rr) -> list:
    """Map the model's [n] markers back onto real chunks; fall back to the top hits.

    Models return a list, a bare number or a string such as "1, 3". Anything
    else is ignored: a malformed source list must not abort a question.
    """
    chunks = [sc.chunk for sc in rr.results]
    if isinstance(source_ids, (int, str)) and not isinstance(source_ids, bool):
        source_ids = re.findall(r"\d+", str(source_ids))
    elif not isinstance(source_ids, list):
        source_ids = []
    picked = []
    for n in source_ids:
        if isinstance(n, bool):
            continue
        try:
            i = int(n) - 1
        except (TypeError, ValueError):
            continue
        if 0 <= i < len(chunks) and chunks[i] not in picked:
            picked.append(chunks[i])
    return picked or chunks[:3]
