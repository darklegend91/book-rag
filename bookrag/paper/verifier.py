"""Independent groundedness check for every generated question.

The generator wrote the question while looking at the excerpts, so asking it
"is this grounded?" is self-marking. This runs a *separate* call with a
moderator persona, temperature 0, and only the excerpts + the finished question,
with no memory of how it was produced. Questions that fail are regenerated with
the failure reason fed back; questions that fail repeatedly are dropped rather
than shipped.

This is the single most important component for "questions from the book, never
outside it". Every malformed or ambiguous verifier reply is therefore read as a
rejection, never as a pass.
"""
from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass

from bookrag.llm.client import OllamaClient
from bookrag.llm.prompts import (MCQ_OPTIONS_SYSTEM, MCQ_OPTIONS_USER,
                                 VERIFY_SYSTEM, VERIFY_USER)
from bookrag.schemas import Question

# The generator prompt asks for exactly this many choices.
MCQ_MIN_OPTIONS = 4


@dataclass
class Verdict:
    answerable: bool
    confidence: float
    reason: str = ""
    unsupported_claims: list[str] = None
    suggested_fix: str = ""

    def __post_init__(self):
        self.unsupported_claims = self.unsupported_claims or []


def _as_bool(value) -> bool:
    """Only an explicit true counts. bool("false") is True, which is how a
    moderator's rejection used to be read as approval."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return False


def _as_confidence(value) -> float:
    if isinstance(value, bool):
        return 0.0
    try:
        c = float(value)
    except (TypeError, ValueError):
        return 0.0
    return c if math.isfinite(c) and 0.0 <= c <= 1.0 else 0.0


def verify_question(llm: OllamaClient, question: Question, context: str,
                    min_confidence: float = 0.7) -> Verdict:
    is_mcq = question.qtype == "mcq" or bool(question.options)
    if is_mcq:
        problem = _mcq_shape_problem(question)
        if problem:
            return Verdict(False, 0.0, problem,
                           suggested_fix=f"Give exactly {MCQ_MIN_OPTIONS} distinct options "
                                         "and set correct_option to the exact text of one.")

    payload = VERIFY_USER.format(context=context, marks=question.marks,
                                 question=_full_question_text(question),
                                 answer=question.answer or "(none supplied)")
    try:
        data = llm.complete_json(VERIFY_SYSTEM, payload, fast=False, temperature=0.0)
    except Exception as exc:
        # A verifier failure must never be read as a pass.
        return Verdict(False, 0.0, f"verifier error: {exc}")
    if not isinstance(data, dict):
        return Verdict(False, 0.0, f"verifier returned {type(data).__name__}, not a JSON object")

    confidence = _as_confidence(data.get("confidence"))
    claims = data.get("unsupported_claims")
    verdict = Verdict(
        answerable=_as_bool(data.get("answerable")) and confidence >= min_confidence,
        confidence=confidence,
        reason=str(data.get("reason", ""))[:400],
        unsupported_claims=[str(c) for c in (claims if isinstance(claims, list) else [])][:5],
        suggested_fix=str(data.get("suggested_fix", ""))[:300],
    )

    # The general moderator reliably catches ungrounded content but not two
    # distractors that are logically equivalent -- measured: it passed
    # "never decreases" alongside "increases or remains constant" at conf 1.00.
    # MCQs therefore get a second, narrower pass that judges each option alone.
    if verdict.answerable and is_mcq:
        verdict = _check_mcq_options(llm, question, context, verdict)

    return verdict


def _norm_option(text: str) -> str:
    """'dS = dQ/T' and 'dS=dQ/T.' are the same option."""
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text)).rstrip(".").casefold()


# ASCII (dQ_rev), LaTeX (T_{hot}) and Unicode (T₁) subscripts.
_SUBSCRIPT = re.compile(r"_\{[^}]*\}|_[^\W_]+|[₀-ₜ]+")


def _bare_option(text: str) -> str:
    """The option with its subscripts removed: 'dS = dQ_rev / T' -> 'ds=dq/t'."""
    return _norm_option(_SUBSCRIPT.sub("", text))


def _mcq_shape_problem(question: Question) -> str:
    """Structural defects no LLM judgement can rescue. Normalises a letter or
    index answer key ("B", "2") to the option text as a side effect."""
    options = [o.strip() for o in question.options]
    if len(options) < MCQ_MIN_OPTIONS or any(not o for o in options):
        return f"MCQ has {len([o for o in options if o])} usable options, needs {MCQ_MIN_OPTIONS}."
    if len({_norm_option(o) for o in options}) != len(options):
        return "MCQ has duplicate options (identical apart from spacing or case)."
    key = (question.correct_option or "").strip()
    matches = [o for o in options if o.casefold() == key.casefold()]
    if not matches:
        letter = key.strip("()").upper()
        if len(letter) == 1 and "A" <= letter < chr(65 + len(options)):
            matches = [options[ord(letter) - 65]]
        elif key.isdigit() and 1 <= int(key) <= len(options):
            matches = [options[int(key) - 1]]
    if len(matches) != 1:
        return "MCQ answer key does not name exactly one of its options."
    key = matches[0]
    question.correct_option = key
    # A distractor that IS the answer with its qualifier dropped ("dQ/T" beside
    # "dQ_rev/T") is defensible whenever the stem already implies the qualifier.
    # Measured: the option checker passed exactly that pair at confidence 1.0.
    # Swapped subscripts ("T_hot/T_cold" beside "T_cold/T_hot") are a fair
    # distractor and are not caught here: only dropping them is.
    for option in options:
        if option != key and (_norm_option(option) == _bare_option(key)
                              or _bare_option(option) == _norm_option(key)):
            return (f"MCQ option '{option}' is the answer '{key}' with a subscript or "
                    "qualifier dropped, so a careful student could defend both.")
    return ""


def _check_mcq_options(llm: OllamaClient, question: Question, context: str,
                       verdict: Verdict) -> Verdict:
    n = len(question.options)
    opts = "\n".join(f"  [{i}] {o}" for i, o in enumerate(question.options))
    try:
        data = llm.complete_json(
            MCQ_OPTIONS_SYSTEM,
            MCQ_OPTIONS_USER.format(context=context, stem=question.text,
                                    options=opts, stated=question.correct_option),
            fast=False, temperature=0.0,
        )
    except Exception as exc:
        return Verdict(False, 0.0, f"MCQ option check failed: {exc}")
    if not isinstance(data, dict):
        return Verdict(False, 0.0, "MCQ option check returned no JSON object")

    # Every option must be judged exactly once. A checker that looked at one
    # of four options has not established that the other three are wrong.
    flags = data.get("options")
    flags = [f for f in flags if isinstance(f, dict)] if isinstance(flags, list) else []
    indices = [f.get("index") for f in flags]
    if (len(flags) != n
            or not all(isinstance(i, int) and not isinstance(i, bool) for i in indices)
            or sorted(indices) != list(range(n))):
        return Verdict(False, verdict.confidence,
                       reason=f"MCQ option check judged {len(flags)} of {n} options "
                              "or returned malformed indices.",
                       suggested_fix="Rewrite so each option is clearly right or wrong.")

    true_idx = [f["index"] for f in flags if _as_bool(f.get("true"))]
    if len(true_idx) != 1:
        return Verdict(
            False, verdict.confidence,
            reason=(f"MCQ has {len(true_idx)} defensible options, not 1. "
                    f"{data.get('reason', '')}")[:400],
            unsupported_claims=[question.options[i] for i in true_idx][:4],
            suggested_fix="Rewrite the distractors so exactly one option is correct.",
        )

    # Realign the answer key if the checker disagrees with the stated option.
    checked = question.options[true_idx[0]]
    if checked.strip() != (question.correct_option or "").strip():
        question.correct_option = checked
        verdict.reason = (verdict.reason + " [answer key realigned]")[:400]
    return verdict


def _full_question_text(q: Question) -> str:
    if q.options:
        opts = "\n".join(f"  {chr(65+i)}) {o}" for i, o in enumerate(q.options))
        return f"{q.text}\n{opts}\nStated correct answer: {q.correct_option}"
    return q.text
