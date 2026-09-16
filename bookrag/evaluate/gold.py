"""Evaluation against a reviewed question set (a "gold set").

The self-supervised evals in harness.py grade the system with questions the
system wrote itself. A gold set is written, or at least checked, by a person who
knows the books, and it includes questions the books do NOT answer. It measures
what users experience: does a supported question get an answer that cites the
right pages and contains the right facts, and does an unsupported one get
refused?

File format, JSON Lines (one case per line; blank lines and # comments allowed):

  {"question": "What is the efficiency of a Carnot engine?", "answerable": true,
   "book": "thermal_physics", "pages": [2], "keywords": ["T_cold", "T_hot"],
   "reviewed": true}
  {"question": "How do I bake a chocolate cake?", "answerable": false, "reviewed": true}

  question    required
  answerable  default true
  book        optional; a case-insensitive substring of a book title or id.
              Restricts retrieval to matching books, as the UI's book filter does.
  pages       optional; pages where the answer is printed
  keywords    optional; terms any correct answer must contain
  reviewed    set true once a person has checked the case
  notes       free text
"""
from __future__ import annotations

import json
import random
import re
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from bookrag.chat.engine import _plain_math
from bookrag.config import Config
from bookrag.paper.generator import _strip_source_references

_REFERS_TO_SOURCE = re.compile(r"\b(passages?|excerpts?|the text|the author)\b", re.IGNORECASE)

_FIELDS ={"question", "answerable", "book", "pages", "keywords", "reviewed", "notes"}

DRAFT_SYSTEM = """You write evaluation questions for a question-answering system over books.

Given ONE passage, write ONE question a student might ask whose answer is stated
in this passage, and 2-4 short keywords -- exact words, names, numbers or symbols
copied from the passage -- that any correct answer must contain.

Prefer specific facts, definitions, formulas and named results over vague themes.
Never mention "the passage", "the text" or "the author".

Return JSON only: {"question": "...", "keywords": ["...", "..."]}"""

# Controls: a system that answers these is answering from its own weights.
OFF_TOPIC = [
    "How do I bake a chocolate cake?",
    "Who won the 2022 FIFA World Cup?",
    "What is the capital of Australia?",
    "How do I reset a forgotten Gmail password?",
    "What is the best way to house-train a puppy?",
]


@dataclass
class GoldCase:
    question: str
    answerable: bool = True
    book: str = ""
    pages: list[int] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    reviewed: bool = False
    notes: str = ""


def case_from_dict(data, where: str = "case") -> GoldCase:
    """Validate one case strictly: a typo'd field silently ignored would make
    the case measure something other than what its author intended."""
    if not isinstance(data, dict):
        raise ValueError(f"{where}: each case must be a JSON object")
    unknown = set(data) - _FIELDS
    if unknown:
        raise ValueError(f"{where}: unknown field(s) {sorted(unknown)}; allowed: {sorted(_FIELDS)}")
    question = data.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"{where}: 'question' must be a non-empty string")
    answerable = data.get("answerable", True)
    if not isinstance(answerable, bool):
        raise ValueError(f"{where}: 'answerable' must be true or false")
    reviewed = data.get("reviewed", False)
    if not isinstance(reviewed, bool):
        raise ValueError(f"{where}: 'reviewed' must be true or false")
    book = data.get("book", "") or ""
    if not isinstance(book, str):
        raise ValueError(f"{where}: 'book' must be a string")
    pages = data.get("pages", []) or []
    if not isinstance(pages, list) or not all(
            isinstance(p, int) and not isinstance(p, bool) and p > 0 for p in pages):
        raise ValueError(f"{where}: 'pages' must be a list of positive whole numbers")
    keywords = data.get("keywords", []) or []
    if not isinstance(keywords, list) or not all(isinstance(k, str) and k.strip() for k in keywords):
        raise ValueError(f"{where}: 'keywords' must be a list of non-empty strings")
    notes = data.get("notes", "") or ""
    if not isinstance(notes, str):
        raise ValueError(f"{where}: 'notes' must be a string")
    return GoldCase(question.strip(), answerable, book.strip(), list(pages),
                    [k.strip() for k in keywords], reviewed, notes)


def load_gold(path: Path) -> list[GoldCase]:
    path = Path(path)
    cases: list[GoldCase] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{lineno}: not valid JSON ({exc.msg})") from exc
        cases.append(case_from_dict(data, f"{path}:{lineno}"))
    if not cases:
        raise ValueError(f"{path}: contains no cases")
    return cases


def write_jsonl(cases: list[dict], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cases),
                    encoding="utf-8")
    return path


# ------------------------------------------------------------------ run
def _matching_books(store, book: str) -> list[str]:
    needle = book.casefold()
    return [b["book_id"] for b in store.books()
            if needle in str(b.get("book_id", "")).casefold()
            or needle in str(b.get("title", "")).casefold()]


def keyword_present(keyword: str, answer: str) -> bool:
    """Whether an answer uses a keyword, ignoring notation, case and punctuation.

    Answers write "$\\frac{T_{\\text{cold}}}{T_{\\text{hot}}}$" or "(GSM) method"
    where a keyword says "T_cold / T_hot" or "GSM method"; a literal substring
    test scored those correct answers as missing the keyword. Keywords of one or
    two characters must equal a whole token of the answer (after the same
    clean-up, so "Ea" matches "E_a" and "V+" matches "V^+"), so "P" isn't found
    inside "Pressure".
    """
    def norm(s: str) -> str:
        return unicodedata.normalize("NFKC", _plain_math(s)).casefold()

    def compact(s: str) -> str:
        return re.sub(r"[\W_]+", "", s)

    key, text = compact(norm(keyword)), norm(answer)
    if not key:
        return False
    if len(key) < 3:
        return key in {compact(token) for token in text.split()}
    return key in compact(text)


def _overlaps(chunks, pages: list[int]) -> bool:
    return any(c.page_start <= p <= c.page_end for c in chunks for p in pages)


def run_gold_eval(cfg: Config, cases: list[GoldCase], progress=print, engine=None) -> dict:
    """Ask every case through the real chat pipeline and score the answers."""
    if engine is None:
        from bookrag.chat.engine import ChatEngine
        engine = ChatEngine(cfg)
    store = engine.retriever.store
    rows: list[dict] = []

    for i, case in enumerate(cases, start=1):
        base = {"question": case.question, "answerable": case.answerable,
                "reviewed": case.reviewed}
        book_ids = None
        if case.book:
            book_ids = _matching_books(store, case.book)
            if not book_ids:
                rows.append({**base, "error": f"no indexed book matches {case.book!r}"})
                progress(f"  [{i}/{len(cases)}] ERROR no book matches {case.book!r}")
                continue
        engine.reset()
        started = time.perf_counter()
        try:
            ans = engine.ask(case.question, book_ids=book_ids)
        except Exception as exc:
            rows.append({**base, "error": f"{type(exc).__name__}: {exc}",
                         "seconds": round(time.perf_counter() - started, 2)})
            progress(f"  [{i}/{len(cases)}] ERROR {type(exc).__name__}: {str(exc)[:100]}")
            continue

        row = {**base, "answered": bool(ans.grounded),
               "seconds": round(time.perf_counter() - started, 2),
               "top_score": ans.top_score, "answer": ans.text[:600], "sources": ans.sources()}
        rr = ans.retrieval
        if case.answerable:
            results = rr.results if rr else []
            cited = [results[n - 1].chunk for n in ans.used_citations if 1 <= n <= len(results)]
            if case.pages:
                row["cited_page_hit"] = bool(ans.grounded) and _overlaps(cited, case.pages)
                row["context_page_hit"] = _overlaps([sc.chunk for sc in results], case.pages)
            if case.keywords:
                missing = [k for k in case.keywords
                           if not (ans.grounded and keyword_present(k, ans.text))]
                row["keyword_recall"] = round(1 - len(missing) / len(case.keywords), 3)
                row["missing_keywords"] = missing
            verdict = "answered" if ans.grounded else "REFUSED"
        else:
            row["correct_refusal"] = not ans.grounded
            verdict = "refused" if not ans.grounded else "ANSWERED (should refuse)"
        rows.append(row)
        progress(f"  [{i}/{len(cases)}] {verdict} {row['seconds']}s  {case.question[:70]}")

    return summarize(rows)


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def summarize(rows: list[dict]) -> dict:
    ok = [r for r in rows if "error" not in r]
    answerable = [r for r in ok if r["answerable"]]
    unanswerable = [r for r in ok if not r["answerable"]]
    seconds = sorted(r["seconds"] for r in ok)
    return {
        "n": len(rows),
        "errors": len(rows) - len(ok),
        "unreviewed": sum(1 for r in rows if not r.get("reviewed")),
        "answer_rate": _mean([float(r["answered"]) for r in answerable]),
        "cited_page_hit": _mean([float(r["cited_page_hit"]) for r in answerable
                                 if "cited_page_hit" in r]),
        "context_page_hit": _mean([float(r["context_page_hit"]) for r in answerable
                                   if "context_page_hit" in r]),
        "keyword_recall": _mean([r["keyword_recall"] for r in answerable if "keyword_recall" in r]),
        "correct_refusal_rate": _mean([float(r["correct_refusal"]) for r in unanswerable]),
        "seconds_mean": _mean(seconds),
        "seconds_p95": seconds[min(len(seconds) - 1, int(0.95 * len(seconds)))] if seconds else None,
        "rows": rows,
    }


def summary_text(res: dict) -> str:
    def pct(x):
        return "n/a" if x is None else f"{x:.0%}"

    lines = [
        f"cases {res['n']}  ·  errors {res['errors']}  ·  unreviewed {res['unreviewed']}",
        f"answerable questions: answered {pct(res['answer_rate'])}  ·  cited the right page "
        f"{pct(res['cited_page_hit'])}  ·  right page retrieved {pct(res['context_page_hit'])}  ·  "
        f"keyword recall {pct(res['keyword_recall'])}",
        f"unanswerable questions: correctly refused {pct(res['correct_refusal_rate'])}",
        f"latency: mean {res['seconds_mean']}s  ·  p95 {res['seconds_p95']}s",
    ]
    if res["unreviewed"]:
        lines.append("note: some cases are unreviewed drafts, so treat these numbers as provisional")
    return "\n".join(lines)


# ------------------------------------------------------------------ draft
def draft_gold(cfg: Config, n: int = 30, seed: int = 7, progress=print,
               llm=None, store=None) -> list[dict]:
    """Draft cases from indexed passages for a person to review and correct.

    Books are sampled round-robin so a 700-page book doesn't crowd out a short
    one, keywords are kept only if they really occur in the passage, and the
    off-topic controls are appended. Every draft is marked reviewed: false.
    """
    if store is None:
        from bookrag.index.store import Store
        store = Store(cfg.index_dir).load()
    if llm is None:
        from bookrag.llm.client import client_from_config
        llm = client_from_config(cfg)

    by_book: dict[str, list] = {}
    for c in store.chunks:
        if c.token_count >= 120:
            by_book.setdefault(c.book_id, []).append(c)
    if not by_book:
        raise ValueError("No indexed passages are long enough to draft questions from. "
                         "Build the index first.")
    titles = {b["book_id"]: b.get("title") or b["book_id"] for b in store.books()}
    rng = random.Random(seed)
    pools = {book: rng.sample(chunks, len(chunks)) for book, chunks in sorted(by_book.items())}
    picked = []
    while len(picked) < n and any(pools.values()):
        for book in pools:
            if pools[book] and len(picked) < n:
                picked.append(pools[book].pop())

    cases: list[dict] = []
    for i, chunk in enumerate(picked, start=1):
        try:
            data = llm.complete_json(DRAFT_SYSTEM, chunk.text[:6000], fast=True, temperature=0.2)
        except Exception as exc:
            progress(f"  ! [{i}/{len(picked)}] question writing failed: {str(exc)[:120]}")
            continue
        question = data.get("question") if isinstance(data, dict) else None
        keywords = data.get("keywords") if isinstance(data, dict) else None
        if not isinstance(question, str) or not question.strip():
            progress(f"  ! [{i}/{len(picked)}] unusable reply, skipped")
            continue
        # Measured: the model still writes "...described in the passage?" despite
        # the prompt. A user never sees a passage, so strip such phrases the same
        # way generated exam questions are, and drop what can't be rescued.
        question = _strip_source_references(question.strip())
        if _REFERS_TO_SOURCE.search(question):
            progress(f"  ! [{i}/{len(picked)}] question refers to the source text, skipped")
            continue
        text = chunk.text.casefold()
        kept: dict[str, str] = {}              # casefolded -> first spelling; drops repeats
        for k in (keywords if isinstance(keywords, list) else []):
            if isinstance(k, str) and k.strip() and k.strip().casefold() in text:
                kept.setdefault(k.strip().casefold(), k.strip())
        keywords = list(kept.values())[:4]
        cases.append({
            "question": question.strip(),
            "answerable": True,
            "book": titles.get(chunk.book_id, chunk.book_id),
            "pages": list(range(max(1, chunk.page_start), max(1, chunk.page_end) + 1)),
            "keywords": keywords,
            "reviewed": False,
            "notes": f"drafted from {chunk.id}",
        })
        progress(f"  [{i}/{len(picked)}] {question.strip()[:80]}")

    cases += [{"question": q, "answerable": False, "reviewed": False, "notes": "off-topic control"}
              for q in OFF_TOPIC]
    return cases
