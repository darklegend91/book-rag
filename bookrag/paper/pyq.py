"""Previous-year question papers -> a reusable exam profile.

The point is *pattern transfer*, not copying: PYQs tell us how this examiner
builds a paper (section layout, mark bands, command verbs, phrasing, which
topics recur and how heavily). New questions are then written from the book,
in that pattern. The PYQ text itself never becomes source material for answers —
that would break the "only from the book" guarantee.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path

from bookrag.ingest.loaders import load_document
from bookrag.llm.client import OllamaClient
from bookrag.llm.prompts import PYQ_PARSE_SYSTEM, PYQ_STYLE_SYSTEM
from bookrag.paper.blueprint import Blueprint, SectionSpec


@dataclass
class PYQQuestion:
    number: str = ""
    section: str = ""
    text: str = ""
    marks: int = 0
    type: str = "short"
    bloom: str = "understand"
    verb: str = ""
    topic: str = ""
    source_paper: str = ""


@dataclass
class PYQProfile:
    papers: list[str] = field(default_factory=list)
    questions: list[PYQQuestion] = field(default_factory=list)
    sections: list[dict] = field(default_factory=list)
    total_marks: int = 0
    duration: str = ""
    title: str = ""
    style_guide: str = ""
    topic_frequency: dict[str, int] = field(default_factory=dict)
    verb_frequency: dict[str, int] = field(default_factory=dict)
    bloom_mix: dict[str, float] = field(default_factory=dict)
    marks_distribution: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["questions"] = [asdict(q) for q in self.questions]
        return d

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
                              encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "PYQProfile":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        qs = [PYQQuestion(**q) for q in raw.pop("questions", [])]
        return cls(questions=qs, **raw)

    # -------------------------------------------------- derived views
    def priority_topics(self, top_n: int = 25) -> list[str]:
        """Topics an examiner returns to — the strongest signal PYQs carry."""
        return [t for t, _ in Counter(self.topic_frequency).most_common(top_n)]

    def example_questions(self, qtype: str = "", marks: int = 0, limit: int = 4) -> list[str]:
        """Few-shot exemplars, matched to the slot being generated."""
        pool = [q for q in self.questions
                if (not qtype or q.type == qtype) and (not marks or q.marks == marks)]
        if not pool:
            pool = [q for q in self.questions if not qtype or q.type == qtype] or self.questions
        return [q.text for q in pool[:limit]]

    def to_blueprint(self, title: str | None = None) -> Blueprint:
        """Rebuild the paper's shape so a new paper mirrors its structure."""
        sections = [
            SectionSpec(name=s.get("name", f"Section {i+1}"),
                        type=s.get("type", "short") or "short",
                        count=int(s.get("count", 5) or 5),
                        marks_each=int(s.get("marks_each", 2) or 2),
                        instructions=s.get("instructions", ""))
            for i, s in enumerate(self.sections)
        ]
        if not sections:
            # No section structure detected: rebuild it from the mark bands
            # used, counted per paper and then taken as the typical count.
            # Summing across papers made two 4-mark papers look like one
            # 8-mark paper.
            per_paper: dict[str, Counter] = {}
            for q in self.questions:
                if q.marks:
                    per_paper.setdefault(q.source_paper, Counter())[(q.marks, q.type)] += 1
            bands: dict[tuple[int, str], list[int]] = {}
            for counts in per_paper.values():
                for band, n in counts.items():
                    bands.setdefault(band, []).append(n)
            for i, ((marks, qtype), ns) in enumerate(sorted(bands.items(), key=lambda kv: kv[0][0])):
                count = sorted(ns)[len(ns) // 2]
                sections.append(SectionSpec(name=f"Section {chr(65+i)} ({marks} marks each)",
                                            type=qtype or "short", count=count, marks_each=marks))
        return Blueprint(
            title=title or self.title or "Examination",
            duration=self.duration or "3 Hours",
            total_marks=self.total_marks or sum(s.total_marks for s in sections),
            sections=sections,
            bloom_mix=self.bloom_mix or {},
            topics=self.priority_topics(),
            style_guide=self.style_guide,
        )


def analyze_pyqs(paths: list[Path], llm: OllamaClient, progress=print) -> PYQProfile:
    profile = PYQProfile()
    all_sections: list[dict] = []

    for path in paths:
        progress(f"Parsing PYQ: {path.name}")
        pages, meta = load_document(path)
        text = "\n".join(p.text for p in pages)
        if not text.strip():
            progress(f"  ! no extractable text in {path.name}, skipping")
            continue

        # Papers are short; chunk only if one overflows the context window.
        for part in _split_for_context(text, max_chars=18000):
            try:
                data = llm.complete_json(PYQ_PARSE_SYSTEM, part, fast=False, temperature=0.0)
            except Exception as exc:
                progress(f"  ! parse failed: {exc}")
                continue
            paper_meta = data.get("paper") or {}
            profile.title = profile.title or paper_meta.get("title", "")
            profile.duration = profile.duration or paper_meta.get("duration", "")
            profile.total_marks = profile.total_marks or int(paper_meta.get("total_marks") or 0)
            all_sections.extend(paper_meta.get("sections") or [])
            for q in data.get("questions") or []:
                if not (q.get("text") or "").strip():
                    continue
                profile.questions.append(PYQQuestion(
                    number=str(q.get("number", "")),
                    section=q.get("section", "") or "",
                    text=q["text"].strip(),
                    marks=int(q.get("marks") or 0),
                    type=(q.get("type") or "short").lower(),
                    bloom=(q.get("bloom") or "understand").lower(),
                    verb=(q.get("verb") or "").strip(),
                    topic=(q.get("topic") or "").strip(),
                    source_paper=path.name,
                ))
        profile.papers.append(path.name)
        progress(f"  -> {len(profile.questions)} questions so far")

    profile.sections = _merge_sections(all_sections)
    _compute_statistics(profile)

    if profile.questions:
        progress("Deriving style guide ...")
        sample = "\n".join(f"[{q.marks}m] {q.text}" for q in profile.questions[:60])
        try:
            profile.style_guide = llm.complete(PYQ_STYLE_SYSTEM, sample, temperature=0.2)
        except Exception as exc:
            progress(f"  ! style guide failed: {exc}")
    return profile


def _split_for_context(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    out, buf = [], []
    size = 0
    for line in text.splitlines(keepends=True):
        if size + len(line) > max_chars and buf:
            out.append("".join(buf))
            buf, size = [], 0
        buf.append(line)
        size += len(line)
    if buf:
        out.append("".join(buf))
    return out


def _merge_sections(sections: list[dict]) -> list[dict]:
    """Collapse the same section seen across several papers into one spec."""
    merged: dict[str, dict] = {}
    for s in sections:
        name = (s.get("name") or "").strip()
        if not name:
            continue
        cur = merged.setdefault(name, {"name": name, "instructions": s.get("instructions", ""),
                                       "type": s.get("type", "short"), "counts": [], "marks": []})
        if s.get("count"):
            cur["counts"].append(int(s["count"]))
        if s.get("marks_each"):
            cur["marks"].append(int(s["marks_each"]))
    out = []
    for s in merged.values():
        out.append({
            "name": s["name"],
            "instructions": s["instructions"],
            "type": s["type"],
            "count": Counter(s["counts"]).most_common(1)[0][0] if s["counts"] else 5,
            "marks_each": Counter(s["marks"]).most_common(1)[0][0] if s["marks"] else 2,
        })
    return out


def _compute_statistics(profile: PYQProfile) -> None:
    profile.topic_frequency = dict(Counter(q.topic for q in profile.questions if q.topic))
    profile.verb_frequency = dict(Counter(q.verb.lower() for q in profile.questions if q.verb))
    profile.marks_distribution = {str(k): v for k, v in
                                  Counter(q.marks for q in profile.questions if q.marks).items()}
    blooms = Counter(q.bloom for q in profile.questions if q.bloom)
    total = sum(blooms.values())
    if total:
        profile.bloom_mix = {k: round(v / total, 3) for k, v in blooms.items()}
