"""The exam blueprint: what to ask, how much, and at what cognitive level.

A blueprint is the contract between "here are some books" and "here is a paper".
It can come from three places:
  * config defaults,
  * a user-supplied JSON/YAML file,
  * a PYQ profile inferred from previous-year papers (see pyq.py).
Topics can be given explicitly, or discovered from the index's chapter structure.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

import yaml

BLOOM_LEVELS = ["remember", "understand", "apply", "analyze", "evaluate", "create"]

_NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                 "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
_ATTEMPT = re.compile(
    r"\b(?:answer|attempt)\s+(?:any\s+)?(\d+|" + "|".join(_NUMBER_WORDS) + r")\b",
    re.IGNORECASE)


def attempt_from_instructions(text: str) -> int:
    """'Answer ANY TWO.' -> 2, 'Attempt any 3 questions' -> 3; 0 if unstated."""
    m = _ATTEMPT.search(text or "")
    if not m:
        return 0
    word = m.group(1).lower()
    return int(word) if word.isdigit() else _NUMBER_WORDS[word]


@dataclass
class SectionSpec:
    name: str
    type: str = "short"            # short | long | mcq | numerical | truefalse
    count: int = 5
    marks_each: int = 2
    instructions: str = ""
    bloom_mix: dict[str, float] = field(default_factory=dict)
    topics: list[str] = field(default_factory=list)
    attempt: int = 0               # questions a candidate answers; 0 = all

    @property
    def answer_count(self) -> int:
        """How many of this section's questions count towards the total. An
        explicit `attempt` wins; otherwise it is read off the instructions."""
        n = self.attempt or attempt_from_instructions(self.instructions)
        return n if 0 < n < self.count else self.count

    @property
    def total_marks(self) -> int:
        return self.answer_count * self.marks_each


@dataclass
class Blueprint:
    title: str = "End-Semester Examination"
    subject: str = ""
    duration: str = "3 Hours"
    total_marks: int = 70
    sections: list[SectionSpec] = field(default_factory=list)
    bloom_mix: dict[str, float] = field(default_factory=dict)
    topics: list[str] = field(default_factory=list)
    style_guide: str = ""          # filled in from PYQs when available
    general_instructions: list[str] = field(default_factory=list)

    @property
    def computed_marks(self) -> int:
        return sum(s.total_marks for s in self.sections)

    def problems(self) -> list[str]:
        """Structural errors that would make generation or marking wrong."""
        out: list[str] = []
        if not self.sections:
            out.append("the paper has no sections")
        seen: set[str] = set()
        for i, s in enumerate(self.sections, start=1):
            name = (s.name or "").strip()
            if not name:
                out.append(f"section {i} has no name")
                continue
            if name.casefold() in seen:
                # Questions are grouped and marked by section name, so two
                # sections sharing one would have their marks mixed together.
                out.append(f"two sections are named '{name}'; section names must be unique")
            seen.add(name.casefold())
            if s.count < 1:
                out.append(f"section '{name}' asks for {s.count} questions")
            if s.marks_each < 1:
                out.append(f"section '{name}' gives {s.marks_each} marks per question")
            if s.attempt < 0 or s.attempt > s.count:
                out.append(f"section '{name}' says attempt {s.attempt} of {s.count} questions")
        return out

    def validate(self) -> None:
        problems = self.problems()
        if problems:
            raise ValueError("Invalid paper blueprint: " + "; ".join(problems))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["sections"] = [asdict(s) for s in self.sections]
        return d

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def from_config(cfg) -> Blueprint:
    sections = [
        SectionSpec(name=s.get("name", f"Section {i+1}"),
                    type=s.get("type", "short"),
                    count=int(s.get("count", 5)),
                    marks_each=int(s.get("marks_each", 2)),
                    instructions=s.get("instructions", ""),
                    attempt=int(s.get("attempt", 0) or 0))
        for i, s in enumerate(cfg.get("paper.sections", []) or [])
    ]
    return Blueprint(
        title=cfg.get("paper.title", "End-Semester Examination"),
        duration=cfg.get("paper.duration", "3 Hours"),
        total_marks=int(cfg.get("paper.total_marks", 70)),
        sections=sections,
        bloom_mix=dict(cfg.get("paper.bloom_mix", {}) or {}),
    )


def load(path: str | Path) -> Blueprint:
    p = Path(path)
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) if p.suffix in {".yaml", ".yml"} \
        else json.loads(p.read_text(encoding="utf-8"))
    sections = [SectionSpec(**s) for s in raw.pop("sections", [])]
    return Blueprint(sections=sections, **raw)


def assign_blooms(count: int, mix: dict[str, float]) -> list[str]:
    """Turn a proportional Bloom mix into a concrete per-question assignment.

    Largest-remainder allocation, so a 0.15 share of 10 questions reliably yields
    at least one rather than rounding away to nothing.
    """
    mix = {k: v for k, v in (mix or {}).items() if k in BLOOM_LEVELS and v > 0}
    if not mix or count <= 0:
        return ["understand"] * max(0, count)
    total = sum(mix.values())
    exact = {k: (v / total) * count for k, v in mix.items()}
    base = {k: int(v) for k, v in exact.items()}
    remainder = count - sum(base.values())
    for k, _ in sorted(exact.items(), key=lambda kv: -(kv[1] - int(kv[1])))[:remainder]:
        base[k] += 1
    out: list[str] = []
    for level in BLOOM_LEVELS:          # emit in cognitive order
        out.extend([level] * base.get(level, 0))
    return out[:count] or ["understand"] * count


def bloom_for_marks(marks: int) -> dict[str, float]:
    """Sensible default cognitive mix when marks imply the depth expected."""
    if marks <= 2:
        return {"remember": 0.5, "understand": 0.5}
    if marks <= 5:
        return {"understand": 0.4, "apply": 0.4, "analyze": 0.2}
    return {"apply": 0.3, "analyze": 0.4, "evaluate": 0.2, "create": 0.1}


def discover_topics(store, book_ids: list[str] | None = None,
                    max_topics: int = 40) -> list[str]:
    """Derive a topic list from detected chapters, weighted by content volume."""
    weight: dict[str, int] = {}
    for c in store.chunks:
        if book_ids and c.book_id not in book_ids:
            continue
        key = c.chapter or c.section or c.book_title
        weight[key] = weight.get(key, 0) + c.token_count
    ranked = sorted(weight.items(), key=lambda kv: -kv[1])
    return [t for t, _ in ranked[:max_topics] if t.strip()]
