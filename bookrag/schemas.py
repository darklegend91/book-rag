"""Data structures shared by ingestion, retrieval and generation.

Kept dependency-light (plain dataclasses) so index files stay portable JSON.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Chunk:
    """One retrievable unit of a book, carrying enough metadata to cite it."""

    id: str
    text: str                  # raw chunk text (no contextual header)
    embed_text: str            # text actually embedded (may carry a header)
    book_id: str
    book_title: str
    chapter: str = ""
    section: str = ""
    page_start: int = 0
    page_end: int = 0
    ordinal: int = 0           # position of this chunk within its book
    token_count: int = 0

    def citation(self) -> str:
        loc = f"p.{self.page_start}" if self.page_start == self.page_end else \
              f"pp.{self.page_start}-{self.page_end}"
        parts = [self.book_title]
        if self.chapter:
            parts.append(self.chapter)
        if self.section and self.section != self.chapter:
            parts.append(self.section)
        return f"{' > '.join(parts)}, {loc}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Chunk":
        return cls(**d)


@dataclass
class Page:
    """A single extracted page, pre-chunking."""

    number: int
    text: str
    blocks: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float
    dense_rank: int | None = None
    bm25_rank: int | None = None
    rerank_score: float | None = None
    is_neighbor: bool = False  # pulled in as context around a hit, not a hit itself


@dataclass
class Question:
    """A generated exam question plus its provenance and verification state."""

    number: int
    section: str
    text: str
    marks: int
    qtype: str                 # short | long | mcq | numerical | truefalse
    bloom: str
    topic: str
    answer: str = ""
    options: list[str] = field(default_factory=list)   # MCQ only
    correct_option: str = ""
    source_chunk_ids: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    verified: bool = False
    verify_confidence: float = 0.0
    verify_notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
