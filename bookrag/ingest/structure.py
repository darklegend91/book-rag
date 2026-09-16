"""Rebuild a book's chapter/section hierarchy from typographic signals.

Two sources of truth, in priority order:
  1. The PDF's embedded table of contents (accurate when present).
  2. Font-size outliers: a line noticeably larger than the modal body size,
     short enough to be a title, is treated as a heading.

Every text line ends up tagged with the chapter and section it lives under.
That tag is what lets a chunk cite "Ch. 4 > 4.2 Kinetics, pp. 88-89" and what
lets the paper generator target a syllabus topic instead of a raw string match.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any

from bookrag.schemas import Page

# Numbered headings: "4.2 Reaction Kinetics", "CHAPTER 3", "Unit II - Optics"
_NUMBERED = re.compile(r"^\s*(\d+(\.\d+){0,3})[\.\)]?\s+\S")
_CHAPTER_WORD = re.compile(
    r"^\s*(chapter|unit|part|module|lesson|section|appendix)\b[\s:\-]*([ivxlcdm\d]+)?",
    re.IGNORECASE,
)


def _body_size(pages: list[Page]) -> float:
    """Modal font size across the book = the body text size."""
    counter: Counter[float] = Counter()
    for p in pages:
        for b in p.blocks:
            counter[b["size"]] += len(b["text"])
    return counter.most_common(1)[0][0] if counter else 11.0


def _looks_like_heading(text: str) -> bool:
    t = text.strip()
    if not t or len(t) > 120:
        return False
    if t.endswith((".", ",", ";", ":")) and not _CHAPTER_WORD.match(t):
        return False
    words = t.split()
    if len(words) > 16:
        return False
    return bool(_NUMBERED.match(t) or _CHAPTER_WORD.match(t) or t.istitle() or t.isupper())


def _toc_index(toc: list[Any]) -> dict[int, list[tuple[int, str]]]:
    """Map page number -> [(level, title), ...] from an embedded PDF TOC."""
    index: dict[int, list[tuple[int, str]]] = {}
    for entry in toc or []:
        try:
            level, title, page = entry[0], entry[1], entry[2]
        except (IndexError, TypeError):
            continue
        if isinstance(page, int) and page > 0 and title:
            index.setdefault(page, []).append((int(level), str(title).strip()))
    return index


def annotate_structure(pages: list[Page], meta: dict, heading_ratio: float = 1.15) -> list[dict]:
    """Return a flat list of line records tagged with chapter/section.

    Each record: {text, page, is_heading, level, chapter, section}
    """
    body = _body_size(pages)
    toc = _toc_index(meta.get("toc", []))
    heading_threshold = body * heading_ratio

    chapter, section = "", ""
    lines: list[dict] = []

    for page in pages:
        # A TOC entry landing on this page overrides typographic guessing.
        for level, title in toc.get(page.number, []):
            if level <= 1:
                chapter, section = title, ""
            else:
                section = title

        for block in page.blocks:
            text = block["text"].strip()
            if not text:
                continue

            is_heading = False
            level = 3
            explicit = block.get("level")
            if explicit and len(text) <= 200:
                # Markup (Markdown #, EPUB <h2>, DOCX "Heading 2") states the
                # level outright; inferring it from synthetic font sizes turned
                # every H2 into a new chapter.
                is_heading, level = True, min(int(explicit), 3)
            elif block["size"] >= heading_threshold and _looks_like_heading(text):
                is_heading = True
                # Bigger font => higher in the hierarchy.
                level = 1 if block["size"] >= body * (heading_ratio + 0.25) else 2
            elif block.get("bold") and _CHAPTER_WORD.match(text) and len(text) < 90:
                is_heading, level = True, 1

            if is_heading:
                if level == 1:
                    chapter, section = text, ""
                else:
                    section = text

            lines.append({
                "text": text,
                "page": page.number,
                "is_heading": is_heading,
                "level": level,
                "chapter": chapter,
                "section": section,
            })

    return lines


def outline(lines: list[dict]) -> list[dict]:
    """Condense annotated lines into a browsable outline (used by the UI/CLI)."""
    seen: list[dict] = []
    for ln in lines:
        if not ln["is_heading"]:
            continue
        if seen and seen[-1]["title"] == ln["text"]:
            continue
        seen.append({"title": ln["text"], "level": ln["level"],
                     "page": ln["page"], "chapter": ln["chapter"]})
    return seen
