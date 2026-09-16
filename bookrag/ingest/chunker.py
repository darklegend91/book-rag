"""Structure-aware, token-accurate chunking.

Why not a naive character splitter: exam questions hinge on complete definitions,
derivations and worked examples. Splitting mid-derivation
produces chunks that retrieve well but answer badly. So we:

  * never merge across a chapter boundary,
  * prefer to break at section boundaries, then paragraph, then sentence,
    then word, and treat chunk_tokens as a hard ceiling,
  * size by real tokens (tiktoken) rather than characters,
  * overlap consecutive chunks so a concept split across a boundary is
    recoverable from either side,
  * cite each chunk by the pages its own text came from, not the pages of the
    whole paragraph or section it was cut out of, and by every section it
    spans ("2.1 Entropy – 2.3 The Carnot Cycle"), not just the first,
  * prepend "Book > Chapter > Section" to the *embedded* text only, which
    measurably lifts retrieval on topic-style queries without polluting the
    text the LLM quotes back.
"""
from __future__ import annotations

import re
from typing import Iterator

from bookrag.schemas import Chunk

_ENC = None


def _encoder():
    global _ENC
    if _ENC is None:
        try:
            import tiktoken
            _ENC = tiktoken.get_encoding("cl100k_base")
        except Exception:                      # offline / no tiktoken cache
            _ENC = False
    return _ENC


def count_tokens(text: str) -> int:
    enc = _encoder()
    if enc:
        return len(enc.encode(text, disallowed_special=()))
    return max(1, int(len(text) / 4))          # ~4 chars/token fallback


def truncate_tokens(text: str, n_tokens: int) -> str:
    """First ~n_tokens of text; never longer than n_tokens by count_tokens."""
    if n_tokens <= 0:
        return ""
    enc = _encoder()
    if enc:
        ids = enc.encode(text, disallowed_special=())
        return text if len(ids) <= n_tokens else enc.decode(ids[:n_tokens])
    return text[:n_tokens * 4]


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(])", text)
    return [p for p in parts if p.strip()]


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    """(start, end) character offsets of each sentence in text."""
    spans, start = [], 0
    for m in re.finditer(r"(?<=[.!?])\s+(?=[A-Z0-9(])", text):
        if text[start:m.start()].strip():
            spans.append((start, m.start()))
        start = m.end()
    if text[start:].strip():
        spans.append((start, len(text)))
    return spans


def _with_section(sections: list[str], section: str) -> list[str]:
    return sections + [section] if section and section not in sections else sections


def _section_label(sections: list[str]) -> str:
    """'2.1 Entropy', or for a chunk spanning sections '2.1 Entropy – 2.3 The Carnot Cycle'.

    Labelling a merged chunk by its first section alone cited the Carnot cycle
    as "2.1 Definition of Entropy" because 2.1-2.3 were short enough to share
    one chunk.
    """
    if not sections:
        return ""
    return sections[0] if len(sections) == 1 else f"{sections[0]} – {sections[-1]}"


def _group_units(lines: list[dict]) -> Iterator[dict]:
    """Collapse annotated lines into paragraph-level units.

    Blank-line information is gone by this point, so a paragraph ends at a
    heading or a chapter/section change. A unit may run across a page break;
    `page_marks` records the character offset where each line (and so each
    page) begins, so pieces cut from the unit can still be cited exactly.
    """
    buf: list[str] = []
    marks: list[tuple[int, int]] = []          # (offset in unit text, page)
    length = 0
    meta: dict | None = None

    def flush():
        nonlocal buf, marks, length, meta
        out = None
        if buf and meta:
            out = dict(meta)
            out["text"] = " ".join(buf)
            out["page_marks"] = marks
        buf, marks, length, meta = [], [], 0, None
        return out

    for ln in lines:
        starts_new = (
            meta is not None
            and (ln["is_heading"]
                 or ln["chapter"] != meta["chapter"]
                 or ln["section"] != meta["section"])
        )
        if starts_new:
            out = flush()
            if out:
                yield out
        text = ln["text"].strip()
        if not text:
            continue
        if meta is None:
            meta = {"chapter": ln["chapter"], "section": ln["section"],
                    "page_start": ln["page"], "page_end": ln["page"],
                    "is_heading": ln["is_heading"]}
        meta["page_end"] = ln["page"]
        if buf:
            length += 1                        # the joining space
        marks.append((length, ln["page"]))
        buf.append(text)
        length += len(text)

        # Headings stand alone so they attach to the following body text.
        if ln["is_heading"]:
            out = flush()
            if out:
                yield out

    out = flush()
    if out:
        yield out


def chunk_book(
    lines: list[dict],
    book_id: str,
    book_title: str,
    chunk_tokens: int = 700,
    overlap_tokens: int = 120,
    min_tokens: int = 60,
    contextual_headers: bool = True,
) -> list[Chunk]:
    units = list(_group_units(lines))
    chunks: list[Chunk] = []
    chunk_sections: list[list[str]] = []   # sections each chunk spans, parallel to chunks

    cur_text: list[str] = []
    cur_tokens = 0
    cur_meta: dict | None = None
    # Sub-minimum text waiting to be prepended to the next chunk of the same
    # chapter. Kept with its own metadata so it is never cited under another.
    pending: tuple[str, dict] | None = None

    def fits(a: str, b: str) -> bool:
        return count_tokens(f"{a}\n\n{b}") <= chunk_tokens

    def append_chunk(body: str, meta: dict) -> None:
        # Joins and overlaps are budgeted with a margin, but tokenisation of a
        # concatenation is not exactly additive, so enforce the ceiling here.
        parts = _hard_split(body, chunk_tokens) if count_tokens(body) > chunk_tokens else [body]
        for part in parts:
            ordinal = len(chunks) + 1
            chunk = Chunk(
                id=f"{book_id}::{ordinal:05d}",
                text=part,
                embed_text="",
                book_id=book_id,
                book_title=book_title,
                chapter=meta["chapter"],
                section=_section_label(meta["sections"]),
                page_start=meta["page_start"],
                page_end=meta["page_end"],
                ordinal=ordinal,
                token_count=count_tokens(part),
            )
            chunk.embed_text = _embed_text(chunk, contextual_headers)
            chunks.append(chunk)
            chunk_sections.append(list(meta["sections"]))

    def emit(force: bool = False) -> None:
        nonlocal cur_text, cur_tokens, cur_meta, pending
        body = "\n\n".join(cur_text).strip()
        meta = cur_meta
        cur_text, cur_tokens, cur_meta = [], 0, None
        if not body or meta is None:
            return

        if pending:
            p_body, p_meta = pending
            pending = None
            if p_meta["chapter"] == meta["chapter"] and fits(p_body, body):
                body = f"{p_body}\n\n{body}"
                sections = list(p_meta["sections"])
                for s in meta["sections"]:
                    sections = _with_section(sections, s)
                meta = {**meta, "page_start": p_meta["page_start"], "sections": sections}
            else:
                append_chunk(p_body, p_meta)

        # Too small to stand alone: fold it into a same-chapter neighbour
        # instead of creating a low-signal fragment that pollutes retrieval.
        # Backward when the previous chunk has room, otherwise forward.
        if not force and count_tokens(body) < min_tokens:
            prev = chunks[-1] if chunks else None
            if prev and prev.chapter == meta["chapter"] and fits(prev.text, body):
                prev.text = f"{prev.text}\n\n{body}"
                prev.page_end = max(prev.page_end, meta["page_end"])
                for s in meta["sections"]:
                    chunk_sections[-1] = _with_section(chunk_sections[-1], s)
                prev.section = _section_label(chunk_sections[-1])
                prev.token_count = count_tokens(prev.text)
                prev.embed_text = _embed_text(prev, contextual_headers)
            else:
                pending = (body, meta)
            return

        append_chunk(body, meta)

    for unit in units:
        u_text = unit["text"]
        u_tokens = count_tokens(u_text)

        # Hard boundary: a new chapter always starts a new chunk.
        if cur_meta and unit["chapter"] != cur_meta["chapter"]:
            emit()

        # A single oversized unit (long derivation, table) is split by
        # sentence, and any sentence still too long by word.
        if u_tokens > chunk_tokens:
            emit()
            for piece, p_start, p_end in _pack_spans(u_text, unit["page_marks"],
                                                     chunk_tokens, overlap_tokens):
                cur_text, cur_tokens = [piece], count_tokens(piece)
                cur_meta = {"chapter": unit["chapter"], "sections": _with_section([], unit["section"]),
                            "page_start": p_start, "page_end": p_end}
                emit(force=True)
            continue

        # +2 per join: the "\n\n" separator costs a token or so of its own.
        if cur_text and cur_tokens + u_tokens + 2 > chunk_tokens:
            tail = _tail_tokens("\n\n".join(cur_text), overlap_tokens)
            carry_meta = dict(cur_meta) if cur_meta else None
            n_before = len(chunks)
            emit()
            # Overlap only after a real chunk was written (a folded or pending
            # fragment would otherwise be duplicated), and only if it leaves
            # room for the unit.
            if (tail and carry_meta and len(chunks) > n_before
                    and count_tokens(tail) + u_tokens + 2 <= chunk_tokens):
                cur_text, cur_tokens = [tail], count_tokens(tail) + 2
                # The overlap tail was written under the previous chunk's last section.
                cur_meta = {"chapter": unit["chapter"],
                            "sections": list(carry_meta["sections"][-1:]),
                            "page_start": carry_meta["page_end"], "page_end": unit["page_end"]}

        if cur_meta is None:
            cur_meta = {"chapter": unit["chapter"], "sections": [],
                        "page_start": unit["page_start"], "page_end": unit["page_end"]}
        cur_meta["page_end"] = max(cur_meta["page_end"], unit["page_end"])
        # Record every section the chunk covers, in order, so its citation names
        # the whole span rather than only where it starts.
        cur_meta["sections"] = _with_section(cur_meta["sections"], unit["section"])
        cur_text.append(u_text)
        cur_tokens += u_tokens + 2

    emit(force=True)
    if pending:
        append_chunk(*pending)
    return chunks


def _embed_text(chunk: Chunk, contextual: bool) -> str:
    if not contextual:
        return chunk.text
    trail = " > ".join(p for p in (chunk.book_title, chunk.chapter, chunk.section) if p)
    return f"[{trail}]\n{chunk.text}" if trail else chunk.text


def _page_at(marks: list[tuple[int, int]], offset: int) -> int:
    page = marks[0][1]
    for off, p in marks:
        if off > offset:
            break
        page = p
    return page


def _pack_spans(text: str, marks: list[tuple[int, int]], chunk_tokens: int,
                overlap_tokens: int) -> list[tuple[str, int, int]]:
    """Split an oversized unit into (piece, page_start, page_end), each piece
    at most chunk_tokens, overlapping by whole sentences."""
    spans: list[tuple[int, int, int]] = []     # (start, end, tokens)
    for s, e in _sentence_spans(text):
        n = count_tokens(text[s:e])
        if n <= chunk_tokens:
            spans.append((s, e, n))
        else:
            spans.extend(_word_windows(text, s, e, chunk_tokens))

    def piece(buf: list[tuple[int, int, int]]) -> tuple[str, int, int]:
        start, end = buf[0][0], buf[-1][1]
        return text[start:end].strip(), _page_at(marks, start), _page_at(marks, end - 1)

    out: list[tuple[str, int, int]] = []
    buf: list[tuple[int, int, int]] = []
    tok = 0
    for span in spans:
        if buf and tok + span[2] + 1 > chunk_tokens:
            out.append(piece(buf))
            keep: list[tuple[int, int, int]] = []
            kept = 0
            for prev in reversed(buf):
                if kept + prev[2] + 1 > overlap_tokens:
                    break
                keep.insert(0, prev)
                kept += prev[2] + 1
            buf, tok = (keep, kept) if kept + span[2] + 1 <= chunk_tokens else ([], 0)
        buf.append(span)
        tok += span[2] + 1
    if buf:
        out.append(piece(buf))
    return out


def _word_windows(text: str, start: int, end: int, limit: int) -> list[tuple[int, int, int]]:
    """Cut a sentence with no usable punctuation into word runs of <= limit tokens."""
    out: list[tuple[int, int, int]] = []
    w_start: int | None = None
    w_end = start
    tok = 0
    for m in re.finditer(r"\S+", text[start:end]):
        a, b = start + m.start(), start + m.end()
        n = count_tokens(" " + m.group())
        if w_start is not None and tok + n > limit:
            out.append((w_start, w_end, tok))
            w_start, tok = None, 0
        if w_start is None:
            w_start = a
        w_end = b
        tok += n
    if w_start is not None:
        out.append((w_start, w_end, tok))
    return out


def _hard_split(text: str, limit: int) -> list[str]:
    """Last resort: fixed token windows, for text with no spaces to split on."""
    enc = _encoder()
    if enc:
        ids = enc.encode(text, disallowed_special=())
        parts = [enc.decode(ids[i:i + limit]) for i in range(0, len(ids), limit)]
    else:
        parts = [text[i:i + limit * 4] for i in range(0, len(text), limit * 4)]
    return [p.strip() for p in parts if p.strip()]


def _tail_tokens(text: str, n_tokens: int) -> str:
    """Last ~n_tokens of text, snapped to a sentence boundary where possible."""
    if n_tokens <= 0:
        return ""
    enc = _encoder()
    if enc:
        ids = enc.encode(text, disallowed_special=())
        tail = enc.decode(ids[-n_tokens:]) if len(ids) > n_tokens else text
    else:
        tail = text[-n_tokens * 4:]
    m = re.search(r"(?<=[.!?])\s+", tail)
    return tail[m.end():].strip() if m and m.end() < len(tail) - 40 else tail.strip()
