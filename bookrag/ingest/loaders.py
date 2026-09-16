"""Document loaders.

The PDF loader is deliberately layout-aware: instead of dumping `page.get_text()`
we walk the span tree so we keep font sizes and boldness. Those signals are what
`structure.py` uses to rebuild the chapter/section hierarchy, which in turn is
what makes citations trustworthy and chunk boundaries semantic.
"""
from __future__ import annotations

import hashlib
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

from bookrag.schemas import Page

SUPPORTED = {".pdf", ".epub", ".txt", ".md", ".docx"}


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def book_id_for(path: Path) -> str:
    """Stable id from the file's content, so re-ingesting the same file is
    idempotent. Name + size collided for different files that happened to share
    both; hashing the bytes also makes a byte-identical copy detectable."""
    h = file_sha256(path)[:12]
    return f"{re.sub(r'[^a-z0-9]+', '-', path.stem.lower()).strip('-')[:40]}-{h}"


def discover_books(books_dir: Path) -> list[Path]:
    return sorted(p for p in books_dir.rglob("*") if p.suffix.lower() in SUPPORTED)


# --------------------------------------------------------------------------
# PDF
# --------------------------------------------------------------------------
def load_pdf(path: Path) -> tuple[list[Page], dict]:
    import pymupdf as fitz  # PyMuPDF (>=1.24 module name)

    doc = fitz.open(path)
    meta = {
        "title": (doc.metadata or {}).get("title") or path.stem,
        "author": (doc.metadata or {}).get("author") or "",
        "n_pages": doc.page_count,
        "toc": doc.get_toc() or [],          # [[level, title, page], ...]
    }

    raw_pages: list[Page] = []
    for pno in range(doc.page_count):
        page = doc.load_page(pno)
        data = page.get_text("dict")
        blocks: list[dict] = []
        for block in data.get("blocks", []):
            if block.get("type") != 0:        # skip images
                continue
            for line in block.get("lines", []):
                text = "".join(s.get("text", "") for s in line.get("spans", []))
                if not text.strip():
                    continue
                spans = line.get("spans", [])
                size = max((s.get("size", 0) for s in spans), default=0)
                flags = spans[0].get("flags", 0) if spans else 0
                blocks.append({
                    "text": text.rstrip(),
                    "size": round(size, 1),
                    "bold": bool(flags & 2 ** 4),
                    "y": round(line.get("bbox", [0, 0, 0, 0])[1], 1),
                    "page": pno + 1,
                })
        raw_pages.append(Page(number=pno + 1, text="\n".join(b["text"] for b in blocks), blocks=blocks))

    doc.close()
    return raw_pages, meta


def strip_page_furniture(pages: list[Page]) -> list[Page]:
    """Remove running headers/footers.

    A line repeated near the top or bottom of many pages is furniture, not
    content. Left in, it pollutes every chunk and drags down retrieval precision.
    """
    if len(pages) < 5:
        return pages

    def norm(t: str) -> str:
        return re.sub(r"\d+", "#", t.strip().lower())

    def edges(p: Page) -> list[bool]:
        if not p.blocks:
            return []
        ys = [b["y"] for b in p.blocks]
        top, bottom = min(ys), max(ys)
        return [b["y"] <= top + 2 or b["y"] >= bottom - 2 for b in p.blocks]

    edge_counter: Counter[str] = Counter()
    for p in pages:
        for b, at_edge in zip(p.blocks, edges(p)):
            if at_edge:
                edge_counter[norm(b["text"])] += 1

    threshold = max(3, int(len(pages) * 0.30))
    furniture = {t for t, c in edge_counter.items() if c >= threshold and len(t) < 120}

    cleaned: list[Page] = []
    for p in pages:
        # Only the edge occurrence is furniture. The same words in the body
        # (a chapter title quoted mid-page, an equation number) are content,
        # so they stay -- flagged, so text_quality measures what it always did.
        kept = []
        for b, at_edge in zip(p.blocks, edges(p)):
            if norm(b["text"]) in furniture:
                if at_edge:
                    continue
                b = {**b, "furniture_pattern": True}
            kept.append(b)
        cleaned.append(Page(number=p.number, text="\n".join(b["text"] for b in kept), blocks=kept))
    return cleaned


# --------------------------------------------------------------------------
# Text-layer quality
# --------------------------------------------------------------------------
def text_quality(pages: list[Page]) -> dict:
    """Measure how badly a PDF's text layer is fragmented.

    Slide decks exported from PowerPoint (and many figure-heavy PDFs) store
    every glyph as its own positioned block. Extraction then yields a stream of
    one-character lines in near-arbitrary order -- equations especially, since
    their sub/superscripts are separate fragments. No reading-order heuristic
    recovers them: the word order inside a phrase is often reversed too.

    Such a book indexes and retrieves fine and then answers nonsense, which is
    the worst possible failure mode for a system whose whole promise is
    faithfulness. Better to detect it at ingest and say so.
    """
    # Mid-page lines matching a header/footer pattern (in practice mostly bare
    # numbers: equation labels, axis ticks) are left out of the measurement.
    # The thresholds below were calibrated when cleanup deleted those lines
    # everywhere; keeping them in the text must not change which books pass.
    lens = [len(b["text"].strip()) for p in pages for b in p.blocks
            if b["text"].strip() and not b.get("furniture_pattern")]
    n_pages = max(len(pages), 1)
    if not lens:
        return {"lines": 0, "median_len": 0.0, "fragment_ratio": 1.0,
                "chars_per_page": 0.0, "ok": False, "reason": "no text layer"}
    lens.sort()
    median = float(lens[len(lens) // 2])
    frag = sum(1 for x in lens if x <= 3) / len(lens)
    per_page = sum(lens) / n_pages

    # Measured on real files: a normal textbook runs median 83, fragment_ratio
    # 0.00, ~2000 chars/page; a glyph-shredded lecture deck runs median 1,
    # fragment_ratio 0.77; a scanned book runs 0.1 chars/page (its only "text"
    # is a watermark, which is short but not fragmented -- so density, not
    # fragmentation, is what catches it).
    reason = ""
    if per_page < 100.0:
        reason = "no usable text layer (scanned images?)"
    elif median < 12.0 or frag > 0.35:
        reason = "text layer fragmented into per-glyph pieces"
    return {"lines": len(lens), "median_len": median, "fragment_ratio": round(frag, 3),
            "chars_per_page": round(per_page, 1), "ok": not reason, "reason": reason}


# --------------------------------------------------------------------------
# EPUB / DOCX / plain text
# --------------------------------------------------------------------------
def load_epub(path: Path) -> tuple[list[Page], dict]:
    from bs4 import BeautifulSoup
    from ebooklib import epub, ITEM_DOCUMENT

    book = epub.read_epub(str(path))
    title = (book.get_metadata("DC", "title") or [[path.stem]])[0][0]
    pages: list[Page] = []
    for i, item in enumerate(book.get_items_of_type(ITEM_DOCUMENT), start=1):
        soup = BeautifulSoup(item.get_content(), "html.parser")
        blocks = []
        for el in soup.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
            text = el.get_text(" ", strip=True)
            if not text:
                continue
            # Map heading levels onto synthetic font sizes so the shared
            # structure detector works identically for EPUB and PDF, and pass
            # the real level too: a size ratio can't tell h1 from h2 reliably.
            size = {"h1": 20.0, "h2": 17.0, "h3": 15.0, "h4": 13.5}.get(el.name, 11.0)
            block = {"text": text, "size": size, "bold": size > 11.0, "y": 0.0, "page": i}
            if el.name.startswith("h"):
                block["level"] = int(el.name[1])
            blocks.append(block)
        if blocks:
            pages.append(Page(number=i, text="\n".join(b["text"] for b in blocks), blocks=blocks))
    return pages, {"title": title, "author": "", "n_pages": len(pages), "toc": []}


def load_docx(path: Path) -> tuple[list[Page], dict]:
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    d = docx.Document(str(path))
    blocks = []
    # Walk the body in document order: `d.paragraphs` skips tables entirely,
    # and a fact that only lives in a table cell would never be indexed.
    for child in d.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            para = Paragraph(child, d)
            text = para.text.strip()
            if not text:
                continue
            style = (para.style.name or "").lower() if para.style is not None else ""
            m = re.search(r"heading\s*(\d)", style)
            level = int(m.group(1)) if m else (1 if style == "title" else 0)
            size = {1: 18.0, 2: 16.0}.get(level, 14.0 if level else 11.0)
            block = {"text": text, "size": size, "bold": size > 11.0, "y": 0.0, "page": 1}
            if level:
                block["level"] = level
            blocks.append(block)
        elif tag == "tbl":
            for row in Table(child, d).rows:
                cells, seen = [], set()
                for cell in row.cells:
                    # Merged cells repeat the same underlying cell per column.
                    if id(cell._tc) in seen:
                        continue
                    seen.add(id(cell._tc))
                    t = " ".join(cell.text.split())
                    if t:
                        cells.append(t)
                if cells:
                    blocks.append({"text": " | ".join(cells), "size": 11.0,
                                   "bold": False, "y": 0.0, "page": 1})
    return [Page(number=1, text="\n".join(b["text"] for b in blocks), blocks=blocks)], \
           {"title": path.stem, "author": "", "n_pages": 1, "toc": []}


def load_text(path: Path) -> tuple[list[Page], dict]:
    content = path.read_text(encoding="utf-8", errors="ignore")
    blocks = []
    for line in content.splitlines():
        if not line.strip():
            continue
        # Markdown headings become structural signals.
        m = re.match(r"^(#{1,4})\s+(.*)", line)
        size = (20.0 - 2.0 * (len(m.group(1)) - 1)) if m else 11.0
        block = {"text": m.group(2).strip() if m else line.strip(),
                 "size": size, "bold": bool(m), "y": 0.0, "page": 1}
        if m:
            block["level"] = len(m.group(1))
        blocks.append(block)
    return [Page(number=1, text="\n".join(b["text"] for b in blocks), blocks=blocks)], \
           {"title": path.stem, "author": "", "n_pages": 1, "toc": []}


def load_document(path: Path, drop_furniture: bool = True) -> tuple[list[Page], dict]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        pages, meta = load_pdf(path)
        if drop_furniture:
            pages = strip_page_furniture(pages)
    elif suffix == ".epub":
        pages, meta = load_epub(path)
    elif suffix == ".docx":
        pages, meta = load_docx(path)
    elif suffix in {".txt", ".md"}:
        pages, meta = load_text(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    meta["path"] = str(path)
    meta["book_id"] = book_id_for(path)
    if not meta.get("title"):
        meta["title"] = path.stem
    return pages, meta
