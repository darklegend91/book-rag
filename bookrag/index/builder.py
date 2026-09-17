"""Ingest pipeline entry point: files on disk -> a searchable index.

Builds are cancellable and resumable. As soon as a book is embedded, its chunks
and vectors are cached under the index dir, keyed by the file's content and
every setting that shapes chunks or vectors. An interrupted build (a Cancel
click, Ctrl+C, a crash) never touches the live index -- that is only replaced by
the atomic save at the very end -- and the next build reuses every cached book
instead of embedding it again.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

from bookrag.config import Config
from bookrag.index.embedder import embedder_from_config
from bookrag.index.store import Store, index_lock
from bookrag.jobs import Cancelled
from bookrag.memory import arbiter_from_config
from bookrag.ingest.chunker import chunk_book
from bookrag.ingest.loaders import discover_books, file_sha256, load_document, text_quality
from bookrag.ingest.structure import annotate_structure, outline
from bookrag.schemas import Chunk

CACHE_DIR = ".build-cache"
OCR_DIR = ".ocr-cache"
# Bump when loading or chunking changes in a way that must invalidate cached books.
# 2: chunks spanning several sections are labelled with the whole range.
_CACHE_VERSION = 2
_CACHE_SETTINGS = (
    "embedding.model", "embedding.passage_prefix", "embedding.normalize",
    "ingest.chunk_tokens", "ingest.chunk_overlap_tokens", "ingest.min_chunk_tokens",
    "ingest.contextual_headers", "ingest.drop_headers_footers", "ingest.heading_size_ratio",
    "ingest.ocr",
)
# Chunks per embedding call; also how often a cancel request is noticed.
_EMBED_BATCH = 64


def build_index(cfg: Config, paths: list[Path] | None = None,
                progress=print, replace: bool | None = None,
                should_cancel: Callable[[], bool] | None = None) -> dict:
    """Index documents.

    With no `paths`, the whole books dir is rebuilt from scratch. With `paths`,
    those files are added to the existing index (a file already indexed under
    the same path is refreshed) unless `replace` is true.

    `should_cancel` is polled between books and embedding batches; when it
    returns true the build raises `Cancelled` and the existing index is left
    as it was.
    """
    files = paths or discover_books(cfg.books_dir)
    if not files:
        raise FileNotFoundError(
            f"No documents found in {cfg.books_dir}. Drop your PDFs there and re-run."
        )
    if replace is None:
        replace = paths is None
    with index_lock(cfg.index_dir):
        return _build(cfg, [Path(p) for p in files], replace, progress,
                      should_cancel or (lambda: False))


def cached_book_count(cfg: Config) -> int:
    """Books embedded by an interrupted build, waiting to be reused."""
    d = Path(cfg.index_dir) / CACHE_DIR
    return len(list(d.glob("*.json"))) if d.is_dir() else 0


def _text_key(text: str) -> str:
    return hashlib.sha1(" ".join(text.lower().split()).encode()).hexdigest()


# ------------------------------------------------------------------ OCR
def _ocr_enabled(cfg: Config) -> bool:
    return str(cfg.get("ingest.ocr", "auto")).strip().lower() in {"auto", "true", "on", "yes"}


def _ocr_copy(cfg: Config, path: Path, sha: str, progress) -> Path | None:
    """OCR a PDF whose text layer is unusable, into a cached copy.

    Scanned books and glyph-shredded slide exports are unreadable as extracted,
    but `ocrmypdf --force-ocr` rebuilds a real text layer. Returns the OCR'd
    copy, or None when the tool is missing or fails (the book is then skipped
    exactly as before, with the reason).
    """
    exe = shutil.which("ocrmypdf")
    if not exe:
        progress("  OCR is enabled but `ocrmypdf` isn't installed "
                 "(macOS: `brew install ocrmypdf`), so this book can't be recovered.")
        return None
    out = Path(cfg.index_dir) / OCR_DIR / f"{sha[:24]}.pdf"
    if out.exists():
        progress("  OCR: reusing an earlier OCR of this file")
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".partial.pdf")
    progress(f"  OCR: running ocrmypdf on {path.name} (large books take several minutes) ...")
    try:
        subprocess.run([exe, "--force-ocr", "--quiet", str(path), str(tmp)],
                       check=True, capture_output=True, text=True,
                       timeout=int(cfg.get("ingest.ocr_timeout_s", 3600)))
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        detail = (getattr(exc, "stderr", None) or str(exc)).strip()
        progress(f"  ! OCR failed for {path.name}: {detail[:200]}")
        tmp.unlink(missing_ok=True)
        return None
    os.replace(tmp, out)
    return out


# ------------------------------------------------------------------ cache
def _cache_key(cfg: Config, sha: str) -> str:
    settings = {k: cfg.get(k) for k in _CACHE_SETTINGS}
    blob = json.dumps([_CACHE_VERSION, sha, settings], sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:24]


def _cache_load(d: Path, key: str):
    meta_p, vec_p = d / f"{key}.json", d / f"{key}.npy"
    if not (meta_p.exists() and vec_p.exists()):
        return None
    try:
        data = json.loads(meta_p.read_text(encoding="utf-8"))
        vectors = np.load(vec_p)
        chunks = [Chunk.from_dict(c) for c in data["chunks"]]
    except Exception:
        return None                              # unreadable entry: rebuild that book
    if len(chunks) != len(vectors):
        return None
    return data["record"], chunks, vectors


def _cache_save(d: Path, key: str, record: dict, chunks: list[Chunk],
                vectors: np.ndarray) -> None:
    d.mkdir(parents=True, exist_ok=True)
    tmp_vec = d / f".{key}.npy.tmp"
    with open(tmp_vec, "wb") as fh:
        np.save(fh, vectors.astype(np.float32))
    os.replace(tmp_vec, d / f"{key}.npy")
    tmp_meta = d / f".{key}.json.tmp"
    tmp_meta.write_text(json.dumps({"record": record, "chunks": [c.to_dict() for c in chunks]},
                                   ensure_ascii=False), encoding="utf-8")
    # Written last: the .json's presence is what marks an entry complete.
    os.replace(tmp_meta, d / f"{key}.json")


def _cache_drop(d: Path, keys: list[str]) -> None:
    for key in keys:
        for suffix in (".json", ".npy"):
            try:
                (d / f"{key}{suffix}").unlink()
            except FileNotFoundError:
                pass


# ------------------------------------------------------------------ build
def _build(cfg: Config, files: list[Path], replace: bool, progress,
           should_cancel: Callable[[], bool]) -> dict:
    started = time.time()
    model = cfg.get("embedding.model")
    cache = Path(cfg.index_dir) / CACHE_DIR

    def checkpoint() -> None:
        if should_cancel():
            raise Cancelled("Index build cancelled. The existing index is unchanged; "
                            "books embedded so far are cached for the next build.")

    existing: Store | None = None
    if not replace and Store.exists(cfg.index_dir):
        existing = Store(cfg.index_dir).load()
        built_with = existing.manifest.get("embedding_model")
        if built_with and built_with != model:
            raise ValueError(
                f"The existing index was embedded with {built_with}, but embedding.model "
                f"is now {model}. Vectors from different models can't be searched together; "
                f"rebuild the whole library instead (`ingest` without --path)."
            )
    old_books = existing.books() if existing else []
    by_hash = {b["sha256"]: b for b in old_books if b.get("sha256")}
    by_path = {b.get("path"): b for b in old_books}

    removed_ids: set[str] = set()
    seen_hashes: dict[str, str] = {}               # sha256 -> file name, this run
    book_records: list[dict] = []
    per_book: list[tuple[list[Chunk], np.ndarray]] = []
    used_keys: list[str] = []
    skipped: list[dict] = []
    min_quality = bool(cfg.get("ingest.skip_fragmented", True))

    embedder = None

    def embed(chunks: list[Chunk]) -> np.ndarray:
        nonlocal embedder
        if embedder is None:
            arbiter_from_config(cfg, log=progress).before("ingest")
            embedder = embedder_from_config(cfg)
        texts = [c.embed_text for c in chunks]
        parts = []
        for i in range(0, len(texts), _EMBED_BATCH):
            checkpoint()
            parts.append(embedder.embed_passages(texts[i:i + _EMBED_BATCH], show_progress=False))
            done = min(i + _EMBED_BATCH, len(texts))
            if len(texts) > _EMBED_BATCH and (done == len(texts) or (i // _EMBED_BATCH) % 5 == 4):
                progress(f"  embedded {done}/{len(texts)} chunks")
        return np.concatenate(parts) if parts else np.zeros((0, 0), np.float32)

    try:
        for path in files:
            checkpoint()
            progress(f"Reading {path.name} ...")
            sha = file_sha256(path)
            if sha in seen_hashes:
                progress(f"  ! {path.name} is byte-identical to {seen_hashes[sha]}. SKIPPED.")
                skipped.append({"path": str(path), "duplicate_of": seen_hashes[sha]})
                continue
            if sha in by_hash and by_hash[sha].get("path") != str(path):
                progress(f"  ! {path.name} is byte-identical to already-indexed "
                         f"{Path(by_hash[sha].get('path', '')).name}. SKIPPED.")
                skipped.append({"path": str(path), "duplicate_of": by_hash[sha].get("path")})
                continue
            seen_hashes[sha] = path.name
            # Re-ingesting a path already in the index replaces its old entry.
            if str(path) in by_path:
                removed_ids.add(by_path[str(path)]["book_id"])
            if sha in by_hash:
                removed_ids.add(by_hash[sha]["book_id"])

            key = _cache_key(cfg, sha)
            hit = _cache_load(cache, key)
            if hit:
                record, chunks, vectors = hit
                record = {**record, "path": str(path)}
                progress(f"  -> {len(chunks)} chunks reused from an interrupted build")
            else:
                try:
                    pages, meta = load_document(
                        path, drop_furniture=bool(cfg.get("ingest.drop_headers_footers", True))
                    )
                except Cancelled:
                    raise
                except Exception as exc:
                    # One corrupt or hostile upload must not block indexing for
                    # everyone sharing the library.
                    progress(f"  ! {path.name} could not be read ({exc}). SKIPPED.")
                    skipped.append({"path": str(path), "error": str(exc)[:500]})
                    continue
                q = text_quality(pages)
                if not q["ok"] and path.suffix.lower() == ".pdf" and _ocr_enabled(cfg):
                    ocr_pdf = _ocr_copy(cfg, path, sha, progress)
                    if ocr_pdf is not None:
                        ocr_pages, ocr_meta = load_document(
                            ocr_pdf,
                            drop_furniture=bool(cfg.get("ingest.drop_headers_footers", True)))
                        ocr_q = text_quality(ocr_pages)
                        progress("  OCR text layer: " + ("usable" if ocr_q["ok"]
                                                         else f"still unusable ({ocr_q['reason']})"))
                        if ocr_q["ok"]:
                            # Keep the original file's identity, not the cache copy's.
                            ocr_meta.update(book_id=meta["book_id"], title=meta["title"],
                                            path=str(path))
                            pages, meta, q = ocr_pages, ocr_meta, ocr_q
                if not q["ok"]:
                    note = (f"  ! {path.name}: {q['reason']} "
                            f"(median line {q['median_len']:.0f} chars, "
                            f"{q['chars_per_page']:.0f} chars/page, "
                            f"{q['fragment_ratio']:.0%} of lines <= 3 chars). "
                            f"Run it through OCR (`ocrmypdf --force-ocr`) to make it usable.")
                    if min_quality:
                        progress(note + " SKIPPED — set ingest.skip_fragmented: false to index anyway.")
                        skipped.append({"path": str(path), "quality": q})
                        continue
                    progress(note + " Indexing anyway (ingest.skip_fragmented is false).")
                meta["text_quality"] = q
                lines = annotate_structure(
                    pages, meta, heading_ratio=float(cfg.get("ingest.heading_size_ratio", 1.15))
                )
                chunks = chunk_book(
                    lines,
                    book_id=meta["book_id"],
                    book_title=meta["title"],
                    chunk_tokens=int(cfg.get("ingest.chunk_tokens", 700)),
                    overlap_tokens=int(cfg.get("ingest.chunk_overlap_tokens", 120)),
                    min_tokens=int(cfg.get("ingest.min_chunk_tokens", 60)),
                    contextual_headers=bool(cfg.get("ingest.contextual_headers", True)),
                )
                record = {
                    "book_id": meta["book_id"],
                    "title": meta["title"],
                    "author": meta.get("author", ""),
                    "path": str(path),
                    "sha256": sha,
                    "n_pages": meta.get("n_pages", 0),
                    "n_chunks": len(chunks),
                    "outline": outline(lines)[:400],
                    "text_quality": q,
                }
                if not chunks:
                    progress(f"  ! {path.name}: produced no chunks despite passing the "
                             f"quality check — nothing from it will ever be retrieved.")
                vectors = embed(chunks) if chunks else np.zeros((0, 0), np.float32)
                _cache_save(cache, key, record, chunks, vectors)
                progress(f"  -> {len(chunks)} chunks, {meta.get('n_pages', 0)} pages")
            used_keys.append(key)
            book_records.append(record)
            per_book.append((chunks, vectors))
    finally:
        if embedder is not None and cfg.get("memory.unload_after_ingest", True):
            # Ingest is done with the encoder; hand its ~1.2 GB back before the
            # caller moves on to chat or paper generation.
            embedder.unload()

    # Carry forward what this run doesn't touch.
    if existing:
        keep = [i for i, c in enumerate(existing.chunks) if c.book_id not in removed_ids]
        kept_chunks = [existing.chunks[i] for i in keep]
        kept_vectors = existing.vectors[keep]
        kept_books = [b for b in old_books if b["book_id"] not in removed_ids]
        touched = {s["path"] for s in skipped} | {b["path"] for b in book_records}
        skipped += [s for s in existing.manifest.get("skipped", []) if s.get("path") not in touched]
        if not any(len(c) for c, _ in per_book) and not removed_ids:
            _cache_drop(cache, used_keys)
            progress("Nothing new to index; the existing index is unchanged.")
            return existing.manifest
    else:
        kept_chunks, kept_vectors, kept_books = [], None, []

    # Identical passages (a book and its reprint, repeated boilerplate) add
    # nothing to retrieval but crowd distinct passages out of the top-k.
    seen_text = {_text_key(c.text) for c in kept_chunks}
    new_chunks: list[Chunk] = []
    vec_parts: list[np.ndarray] = []
    dropped = 0
    for record, (chunks, vectors) in zip(book_records, per_book):
        rows = []
        for i, c in enumerate(chunks):
            k = _text_key(c.text)
            if k in seen_text:
                dropped += 1
                continue
            seen_text.add(k)
            rows.append(i)
        record["n_chunks"] = len(rows)
        new_chunks.extend(chunks[i] for i in rows)
        if rows:
            vec_parts.append(vectors[rows])
    if dropped:
        progress(f"Dropped {dropped} chunk(s) duplicating indexed text.")

    all_chunks = kept_chunks + new_chunks
    if not all_chunks:
        raise ValueError(
            "Nothing to index: every document was skipped as unreadable or duplicate. "
            "See the messages above, or set ingest.skip_fragmented: false."
        )

    parts = ([kept_vectors] if kept_vectors is not None and len(kept_vectors) else []) + vec_parts
    vectors = np.concatenate(parts).astype(np.float32) if parts else np.zeros((0, 0), np.float32)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "embedding_model": model,
        "embedding_dim": int(vectors.shape[1]) if len(vectors) else 0,
        "chunk_tokens": cfg.get("ingest.chunk_tokens"),
        "n_chunks": len(all_chunks),
        "books": kept_books + book_records,
        "skipped": skipped,
        "build_seconds": round(time.time() - started, 1),
    }

    checkpoint()
    Store(cfg.index_dir).save(all_chunks, vectors, manifest)
    _cache_drop(cache, used_keys)
    progress(f"Index written to {cfg.index_dir} "
             f"({len(all_chunks)} chunks, {manifest['build_seconds']}s)")
    return manifest
