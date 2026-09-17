"""Vector + lexical store, persisted as plain files.

A dedicated vector database buys approximate search that only starts paying off
above ~1M vectors. A few books is 10k-100k chunks, where an exact float32
matrix multiply on Apple Silicon returns in milliseconds and is *exactly*
correct — no recall loss from ANN approximation. That matters when the whole
brief is maximum accuracy, so the store is: one .npy + one .jsonl.

Each save writes a complete generation directory and then switches a one-line
`CURRENT` pointer to it with an atomic rename. Overwriting the four files in
place let a reader (or a crash) see vectors from one build next to chunks from
another. Indexes written before this layout existed (files directly in the
index dir) still load.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from bookrag.schemas import Chunk

VECTORS_FILE = "vectors.npy"
CHUNKS_FILE = "chunks.jsonl"
# Written by older versions only; never read. Unpickling a file runs whatever
# code it names, so a replaced index file would have been code execution. BM25 is
# rebuilt from chunks.jsonl instead: 0.09 s for 780 chunks, ~2 s for 15,600.
BM25_FILE = "bm25.pkl"
MANIFEST_FILE = "manifest.json"
POINTER_FILE = "CURRENT"
LOCK_FILE = ".lock"
_INDEX_FILES = (VECTORS_FILE, CHUNKS_FILE, BM25_FILE, MANIFEST_FILE)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class IndexBusy(RuntimeError):
    pass


@contextmanager
def index_lock(index_dir: Path):
    """Exclusive, non-blocking lock around a read-modify-write of the index.

    Two UI sessions (or a UI and the CLI) building at once would otherwise
    each read the old library and the second save would silently drop the
    first one's books.
    """
    d = Path(index_dir)
    d.mkdir(parents=True, exist_ok=True)
    with open(d / LOCK_FILE, "a+") as fh:
        try:
            import fcntl
        except ImportError:                    # Windows: no advisory lock
            yield
            return
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise IndexBusy(f"Another index build is already running on {d}. "
                            "Wait for it to finish and try again.") from exc
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


class Store:
    def __init__(self, index_dir: Path):
        self.dir = Path(index_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._data_dir = self.dir
        self.chunks: list[Chunk] = []
        self._book_col: np.ndarray = np.array([], dtype=object)
        self._book_set: set[str] = set()
        self.vectors: np.ndarray | None = None
        self._bm25 = None
        self._bm25_lock = threading.Lock()
        self._id_to_pos: dict[str, int] = {}
        self.manifest: dict = {}

    @staticmethod
    def exists(index_dir: Path) -> bool:
        d = Path(index_dir)
        return (d / POINTER_FILE).exists() or (d / CHUNKS_FILE).exists()

    def _current_dir(self) -> Path:
        ptr = self.dir / POINTER_FILE
        if ptr.exists():
            name = ptr.read_text(encoding="utf-8").strip()
            if name and (self.dir / name / CHUNKS_FILE).exists():
                return self.dir / name
        return self.dir

    # ---------------- persistence ----------------
    def save(self, chunks: list[Chunk], vectors: np.ndarray, manifest: dict) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        from rank_bm25 import BM25Okapi

        previous = self._current_dir().name if (self.dir / POINTER_FILE).exists() else None
        gen = f"gen-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
        tmp = self.dir / f".{gen}.tmp"
        tmp.mkdir()
        try:
            np.save(tmp / VECTORS_FILE, vectors.astype(np.float32))
            with open(tmp / CHUNKS_FILE, "w", encoding="utf-8") as fh:
                for c in chunks:
                    fh.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
            bm25 = BM25Okapi([tokenize(c.embed_text) for c in chunks])
            with open(tmp / MANIFEST_FILE, "w", encoding="utf-8") as fh:
                json.dump(manifest, fh, indent=2, ensure_ascii=False)
            os.replace(tmp, self.dir / gen)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise

        ptr_tmp = self.dir / f".{POINTER_FILE}.{gen}.tmp"
        ptr_tmp.write_text(gen, encoding="utf-8")
        os.replace(ptr_tmp, self.dir / POINTER_FILE)
        self._prune(keep={gen, previous})

        self.chunks, self.vectors, self.manifest = chunks, vectors, manifest
        self._bm25 = bm25
        self._data_dir = self.dir / gen
        self._reindex()

    def _prune(self, keep: set[str | None]) -> None:
        """Drop generations older than the previous one (a reader may still be
        mid-load of that) and the pre-generation flat files."""
        for p in self.dir.glob("gen-*"):
            if p.is_dir() and p.name not in keep:
                shutil.rmtree(p, ignore_errors=True)
        for name in _INDEX_FILES:
            try:
                (self.dir / name).unlink()
            except FileNotFoundError:
                pass

    def load(self) -> "Store":
        d = self._current_dir()
        cpath = d / CHUNKS_FILE
        if not cpath.exists():
            raise FileNotFoundError(
                f"No index at {self.dir}. Run `python -m bookrag.cli ingest` first."
            )
        self.chunks = [Chunk.from_dict(json.loads(l)) for l in cpath.read_text(encoding="utf-8").splitlines() if l.strip()]
        self.vectors = np.load(d / VECTORS_FILE)
        if len(self.vectors) != len(self.chunks):
            raise RuntimeError(
                f"Index at {d} is inconsistent: {len(self.chunks)} chunks but "
                f"{len(self.vectors)} vectors. Rebuild it with `python -m bookrag.cli ingest`."
            )
        self._bm25 = None                                  # built on first keyword search
        mpath = d / MANIFEST_FILE
        self.manifest = json.loads(mpath.read_text(encoding="utf-8")) if mpath.exists() else {}
        self._data_dir = d
        self._reindex()
        return self

    def _reindex(self) -> None:
        self._id_to_pos = {c.id: i for i, c in enumerate(self.chunks)}
        # Book id per row, as an array. _top() masks on every search, and
        # rebuilding that mask from a Python comprehension over every chunk
        # cost more than the similarity matmul it was filtering.
        self._book_col = np.array([c.book_id for c in self.chunks], dtype=object)
        self._book_set = set(self._book_col.tolist())

    # ---------------- search ----------------
    @property
    def bm25(self):
        if self._bm25 is None:
            from rank_bm25 import BM25Okapi
            with self._bm25_lock:
                if self._bm25 is None:
                    self._bm25 = BM25Okapi([tokenize(c.embed_text) for c in self.chunks])
        return self._bm25

    def dense_search(self, query_vec: np.ndarray, top_k: int,
                     book_ids: list[str] | None = None) -> list[tuple[int, float]]:
        if self.vectors is None or len(self.chunks) == 0:
            return []
        sims = self.vectors @ query_vec.astype(np.float32)   # vectors are L2-normalised
        return self._top(sims, top_k, book_ids)

    def bm25_search(self, query: str, top_k: int,
                    book_ids: list[str] | None = None) -> list[tuple[int, float]]:
        if len(self.chunks) == 0:
            return []
        scores = np.asarray(self.bm25.get_scores(tokenize(query)), dtype=np.float32)
        return self._top(scores, top_k, book_ids)

    def _top(self, scores: np.ndarray, top_k: int,
             book_ids: list[str] | None) -> list[tuple[int, float]]:
        # None means every book. A list always filters, even one naming books
        # that don't exist -- those must match nothing, not everything.
        if book_ids is not None and not set(book_ids) >= self._book_set:
            mask = np.isin(self._book_col, np.array(sorted(set(book_ids)), dtype=object))
            scores = np.where(mask, scores, -np.inf)
        k = min(top_k, len(scores))
        if k <= 0:
            return []
        idx = np.argpartition(-scores, k - 1)[:k] if k < len(scores) else np.arange(len(scores))
        idx = idx[np.argsort(-scores[idx])]
        return [(int(i), float(scores[i])) for i in idx if np.isfinite(scores[i])]

    # ---------------- access ----------------
    def by_id(self, chunk_id: str) -> Chunk | None:
        pos = self._id_to_pos.get(chunk_id)
        return self.chunks[pos] if pos is not None else None

    def neighbors(self, chunk: Chunk, window: int = 1) -> list[Chunk]:
        """Adjacent chunks from the same book — restores context lost at a split."""
        out = []
        for delta in range(-window, window + 1):
            if delta == 0:
                continue
            nid = f"{chunk.book_id}::{chunk.ordinal + delta:05d}"
            n = self.by_id(nid)
            if n:
                out.append(n)
        return out

    def books(self) -> list[dict]:
        return self.manifest.get("books", [])

    def chapters(self, book_id: str | None = None) -> list[str]:
        seen: list[str] = []
        for c in self.chunks:
            if book_id and c.book_id != book_id:
                continue
            if c.chapter and c.chapter not in seen:
                seen.append(c.chapter)
        return seen

    def __len__(self) -> int:
        return len(self.chunks)
