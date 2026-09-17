"""Cross-encoder reranking with BGE-reranker-v2-m3.

Bi-encoder retrieval scores a query and a passage independently, so it captures
topical similarity but not whether the passage actually *answers* the query. The
cross-encoder reads both together and is far more discriminative. It is also the
component that makes a hard refusal threshold meaningful: its scores are logits
on an "is this relevant" head, so a low top score is genuine evidence that the
books do not cover the question.
"""
from __future__ import annotations

import logging

from bookrag.index.embedder import DEFAULT_MIN_FREE_GPU_GB

log = logging.getLogger(__name__)

_RERANKER_CACHE: dict[str, object] = {}


class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3",
                 device: str = "auto", batch_size: int = 8, fp16: bool = True,
                 release_cache: bool = True, max_length: int = 512,
                 local_files_only: str | bool = "auto", min_free_gpu_gb: float = DEFAULT_MIN_FREE_GPU_GB):
        self.local_files_only = local_files_only
        from bookrag.index.embedder import resolve_device
        self.model_name = model_name
        self.device = resolve_device(device, min_free_gpu_gb)
        self.batch_size = batch_size
        self.fp16 = fp16
        self.release_cache = release_cache
        # Chunks are cut at ingest.chunk_tokens (700 by default) and the query
        # is short, so 1024 only ever paid off on the tail of oversized chunks
        # while costing every pair 2x the attention. Keep this >= chunk_tokens
        # only if you raise chunk sizes.
        self.max_length = max_length
        self._model = None

    @property
    def model(self):
        if self._model is None:
            key = f"{self.model_name}:{self.device}:{self.max_length}:fp16={self.fp16}"
            if key not in _RERANKER_CACHE:
                from sentence_transformers import CrossEncoder
                # See Embedder: dtype must be set at load time, not via .half().
                kwargs = {}
                if self.fp16 and self.device in {"mps", "cuda"}:
                    import torch
                    kwargs["model_kwargs"] = {"torch_dtype": torch.float16}
                from bookrag.weights import load_local_first
                model = load_local_first(CrossEncoder, self.model_name, self.local_files_only,
                                         device=self.device, max_length=self.max_length, **kwargs)
                _RERANKER_CACHE[key] = model
            self._model = _RERANKER_CACHE[key]
        return self._model

    def unload(self) -> None:
        from bookrag.memory import free_torch_cache
        key = f"{self.model_name}:{self.device}:{self.max_length}:fp16={self.fp16}"
        _RERANKER_CACHE.pop(key, None)
        self._model = None
        free_torch_cache()

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        pairs = [(query, p) for p in passages]
        try:
            scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        except Exception as exc:
            from bookrag.index.embedder import is_gpu_oom
            if self.device == "cpu" or not is_gpu_oom(exc):
                raise
            log.warning("GPU ran out of memory while reranking; moving the reranker to CPU")
            self.unload()
            self.device = "cpu"
            scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        if self.release_cache:
            # Scoring ~40 passages at 1024 tokens leaves roughly 1 GB of
            # activation buffers in the allocator's cache. The live weights are
            # only ~1.06 GB, so returning the cache almost halves this stage's
            # footprint. The buffers are rebuilt next query at negligible cost.
            from bookrag.memory import free_torch_cache
            free_torch_cache()
        return [float(s) for s in scores]


def reranker_from_config(cfg) -> Reranker | None:
    if not cfg.get("rerank.enabled", True):
        return None
    return Reranker(
        model_name=cfg.get("rerank.model", "BAAI/bge-reranker-v2-m3"),
        device=cfg.get("embedding.device", "auto"),
        batch_size=int(cfg.get("rerank.batch_size", 8)),
        fp16=bool(cfg.get("memory.fp16_encoders", True)),
        release_cache=bool(cfg.get("memory.empty_cache_after_rerank", False)),
        local_files_only=cfg.get("rerank.local_files_only", "auto"),
        max_length=int(cfg.get("rerank.max_length", 512)),
        min_free_gpu_gb=float(cfg.get("embedding.min_free_gpu_gb", DEFAULT_MIN_FREE_GPU_GB)),
    )
