"""BGE-M3 dense embeddings via sentence-transformers.

Loaded lazily and cached process-wide — on 16 GB unified memory you do not want
two copies of a 2 GB model resident because two modules both imported it.
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

_MODEL_CACHE: dict[str, object] = {}

# Free GPU memory one encoder needs before auto puts it on CUDA: ~1.2 GB of fp16
# weights, the CUDA context and activation buffers, with margin. Checked per
# model as it loads, so the reranker sees what is left after the embedder.
DEFAULT_MIN_FREE_GPU_GB = 3.0


def resolve_device(pref: str = "auto", min_free_gpu_gb: float = DEFAULT_MIN_FREE_GPU_GB) -> str:
    """The device the encoders run on.

    `auto` picks the GPU only if it has room. A co-located LLM server claims
    most of the card up front (vLLM reserves ~90% by default); on a 48 GB GPU
    that left 3.9 MB free, and "auto -> cuda" failed every query with CUDA out
    of memory. An explicit "cuda", "mps" or "cpu" is always honoured.
    """
    if pref and pref != "auto":
        return pref
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"          # Apple Silicon GPU (unified memory)
        if torch.cuda.is_available():
            free, _total = torch.cuda.mem_get_info()
            if free / 1e9 >= min_free_gpu_gb:
                return "cuda"
            log.warning("GPU has %.1f GB free, below the %.1f GB the encoders need; running "
                        "them on CPU. A co-located LLM server is the usual cause (vLLM "
                        "reserves ~90%% of the GPU unless --gpu-memory-utilization is lower).",
                        free / 1e9, min_free_gpu_gb)
    except Exception:
        pass
    return "cpu"


def is_gpu_oom(exc: BaseException) -> bool:
    """Whether an exception is the GPU running out of memory."""
    try:
        import torch
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except Exception:
        pass
    text = str(exc).lower()
    return "out of memory" in text and ("cuda" in text or "mps" in text)


class Embedder:
    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = "auto",
                 batch_size: int = 8, normalize: bool = True,
                 query_prefix: str = "", passage_prefix: str = "",
                 fp16: bool = True, local_files_only: str | bool = "auto",
                 min_free_gpu_gb: float = DEFAULT_MIN_FREE_GPU_GB):
        self.local_files_only = local_files_only
        self.model_name = model_name
        self.device = resolve_device(device, min_free_gpu_gb)
        self.batch_size = batch_size
        self.normalize = normalize
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.fp16 = fp16
        self._model = None

    @property
    def model(self):
        if self._model is None:
            key = f"{self.model_name}:{self.device}:fp16={self.fp16}"
            if key not in _MODEL_CACHE:
                from sentence_transformers import SentenceTransformer
                # fp16 halves the ~2.3 GB footprint at no measurable retrieval
                # cost. Set the dtype at load time via model_kwargs rather than
                # calling .half() afterwards -- reassigning the inner module
                # breaks sentence-transformers 6.x.
                kwargs = {}
                if self.fp16 and self.device in {"mps", "cuda"}:
                    import torch
                    kwargs["model_kwargs"] = {"torch_dtype": torch.float16}
                from bookrag.weights import load_local_first
                model = load_local_first(
                    SentenceTransformer, self.model_name, self.local_files_only,
                    device=self.device, **kwargs)
                _MODEL_CACHE[key] = model
            self._model = _MODEL_CACHE[key]
        return self._model

    def unload(self) -> None:
        """Release the model so the next pipeline stage can have the memory."""
        from bookrag.memory import free_torch_cache
        key = f"{self.model_name}:{self.device}:fp16={self.fp16}"
        _MODEL_CACHE.pop(key, None)
        self._model = None
        free_torch_cache()

    def _encode(self, texts: list[str], prefix: str, show_progress: bool) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1024), dtype=np.float32)
        payload = [prefix + t for t in texts] if prefix else texts
        kwargs = dict(batch_size=self.batch_size, show_progress_bar=show_progress,
                      convert_to_numpy=True, normalize_embeddings=self.normalize)
        try:
            vecs = self.model.encode(payload, **kwargs)                  # type:ignore
        except Exception as exc:
            # Another process can take the GPU after we started. Finish the
            # request on CPU rather than failing it, and stay there.
            if self.device == "cpu" or not is_gpu_oom(exc):
                raise
            log.warning("GPU ran out of memory while embedding; moving the embedder to CPU")
            self.unload()
            self.device = "cpu"
            vecs = self.model.encode(payload, **kwargs)                  # type:ignore
        return np.asarray(vecs, dtype=np.float32)

    def embed_passages(self, texts: list[str], show_progress: bool = True) -> np.ndarray:
        return self._encode(texts, self.passage_prefix, show_progress)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, self.query_prefix, False)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_queries([text])[0]


def embedder_from_config(cfg) -> Embedder:
    return Embedder(
        model_name=cfg.get("embedding.model", "BAAI/bge-m3"),
        device=cfg.get("embedding.device", "auto"),
        batch_size=int(cfg.get("embedding.batch_size", 8)),
        normalize=bool(cfg.get("embedding.normalize", True)),
        query_prefix=cfg.get("embedding.query_prefix", "") or "",
        passage_prefix=cfg.get("embedding.passage_prefix", "") or "",
        fp16=bool(cfg.get("memory.fp16_encoders", True)),
        local_files_only=cfg.get("embedding.local_files_only", "auto"),
        min_free_gpu_gb=float(cfg.get("embedding.min_free_gpu_gb", DEFAULT_MIN_FREE_GPU_GB)),
    )
