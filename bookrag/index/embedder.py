"""BGE-M3 dense embeddings via sentence-transformers.

Loaded lazily and cached process-wide — on 16 GB unified memory you do not want
two copies of a 2 GB model resident because two modules both imported it.
"""
from __future__ import annotations

import numpy as np

_MODEL_CACHE: dict[str, object] = {}


def resolve_device(pref: str = "auto") -> str:
    if pref and pref != "auto":
        return pref
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"          # Apple Silicon GPU
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


class Embedder:
    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = "auto",
                 batch_size: int = 8, normalize: bool = True,
                 query_prefix: str = "", passage_prefix: str = "",
                 fp16: bool = True, local_files_only: str | bool = "auto"):
        self.local_files_only = local_files_only
        self.model_name = model_name
        self.device = resolve_device(device)
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
        vecs = self.model.encode(                                                                                            # type:ignore
            payload,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
        )
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
    )
