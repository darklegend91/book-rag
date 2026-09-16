"""Where the encoder weights live, and whether they can be loaded offline.

Loading a sentence-transformers model by repo id contacts Hugging Face to check
for a newer revision, even when the files are already cached: that is seconds of
network on every cold start, a hard dependency on internet at startup, and --
measured on this machine -- a re-download that emptied the cache mid-session.
When every needed file is present we load with `local_files_only`, which skips
the network entirely.
"""
from __future__ import annotations

from pathlib import Path

# Any one of these is enough for transformers to build the model.
_WEIGHT_FILES = ("model.safetensors", "model.safetensors.index.json",
                 "pytorch_model.bin", "pytorch_model.bin.index.json")


def weights_cached(repo: str) -> bool | None:
    """True if config + weights are in the local cache, False if not,
    None when it cannot be determined (no huggingface_hub, or a local path)."""
    if not repo:
        return None
    if Path(repo).expanduser().is_dir():
        return True                       # a local directory is always "cached"
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None

    def hit(filename: str) -> bool:
        return isinstance(try_to_load_from_cache(repo, filename), str)

    return hit("config.json") and any(hit(f) for f in _WEIGHT_FILES)


def offline_ok(repo: str, setting: str | bool | None = "auto") -> bool:
    """Whether to pass local_files_only=True when loading `repo`.

    "auto" (the default) means offline when the weights are already cached, so
    the first run still downloads and later runs never touch the network.
    """
    if isinstance(setting, bool):
        return setting
    text = str(setting or "auto").strip().lower()
    if text in {"true", "yes", "on", "always"}:
        return True
    if text in {"false", "no", "off", "never"}:
        return False
    return weights_cached(repo) is True


def load_local_first(cls, model_name: str, local_files_only, **kwargs):
    """Load a sentence-transformers class without touching the network when the
    weights are already cached.

    Falls back to a normal (downloading) load if the cache turns out to be
    incomplete, so a half-finished download still recovers by itself.
    """
    if offline_ok(model_name, local_files_only):
        try:
            return cls(model_name, local_files_only=True, **kwargs)
        except Exception:
            pass
    return cls(model_name, **kwargs)
