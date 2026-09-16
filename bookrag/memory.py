"""Memory management for a 16 GB unified-memory machine.

This pipeline can want four models resident at once — an LLM for generation, a
second smaller LLM for cheap calls, a bi-encoder for embedding and a
cross-encoder for reranking. Naively that is ~15 GB and the machine swaps.

Three levers, applied here:
  * half precision for the encoders (fp16 on MPS/CUDA halves their footprint),
  * explicit unloading, so a stage releases its model before the next allocates,
  * telling Ollama to evict an LLM rather than hold it for the keep-alive window.

Nothing here changes retrieval quality: fp16 is the precision these encoders
were trained and published in, and unloading only costs reload time.
"""
from __future__ import annotations

import gc

import httpx


def free_torch_cache() -> None:
    """Return cached allocator blocks to the OS. Cheap; safe to call often."""
    gc.collect()
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def process_rss_gb() -> float:
    """Resident set size of this Python process, in GB."""
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS reports bytes; Linux reports kilobytes.
        import sys
        return rss / (1024 ** 3) if sys.platform == "darwin" else rss / (1024 ** 2)
    except Exception:
        return 0.0


def system_memory_gb() -> tuple[float, float]:
    """(total, available) system memory in GB. Best-effort, no hard dependency."""
    try:
        import subprocess
        total = int(subprocess.run(["sysctl", "-n", "hw.memsize"],
                                   capture_output=True, text=True).stdout.strip()) / (1024 ** 3)
    except Exception:
        return (0.0, 0.0)
    try:
        import subprocess
        vm = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
        page = 16384
        free = inactive = speculative = 0
        for line in vm.splitlines():
            if "page size of" in line:
                page = int(line.split("page size of")[1].split("bytes")[0].strip())
            if line.startswith("Pages free:"):
                free = int(line.split(":")[1].strip().rstrip("."))
            if line.startswith("Pages inactive:"):
                inactive = int(line.split(":")[1].strip().rstrip("."))
            if line.startswith("Pages speculative:"):
                speculative = int(line.split(":")[1].strip().rstrip("."))
        available = (free + inactive + speculative) * page / (1024 ** 3)
        return (round(total, 1), round(available, 1))
    except Exception:
        return (round(total, 1), 0.0)


def ollama_loaded(host: str = "http://localhost:11434") -> list[dict]:
    """Models Ollama currently holds in memory, with their sizes."""
    try:
        r = httpx.get(f"{host}/api/ps", timeout=5)
        r.raise_for_status()
        return [{"name": m.get("name", ""),
                 "size_gb": round(m.get("size", 0) / (1024 ** 3), 2)}
                for m in r.json().get("models", [])]
    except Exception:
        return []


def ollama_unload(model: str, host: str = "http://localhost:11434",
                  wait_s: float = 15.0) -> bool:
    """Evict a model from Ollama and wait until the memory is actually back.

    `keep_alive: 0` on an empty request tells Ollama to drop the model now
    instead of holding it for the default five minutes — which matters when the
    next stage of the pipeline needs those gigabytes for an encoder.

    The eviction is asynchronous: the POST returns well before the process has
    released the pages. Callers that read free memory straight afterwards see
    the *old* number and conclude nothing happened, so this polls /api/ps until
    the model is gone. Measured on a 16 GB box: ~1s for an 8B, and free memory
    goes 1.8 GB -> 8.3 GB once it lands.

    Returns True if the model is no longer resident, False on timeout or error.
    """
    import time
    try:
        httpx.post(f"{host}/api/generate",
                   json={"model": model, "keep_alive": 0, "prompt": ""},
                   timeout=httpx.Timeout(30, connect=5))
    except Exception:
        return False
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        if not any(m["name"] == model for m in ollama_loaded(host)):
            return True
        time.sleep(0.25)
    # One last look: the deadline may have expired mid-eviction.
    return not any(m["name"] == model for m in ollama_loaded(host))


def ollama_unload_all(host: str = "http://localhost:11434",
                      wait_s: float = 15.0) -> list[str]:
    freed = []
    for m in ollama_loaded(host):
        if ollama_unload(m["name"], host, wait_s=wait_s):
            freed.append(m["name"])
    return freed


# --------------------------------------------------------------------------
# Stage arbitration
# --------------------------------------------------------------------------
# Approximate resident cost of each component, in GB, at the settings this
# project ships (fp16 encoders, Q4_K_M 8B). Used only to decide what to release
# first -- the actual decision is driven by measured free memory, not these.
_COST_GB = {"embedder": 1.15, "reranker": 1.06, "llm": 6.2}

PROFILES = {
    # Keep everything resident and only intervene when the machine is actually
    # under pressure. Encoders reload from the OS page cache in ~1-2s, so
    # dropping them is cheap; evicting the LLM costs ~10s and is a last resort.
    "balanced": {
        "headroom_mult": 1.0,
        "handoff": False,        # do not release encoders before every LLM call
        "evict_llm": False,      # never evict the LLM to make room for encoders
    },
    # Strict hand-off: each stage owns the machine alone. Peak footprint is one
    # model instead of three, at the cost of a reload on every stage change.
    # For 8 GB machines, or when something else large is running.
    "conservative": {
        "headroom_mult": 2.0,
        "handoff": True,
        "evict_llm": True,
    },
}


class Arbiter:
    """Decides what to unload before a pipeline stage allocates.

    Registered components are held weakly: the arbiter must never be the reason
    a model stays alive. Every decision is checked against measured free memory
    rather than the estimates in _COST_GB, so it degrades gracefully on machines
    whose numbers differ from the ones this was tuned on.
    """

    def __init__(self, profile: str = "balanced", min_free_gb: float = 2.0,
                 host: str = "http://localhost:11434", log=None):
        import weakref
        self.settings = dict(PROFILES.get(profile, PROFILES["balanced"]))
        # The profile scales the user's single min_free_gb knob rather than
        # carrying its own absolute number, so there is one place to tune and
        # the profiles keep their relative aggressiveness whatever it is set to.
        self.settings["headroom_gb"] = float(min_free_gb) * self.settings["headroom_mult"]
        self.profile = profile if profile in PROFILES else "balanced"
        self.host = host
        self.log = log or (lambda msg: None)
        self._refs: dict[str, "weakref.ref"] = {}
        self._weakref = weakref

    def register(self, **components) -> None:
        for name, obj in components.items():
            if obj is not None:
                try:
                    self._refs[name] = self._weakref.ref(obj)
                except TypeError:
                    pass

    def _get(self, name: str):
        ref = self._refs.get(name)
        return ref() if ref else None

    def _release(self, name: str) -> float:
        obj = self._get(name)
        if obj is None or getattr(obj, "_model", "missing") is None:
            return 0.0
        try:
            obj.unload()
        except Exception:
            return 0.0
        self.log(f"[memory] released {name} (~{_COST_GB.get(name, 0):.1f} GB)")
        return _COST_GB.get(name, 0.0)

    def available_gb(self) -> float:
        return system_memory_gb()[1]

    def before(self, stage: str) -> None:
        """Prepare for `stage` in {"embed", "rerank", "generate", "ingest"}."""
        s = self.settings
        if stage == "generate":
            if s["handoff"]:
                self._release("reranker")
                self._release("embedder")
                return
            if self.available_gb() < s["headroom_gb"]:
                free_torch_cache()
                if self.available_gb() < s["headroom_gb"]:
                    self._release("reranker")
                if self.available_gb() < s["headroom_gb"]:
                    self._release("embedder")
        elif stage == "ingest":
            # Ingest embeds every chunk in the book; it wants the whole machine.
            freed = ollama_unload_all(self.host)
            if freed:
                self.log(f"[memory] evicted from Ollama for ingest: {', '.join(freed)}")
        elif stage in {"embed", "rerank"}:
            if s["evict_llm"] and self.available_gb() < s["headroom_gb"]:
                freed = ollama_unload_all(self.host)
                if freed:
                    self.log(f"[memory] evicted from Ollama: {', '.join(freed)}")

    def snapshot(self) -> dict:
        total, available = system_memory_gb()
        return {
            "profile": self.profile,
            "total_gb": total,
            "available_gb": available,
            "headroom_gb": self.settings["headroom_gb"],
            "process_rss_gb": round(process_rss_gb(), 2),
            "encoders_loaded": [n for n in ("embedder", "reranker")
                                if getattr(self._get(n), "_model", None) is not None],
            "ollama_loaded": ollama_loaded(self.host),
        }


def arbiter_from_config(cfg, log=None) -> Arbiter:
    return Arbiter(
        profile=str(cfg.get("memory.profile", "balanced")),
        min_free_gb=float(cfg.get("memory.min_free_gb", 2.0)),
        host=str(cfg.get("llm.host", "http://localhost:11434")),
        log=log,
    )
