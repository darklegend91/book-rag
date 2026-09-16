"""Typed access to config.yaml.

Everything downstream imports `load_config()` and reads dotted paths, so a
single YAML file stays the only place tuning knobs live.
"""
from __future__ import annotations

import copy
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"
DEFAULT_ENV = PROJECT_ROOT / ".env"

# ${VAR} or ${VAR:-fallback}
_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def load_dotenv(path: Path | None = None) -> dict[str, str]:
    """Read a .env file into os.environ without overwriting the real environment.

    A real shell variable always wins, so `BOOKRAG_LLM_BASE_URL=... streamlit run`
    overrides the file for one run. Deliberately dependency-free: KEY=VALUE,
    `export` prefixes and # comments, with optional surrounding quotes.
    """
    path = path or Path(os.environ.get("BOOKRAG_ENV", DEFAULT_ENV))
    loaded: dict[str, str] = {}
    if not path.exists():
        return loaded
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        loaded[key] = value
        os.environ.setdefault(key, value)
    return loaded


def _expand(node: Any) -> Any:
    """Substitute ${VAR} / ${VAR:-default} through the loaded YAML tree."""
    if isinstance(node, dict):
        return {k: _expand(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_expand(v) for v in node]
    if not isinstance(node, str):
        return node

    def sub(m: re.Match) -> str:
        return os.environ.get(m.group(1), m.group(2) if m.group(2) is not None else "")

    out = _ENV_REF.sub(sub, node)
    if out == node:
        return node
    # An interpolated scalar should still come back typed: a port stays an int,
    # a flag stays a bool, an unset variable with no default becomes None rather
    # than the empty string.
    if out == "":
        return None
    try:
        parsed = yaml.safe_load(out)
        return parsed if parsed is not None else out
    except yaml.YAMLError:
        return out


@dataclass
class Config:
    raw: dict[str, Any] = field(default_factory=dict)
    root: Path = PROJECT_ROOT

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.raw
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def path(self, dotted: str) -> Path:
        """Resolve a configured path relative to the project root."""
        value = self.get(dotted)
        if value is None:
            raise KeyError(f"No path configured at '{dotted}'")
        p = Path(value)
        return p if p.is_absolute() else self.root / p

    # Convenience accessors used all over the codebase.
    @property
    def index_dir(self) -> Path:
        return self.path("paths.index_dir")

    @property
    def books_dir(self) -> Path:
        return self.path("paths.books_dir")

    @property
    def export_dir(self) -> Path:
        return self.path("paths.export_dir")

    def with_overrides(self, overrides: dict[str, Any]) -> "Config":
        """A copy with dotted keys replaced, e.g. {"retrieval.max_context_tokens": 12000}.

        Lets an evaluation run compare settings without editing config.yaml;
        the original Config (which load_config caches) is never mutated.
        """
        raw = copy.deepcopy(self.raw)
        for dotted, value in overrides.items():
            parts = dotted.split(".")
            node = raw
            for part in parts[:-1]:
                if not isinstance(node.get(part), dict):
                    node[part] = {}
                node = node[part]
            node[parts[-1]] = value
        return Config(raw=raw, root=self.root)


_OVERRIDE_KEY = re.compile(r"[A-Za-z0-9_]+(\.[A-Za-z0-9_]+)*")


def parse_overrides(pairs: list[str] | None) -> dict[str, Any]:
    """["retrieval.hyde=true", "rerank.top_k=6"] -> {"retrieval.hyde": True, "rerank.top_k": 6}.

    Values are parsed as YAML, exactly like config.yaml, so numbers and
    booleans keep their types.
    """
    out: dict[str, Any] = {}
    for pair in pairs or []:
        key, sep, value = pair.partition("=")
        key = key.strip()
        if not sep or not _OVERRIDE_KEY.fullmatch(key):
            raise ValueError(f"Expected key=value with a dotted key (e.g. rerank.top_k=6), got {pair!r}")
        try:
            out[key] = yaml.safe_load(value) if value.strip() else ""
        except yaml.YAMLError:
            out[key] = value
    return out


_cache: dict[str, Config] = {}


def load_config(path: str | Path | None = None) -> Config:
    cfg_path = Path(path) if path else Path(os.environ.get("BOOKRAG_CONFIG", DEFAULT_CONFIG))
    key = str(cfg_path.resolve())
    if key not in _cache:
        load_dotenv()
        with open(cfg_path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        _cache[key] = Config(raw=_expand(raw), root=cfg_path.resolve().parent)
    return _cache[key]
