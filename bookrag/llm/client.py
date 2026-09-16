"""Thin Ollama client.

Two roles are exposed: `primary` (qwen3:8b — grounded answering, question
writing, verification) and `fast` (qwen3:4b-instruct — query rewriting, HyDE,
PYQ parsing). Splitting them keeps the expensive model for the calls where
reasoning quality actually changes the output.
"""
from __future__ import annotations

import json
import re
from typing import Any, Iterator

import httpx


class LLMError(RuntimeError):
    status_code: int | None = None       # HTTP status, when the server answered with one


class LLMReplyError(LLMError):
    """The backend answered, but the reply was unusable (invalid JSON, no choices).

    Retrying the same request on a different backend is not what a fallback is
    for, and when that backend is down it replaces this error with an
    unrelated "connection refused".
    """


def _http_status(exc: BaseException | None, depth: int = 0) -> int | None:
    if exc is None or depth > 5:
        return None
    if getattr(exc, "status_code", None):
        return int(exc.status_code)
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code
    return _http_status(exc.__cause__, depth + 1)


def should_fail_over(exc: BaseException) -> bool:
    """Whether an error means "this backend can't serve requests right now".

    Connection failures, timeouts, 5xx and 429 do. A reply the model got wrong,
    or a 4xx the request itself caused (e.g. prompt too long), does not: every
    backend would reject it, and failing over only hides the real message.
    """
    if isinstance(exc, LLMReplyError):
        return False
    status = _http_status(exc)
    if status is not None and 400 <= status < 500 and status not in (408, 429):
        return False
    return True


class OllamaClient:
    def __init__(self, host: str = "http://localhost:11434",
                 primary: str = "qwen3:8b", fast: str = "qwen3:4b-instruct",
                 temperature: float = 0.1, num_ctx: int = 16384,
                 timeout_s: int = 600, think: bool = False,
                 keep_alive: str = "5m", connect_timeout_s: float = 5.0):
        self.host = host.rstrip("/")
        self.primary = primary
        self.fast = fast
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.timeout_s = timeout_s
        self.think = think
        self.keep_alive = keep_alive
        self.connect_timeout_s = connect_timeout_s
        # Connecting and generating get separate limits: a server that is down
        # must fail (and fail over) in seconds, while one that is up may take
        # minutes to generate. One 600 s timeout covered both.
        self._client = httpx.Client(timeout=httpx.Timeout(timeout_s, connect=connect_timeout_s))

    # ---------------- health ----------------
    def available_models(self) -> list[str]:
        try:
            r = self._client.get(f"{self.host}/api/tags",
                                 timeout=httpx.Timeout(10, connect=self.connect_timeout_s))
            r.raise_for_status()
            return [m["name"] for m in r.json().get("models", [])]
        except Exception as exc:
            raise LLMError(
                f"Cannot reach Ollama at {self.host}. Start it with `ollama serve`. ({exc})"
            ) from exc

    def model_catalog(self) -> list[dict]:
        """Installed models, each annotated with what it is good for here."""
        from bookrag.llm.catalog import describe_model
        try:
            r = self._client.get(f"{self.host}/api/tags",
                                 timeout=httpx.Timeout(10, connect=self.connect_timeout_s))
            r.raise_for_status()
            raw = r.json().get("models", [])
        except Exception:
            return []
        out = [describe_model(m.get("name", ""), m.get("size", 0), m.get("details", {}))
               for m in raw]
        # Chat-capable first, then heaviest first -- accuracy above speed, since
        # this project's whole brief is faithfulness.
        out.sort(key=lambda d: (not d["chat_capable"], -d["size_gb"]))
        return out

    def health_check(self) -> dict:
        models = self.available_models()
        def present(tag: str) -> bool:
            # Exact tag only: qwen3:4b being installed says nothing about
            # whether qwen3:8b is. An untagged name means ":latest" to Ollama.
            return tag in models or (":" not in tag and f"{tag}:latest" in models)
        return {"host": self.host, "models": models,
                "primary_ok": present(self.primary), "fast_ok": present(self.fast)}

    # ---------------- memory ----------------
    def unload(self, model: str | None = None) -> None:
        """Evict a model from Ollama now, freeing its VRAM/unified memory."""
        from bookrag.memory import ollama_unload
        for m in ([model] if model else {self.primary, self.fast}):
            ollama_unload(m, self.host)

    def loaded(self) -> list[dict]:
        from bookrag.memory import ollama_loaded
        return ollama_loaded(self.host)

    # ---------------- generation ----------------
    def _options(self, temperature: float | None, max_tokens: int | None) -> dict:
        opts: dict[str, Any] = {
            "temperature": self.temperature if temperature is None else temperature,
            "num_ctx": self.num_ctx,
            "top_p": 0.9,
            "repeat_penalty": 1.05,
        }
        if max_tokens:
            opts["num_predict"] = max_tokens
        return opts

    def chat(self, messages: list[dict], model: str | None = None,
             temperature: float | None = None, max_tokens: int | None = None,
             json_mode: bool = False) -> str:
        payload: dict[str, Any] = {
            "model": model or self.primary,
            "messages": messages,
            "stream": False,
            "think": self.think,
            "keep_alive": self.keep_alive,
            "options": self._options(temperature, max_tokens),
        }
        if json_mode:
            payload["format"] = "json"
        try:
            r = self._client.post(f"{self.host}/api/chat", json=payload)
            r.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(f"Ollama request failed: {exc}") from exc
        return strip_thinking(r.json().get("message", {}).get("content", ""))

    def stream_chat(self, messages: list[dict], model: str | None = None,
                    temperature: float | None = None) -> Iterator[str]:
        payload = {
            "model": model or self.primary,
            "messages": messages,
            "stream": True,
            "think": self.think,
            "keep_alive": self.keep_alive,
            "options": self._options(temperature, None),
        }
        with self._client.stream("POST", f"{self.host}/api/chat", json=payload) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                piece = data.get("message", {}).get("content", "")
                if piece:
                    yield piece
                if data.get("done"):
                    break

    def complete(self, system: str, user: str, fast: bool = False,
                 temperature: float | None = None, max_tokens: int | None = None,
                 json_mode: bool = False) -> str:
        return self.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            model=self.fast if fast else self.primary,
            temperature=temperature, max_tokens=max_tokens, json_mode=json_mode,
        )

    def complete_json(self, system: str, user: str, fast: bool = False,
                      temperature: float | None = None, retries: int = 2) -> Any:
        """JSON-constrained call with a repair loop.

        Local models occasionally emit trailing prose or a fenced block even in
        JSON mode; `parse_json` salvages those instead of failing the run.
        """
        last = ""
        for attempt in range(retries + 1):
            last = self.complete(system, user, fast=fast, temperature=temperature, json_mode=True)
            parsed = parse_json(last)
            if parsed is not None:
                return parsed
            user = (f"{user}\n\nYour previous reply was not valid JSON:\n{last[:500]}\n"
                    f"Reply with ONLY valid JSON.")
        raise LLMReplyError(f"Model did not return valid JSON after {retries + 1} attempts: {last[:300]}")


def strip_thinking(text: str) -> str:
    """Remove qwen3 <think>...</think> blocks if the model emits them anyway."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


class ThinkStripper:
    """strip_thinking for a token stream.

    A server that ignores the thinking switch streams the reasoning inline, and
    the tags themselves can be split across pieces ("<thi", "nk>"). Text that
    might be the start of a tag is held back until the next piece decides it.
    """
    OPEN, CLOSE = "<think>", "</think>"

    def __init__(self):
        self._buf = ""
        self._inside = False

    def feed(self, piece: str) -> str:
        self._buf += piece
        out: list[str] = []
        while True:
            if self._inside:
                i = self._buf.find(self.CLOSE)
                if i == -1:
                    self._buf = self._buf[-(len(self.CLOSE) - 1):]
                    break
                self._buf = self._buf[i + len(self.CLOSE):]
                self._inside = False
            else:
                i = self._buf.find(self.OPEN)
                if i == -1:
                    keep = next((k for k in range(len(self.OPEN) - 1, 0, -1)
                                 if self._buf.endswith(self.OPEN[:k])), 0)
                    out.append(self._buf[:len(self._buf) - keep])
                    self._buf = self._buf[len(self._buf) - keep:]
                    break
                out.append(self._buf[:i])
                self._buf = self._buf[i + len(self.OPEN):]
                self._inside = True
        return "".join(out)

    def flush(self) -> str:
        rest = "" if self._inside else self._buf
        self._buf = ""
        return rest


# A real JSON escape ("keep"), or a backslash that isn't one ("stray"). \b \f \r
# \t count as real only when no letter follows: "\frac", "\text", "\beta" and
# "\rho" are LaTeX commands that merely start with an escape letter, and a model
# never means backspace or form feed before a word. \n always stays a newline,
# because "one.\nTwo" is ordinary text (so a LaTeX "\nu" is the one casualty).
_JSON_ESCAPE = re.compile(
    r'\\(?:(?P<keep>\\|"|/|u[0-9a-fA-F]{4}|n|[bfrt](?![A-Za-z]))|(?P<stray>.))', re.DOTALL)


def _repair_escapes(text: str) -> str:
    """Make stray backslashes literal, keeping genuine JSON escapes.

    Models writing about maths put raw LaTeX inside JSON strings with single
    backslashes ("$\\eta = 1 - \\frac{T_c}{T_h}$"), which json.loads rejects.
    Measured: every claim check on an answer containing a formula failed this
    way, so any formula answer was refused.
    """
    def fix(m: "re.Match") -> str:
        return m.group(0) if m.group("keep") is not None else "\\\\" + m.group("stray")
    return _JSON_ESCAPE.sub(fix, text)


def _loads(text: str) -> Any | None:
    # The repaired text goes first whenever it differs: "$\frac{1}{2}$" is valid
    # JSON as written, but decodes to a form feed followed by "rac". Correctly
    # escaped JSON has no stray backslashes, so it comes back unchanged.
    repaired = _repair_escapes(text)
    for candidate in ((repaired, text) if repaired != text else (text,)):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def parse_json(text: str) -> Any | None:
    text = strip_thinking(text).strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    parsed = _loads(text)
    if parsed is not None:
        return parsed
    # Salvage the outermost JSON object or array.
    for open_c, close_c in (("{", "}"), ("[", "]")):
        start, end = text.find(open_c), text.rfind(close_c)
        if start != -1 and end > start:
            parsed = _loads(text[start:end + 1])
            if parsed is not None:
                return parsed
    return None


def _ollama_from_config(cfg) -> OllamaClient:
    return OllamaClient(
        host=cfg.get("llm.host", "http://localhost:11434"),
        primary=cfg.get("llm.primary", "qwen3:8b"),
        fast=cfg.get("llm.fast", "qwen3:4b-instruct"),
        temperature=float(cfg.get("llm.temperature", 0.1)),
        num_ctx=int(cfg.get("llm.num_ctx", 16384)),
        timeout_s=int(cfg.get("llm.timeout_s", 600)),
        connect_timeout_s=float(cfg.get("llm.connect_timeout_s", 5)),
        think=bool(cfg.get("llm.think", False)),
        keep_alive=str(cfg.get("memory.ollama_keep_alive", "5m")),
    )


def _openai_from_config(cfg):
    from bookrag.llm.openai_client import OpenAICompatClient
    base_url = cfg.get("llm.openai.base_url")
    model = cfg.get("llm.openai.model")
    if not base_url or not model:
        raise LLMError(
            "llm.provider is 'openai' but llm.openai.base_url / llm.openai.model "
            "are not set. Fill them in, or set them in .env as "
            "BOOKRAG_LLM_BASE_URL and BOOKRAG_LLM_MODEL."
        )
    think = cfg.get("llm.openai.enable_thinking")
    return OpenAICompatClient(
        base_url=str(base_url),
        model=str(model),
        api_key=str(cfg.get("llm.openai.api_key") or ""),
        fast_model=str(cfg.get("llm.openai.fast_model") or ""),
        temperature=float(cfg.get("llm.temperature", 0.1)),
        num_ctx=int(cfg.get("llm.num_ctx", 8192)),
        timeout_s=int(cfg.get("llm.timeout_s", 600)),
        connect_timeout_s=float(cfg.get("llm.connect_timeout_s", 5)),
        think=None if think is None else bool(think),
        json_mode=bool(cfg.get("llm.openai.json_mode", True)),
    )


def client_from_config(cfg, log=None):
    """Build the client the config asks for, wrapped in a fallback if enabled.

    Returns an OllamaClient, an OpenAICompatClient, or a FallbackClient over
    both. Every caller sees the same surface either way.
    """
    provider = str(cfg.get("llm.provider", "ollama")).lower()
    builders = {"ollama": _ollama_from_config, "openai": _openai_from_config}
    if provider not in builders:
        raise LLMError(f"Unknown llm.provider '{provider}'. Use 'ollama' or 'openai'.")

    primary = builders[provider](cfg)
    if not bool(cfg.get("llm.fallback.enabled", True)):
        return primary
    fb_provider = str(cfg.get("llm.fallback.provider", "ollama")).lower()
    if fb_provider == provider or fb_provider not in builders:
        return primary
    try:
        fallback = builders[fb_provider](cfg)
    except LLMError:
        # A misconfigured fallback must never take the primary down with it.
        return primary

    from bookrag.llm.router import FallbackClient
    return FallbackClient(primary, fallback, log=log,
                          label_primary=provider, label_fallback=fb_provider)
