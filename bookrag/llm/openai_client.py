"""Client for an OpenAI-compatible chat server.

vLLM, llama.cpp's `server`, LM Studio, LocalAI, text-generation-webui and
Ollama's own /v1 endpoint all speak this. One client covers all of them, so
pointing this project at a different local box is a URL change, not a rewrite.

The surface deliberately mirrors OllamaClient -- same method names, same return
types -- so the rest of the pipeline never learns which one it is talking to.
Built on httpx, which is already a dependency; the `openai` package is not
required.
"""
from __future__ import annotations

import json
from typing import Any, Iterator

import httpx

from bookrag.llm.client import LLMError, LLMReplyError, ThinkStripper, parse_json, strip_thinking


class OpenAICompatClient:
    def __init__(self, base_url: str, model: str, api_key: str = "",
                 fast_model: str = "", temperature: float = 0.1,
                 num_ctx: int = 8192, timeout_s: int = 600,
                 max_tokens_default: int | None = None,
                 think: bool | None = None, json_mode: bool = True,
                 connect_timeout_s: float = 5.0):
        # think: None sends nothing (for servers that reject unknown fields);
        # True/False sets the chat-template switch vLLM and SGLang honour for
        # reasoning models such as Qwen3.
        self.think = think
        # json_mode False skips response_format and relies on the prompt plus
        # the repair loop -- for servers where constrained decoding is slow.
        self.json_mode = json_mode
        self.base_url = base_url.rstrip("/")
        self.host = self.base_url          # for parity with OllamaClient
        self.primary = model
        self.fast = fast_model or model
        self.api_key = api_key
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.timeout_s = timeout_s
        self.max_tokens_default = max_tokens_default
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self.connect_timeout_s = connect_timeout_s
        # Separate connect limit: an unreachable server fails over in seconds,
        # instead of after the full generation timeout.
        self._client = httpx.Client(timeout=httpx.Timeout(timeout_s, connect=connect_timeout_s),
                                    headers=headers)

    # ---------------- health ----------------
    def available_models(self) -> list[str]:
        try:
            r = self._client.get(f"{self.base_url}/models",
                                 timeout=httpx.Timeout(10, connect=self.connect_timeout_s))
            r.raise_for_status()
            return [m["id"] for m in r.json().get("data", [])]
        except Exception as exc:
            raise LLMError(
                f"Cannot reach the OpenAI-compatible server at {self.base_url}. ({exc})"
            ) from exc

    def health_check(self) -> dict:
        models = self.available_models()
        # Some servers return an empty model list. That means "can't confirm",
        # and must not be reported as confirmed: callers see models_listed=False
        # and can say so instead of showing a green tick.
        listed = bool(models)
        return {"host": self.base_url, "models": models, "models_listed": listed,
                "primary_ok": listed and self.primary in models,
                "fast_ok": listed and self.fast in models}

    def model_catalog(self) -> list[dict]:
        from bookrag.llm.catalog import describe_model
        try:
            names = self.available_models()
        except LLMError:
            return []
        # A remote server exposes no size or quantisation, so the catalog falls
        # back to name-pattern guidance and reports no footprint.
        return [describe_model(n) for n in names]

    # ---------------- memory (no-ops: not our process) ----------------
    def unload(self, model: str | None = None) -> None:
        """A remote server owns its own memory; nothing to release from here."""
        return None

    def loaded(self) -> list[dict]:
        return []

    # ---------------- generation ----------------
    def _payload(self, messages: list[dict], model: str | None,
                 temperature: float | None, max_tokens: int | None,
                 stream: bool, json_mode: bool) -> dict:
        payload: dict[str, Any] = {
            "model": model or self.primary,
            "messages": messages,
            "stream": stream,
            "temperature": self.temperature if temperature is None else temperature,
            "top_p": 0.9,
        }
        limit = max_tokens or self.max_tokens_default
        if limit:
            payload["max_tokens"] = limit
        if json_mode and self.json_mode:
            payload["response_format"] = {"type": "json_object"}
        if self.think is not None:
            payload["chat_template_kwargs"] = {"enable_thinking": bool(self.think)}
        return payload

    def chat(self, messages: list[dict], model: str | None = None,
             temperature: float | None = None, max_tokens: int | None = None,
             json_mode: bool = False) -> str:
        try:
            r = self._client.post(
                f"{self.base_url}/chat/completions",
                json=self._payload(messages, model, temperature, max_tokens, False, json_mode))
            r.raise_for_status()
        except httpx.HTTPStatusError as exc:
            # The body carries the useful part, e.g. "maximum context length is 8192".
            raise LLMError(f"{self.base_url} request failed: {exc.response.status_code} "
                           f"{exc.response.text[:300]}") from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"{self.base_url} request failed: {exc}") from exc
        choices = r.json().get("choices") or []
        if not choices:
            raise LLMReplyError(f"{self.base_url} returned no choices")
        return strip_thinking(choices[0].get("message", {}).get("content", "") or "")

    def stream_chat(self, messages: list[dict], model: str | None = None,
                    temperature: float | None = None) -> Iterator[str]:
        payload = self._payload(messages, model, temperature, None, True, False)
        stripper = ThinkStripper()
        started = False
        try:
            with self._client.stream("POST", f"{self.base_url}/chat/completions",
                                     json=payload) as resp:
                if resp.status_code >= 400:
                    body = resp.read().decode("utf-8", "replace")[:300]
                    error = LLMError(f"{self.base_url} stream failed: {resp.status_code} {body}")
                    error.status_code = resp.status_code
                    raise error
                for line in resp.iter_lines():
                    if not line:
                        continue
                    if line.startswith("data:"):
                        line = line[5:].strip()
                    if line == "[DONE]":
                        break
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    for ch in data.get("choices", []):
                        piece = stripper.feed((ch.get("delta") or {}).get("content") or "")
                        if not started:
                            # Drop the blank lines a stripped <think> block leaves.
                            piece = piece.lstrip()
                            started = bool(piece)
                        if piece:
                            yield piece
                tail = stripper.flush()
                if tail and (started or tail.strip()):
                    yield tail if started else tail.lstrip()
        except httpx.HTTPError as exc:
            raise LLMError(f"{self.base_url} stream failed: {exc}") from exc

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
        last = ""
        for _ in range(retries + 1):
            last = self.complete(system, user, fast=fast, temperature=temperature,
                                 json_mode=True)
            parsed = parse_json(last)
            if parsed is not None:
                return parsed
            user = (f"{user}\n\nYour previous reply was not valid JSON:\n{last[:500]}\n"
                    f"Reply with ONLY valid JSON.")
        raise LLMReplyError(f"Model did not return valid JSON after {retries + 1} attempts: {last[:300]}")
