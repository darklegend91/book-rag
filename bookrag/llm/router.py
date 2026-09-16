"""Primary LLM with an automatic fallback.

The point is that a local server you run yourself is allowed to be down. When
it is, questions should keep working against Ollama rather than erroring, and
you should be able to see that it happened -- a silent downgrade to a different
model would quietly change what the grounding thresholds are being applied to.

Failover is per-call and stateless apart from `last_used` and `failures`, so a
primary that comes back up is picked up on the next question with no restart.
Streaming fails over only before the first token: once bytes have reached the
caller, switching models mid-answer would splice two different completions
together.
"""
from __future__ import annotations

from typing import Any, Iterator

from bookrag.llm.client import LLMError, should_fail_over


class FallbackClient:
    def __init__(self, primary_client, fallback_client, log=None,
                 label_primary: str = "primary", label_fallback: str = "fallback"):
        self._primary = primary_client
        self._fallback = fallback_client
        self.log = log or (lambda msg: None)
        self.label_primary = label_primary
        self.label_fallback = label_fallback
        self.last_used = label_primary
        self.failures = 0

    # The pipeline reads these off the client; keep them pointed at whichever
    # backend actually answered last, so the UI and `use_model` stay coherent.
    @property
    def primary(self) -> str:
        return self._active().primary

    @primary.setter
    def primary(self, name: str) -> None:
        # The model list a user picks from is the answering backend's, so the
        # choice belongs to that backend. Always writing it to the primary left
        # a failed-over session answering with -- and reporting -- the old model.
        self._active().primary = name

    @property
    def fast(self) -> str:
        return self._active().fast

    @property
    def host(self) -> str:
        return self._active().host

    def _active(self):
        return self._fallback if self.last_used == self.label_fallback else self._primary

    def _both_failed(self, primary_exc: BaseException, fallback_exc: BaseException) -> LLMError:
        # Report both: the fallback's "connection refused" alone hides why the
        # primary -- the backend actually meant to answer -- failed.
        return LLMError(f"{self.label_primary} failed ({primary_exc}), and the fallback "
                        f"({self.label_fallback}) also failed ({fallback_exc})")

    def _run(self, method: str, *args, **kwargs):
        try:
            result = getattr(self._primary, method)(*args, **kwargs)
            self.last_used = self.label_primary
            return result
        except Exception as exc:
            # Fail over only when the primary can't serve requests. A bad reply
            # or a request the server rejects would fail on any backend.
            if self._fallback is None or not should_fail_over(exc):
                raise
            self.failures += 1
            self.log(f"[llm] {self.label_primary} failed ({exc}); "
                     f"falling back to {self.label_fallback}")
            try:
                result = getattr(self._fallback, method)(*args, **kwargs)
            except Exception as fallback_exc:
                raise self._both_failed(exc, fallback_exc) from fallback_exc
            self.last_used = self.label_fallback
            return result

    # ---------------- generation ----------------
    def chat(self, *a, **k) -> str:
        return self._run("chat", *a, **k)

    def complete(self, *a, **k) -> str:
        return self._run("complete", *a, **k)

    def complete_json(self, *a, **k) -> Any:
        return self._run("complete_json", *a, **k)

    def stream_chat(self, *a, **k) -> Iterator[str]:
        try:
            stream = self._primary.stream_chat(*a, **k)
            first = next(stream, None)
            self.last_used = self.label_primary
        except Exception as exc:
            if self._fallback is None or not should_fail_over(exc):
                raise
            self.failures += 1
            self.log(f"[llm] {self.label_primary} failed ({exc}); "
                     f"falling back to {self.label_fallback}")
            self.last_used = self.label_fallback
            try:
                fallback_stream = self._fallback.stream_chat(*a, **k)
                fallback_first = next(fallback_stream, None)
            except Exception as fallback_exc:
                raise self._both_failed(exc, fallback_exc) from fallback_exc
            if fallback_first is not None:
                yield fallback_first
            yield from fallback_stream
            return
        if first is not None:
            yield first
        yield from stream

    # ---------------- introspection ----------------
    def available_models(self) -> list[str]:
        return self._run("available_models")

    def model_catalog(self) -> list[dict]:
        return self._run("model_catalog")

    def health_check(self) -> dict:
        """Both backends, always — this is the one place you want the full picture."""
        out: dict[str, Any] = {"router": True, "last_used": self.last_used,
                               "failures": self.failures}
        for label, client in ((self.label_primary, self._primary),
                              (self.label_fallback, self._fallback)):
            if client is None:
                continue
            try:
                out[label] = {"ok": True, **client.health_check()}
            except Exception as exc:
                out[label] = {"ok": False, "host": getattr(client, "host", ""),
                              "error": str(exc)}
        def usable(label: str, key: str) -> bool:
            # Reachable is not enough: the backend must also have the model.
            entry = out.get(label, {})
            return bool(entry.get("ok")) and bool(entry.get(key, True))

        out["models"] = (out.get(self.label_primary, {}).get("models")
                         or out.get(self.label_fallback, {}).get("models") or [])
        out["host"] = out.get(self.label_primary, {}).get("host", "")
        out["primary_ok"] = usable(self.label_primary, "primary_ok") or \
            usable(self.label_fallback, "primary_ok")
        out["fast_ok"] = usable(self.label_primary, "fast_ok") or \
            usable(self.label_fallback, "fast_ok")
        return out

    def unload(self, model: str | None = None) -> None:
        for c in (self._primary, self._fallback):
            if c is not None:
                try:
                    c.unload(model)
                except Exception:
                    pass

    def loaded(self) -> list[dict]:
        out: list[dict] = []
        for c in (self._primary, self._fallback):
            if c is not None:
                try:
                    out.extend(c.loaded())
                except Exception:
                    pass
        return out
