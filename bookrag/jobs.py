"""Long-running work off the Streamlit script thread.

Streamlit reruns the page script on every click, so an index build or a paper
generation done inline freezes the tab and is killed by the first interaction.
A Job runs the work in a daemon thread and keeps its log, status and result
where any later rerun -- or, for a process-wide job, any session -- can read
them.

Cancellation is cooperative: the work function receives `should_cancel` and
checks it between units of work (a book, an embedding batch, a question), so a
cancel takes effect when the current unit finishes. No thread is killed
mid-write.
"""
from __future__ import annotations

import threading
import time
import traceback
from typing import Any, Callable


class Cancelled(Exception):
    """Raised by work that stopped because cancellation was requested."""


class Job:
    MAX_LOG = 500

    def __init__(self, name: str, fn: Callable[..., Any], *args, **kwargs):
        self.name = name
        self.meta: dict = {}
        self.status = "pending"        # pending | running | done | cancelled | failed
        self.result: Any = None
        self.error = ""
        self.started = 0.0
        self.finished = 0.0
        self._log: list[str] = []
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._thread = threading.Thread(target=self._run, args=(fn, args, kwargs),
                                        name=f"job:{name}", daemon=True)

    def start(self) -> "Job":
        self.started = time.time()
        self.status = "running"
        self._thread.start()
        return self

    def _run(self, fn, args, kwargs) -> None:
        try:
            self.result = fn(*args, progress=self.progress,
                             should_cancel=self._cancel.is_set, **kwargs)
            # Work that stops early but still returns something useful (a
            # partial paper) reports it on the result instead of raising.
            report = getattr(self.result, "report", self.result)
            status = "cancelled" if getattr(report, "cancelled", False) else "done"
        except Cancelled as exc:
            self.progress(str(exc) or "Cancelled.")
            status = "cancelled"
        except BaseException as exc:            # surface everything to the UI
            self.error = f"{type(exc).__name__}: {exc}"
            self.progress(traceback.format_exc(limit=4))
            status = "failed"
        self.finished = time.time()
        self.status = status

    def progress(self, msg: Any) -> None:
        with self._lock:
            self._log.append(str(msg))
            if len(self._log) > self.MAX_LOG:
                del self._log[:-self.MAX_LOG]

    def tail(self, n: int = 12) -> list[str]:
        with self._lock:
            return list(self._log[-n:])

    def cancel(self) -> None:
        self._cancel.set()

    @property
    def cancel_requested(self) -> bool:
        return self._cancel.is_set()

    @property
    def running(self) -> bool:
        return self.status == "running"

    @property
    def elapsed(self) -> float:
        return (self.finished or time.time()) - self.started if self.started else 0.0

    def wait(self, timeout: float | None = None) -> bool:
        """Block until the work finishes; True if it did within timeout."""
        self._thread.join(timeout)
        return not self._thread.is_alive()
