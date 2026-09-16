"""One place to turn logging on.

The library logs through `logging` (retrieval degradations, failovers), but
nothing configured a handler, so those messages went nowhere. A deployment
wants them on stderr, where systemd/journald or Docker collects them.
"""
from __future__ import annotations

import logging
import os

_CONFIGURED = False


def setup_logging(level: str | int | None = None) -> None:
    """Send bookrag logs to stderr. Level from BOOKRAG_LOG_LEVEL, default WARNING.

    Safe to call repeatedly and from any entry point (CLI, Streamlit, tests);
    only the first call configures anything.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return
    chosen = level if level is not None else os.environ.get("BOOKRAG_LOG_LEVEL", "WARNING")
    if isinstance(chosen, str):
        chosen = getattr(logging, chosen.strip().upper(), logging.WARNING)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger = logging.getLogger("bookrag")
    logger.handlers = [handler]
    logger.setLevel(chosen)
    logger.propagate = False
    _CONFIGURED = True
