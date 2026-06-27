"""Structured logging for vibe-thinker.

Provides a thin wrapper around Python's ``logging`` module with:
  - Consistent format (timestamp, level, component, message)
  - Component-based loggers (e.g. ``get_logger("Federation")``)
  - Level control via ``VIBE_THINKER_LOG_LEVEL`` env var
  - Backward-compatible: existing ``print()`` statements can be
    incrementally migrated to ``log.info()`` / ``log.warning()`` etc.

Usage:
    from vt_logging import get_logger

    log = get_logger("Federation")
    log.info("Server started on %s:%d", host, port)
    log.warning("Reaper detected %d zombie claims", len(reaped))
    log.error("Failed to parse constraints: %s", err)

The log level is controlled by the ``VIBE_THINKER_LOG_LEVEL`` env var
(default: INFO). Set to DEBUG for verbose output, WARNING for production.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Dict

# Track configured loggers so we don't add duplicate handlers.
_loggers: Dict[str, logging.Logger] = {}

# Read log level from env, default to INFO.
_LEVEL = os.environ.get("VIBE_THINKER_LOG_LEVEL", "INFO").upper()
_NUMERIC_LEVEL = getattr(logging, _LEVEL, logging.INFO)

# Configure the root handler once.
_root_configured = False


def _ensure_root_handler() -> None:
    global _root_configured
    if _root_configured:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root = logging.getLogger("vibe-thinker")
    root.addHandler(handler)
    root.setLevel(_NUMERIC_LEVEL)
    _root_configured = True


def get_logger(component: str) -> logging.Logger:
    """Get a logger for a specific component (e.g. "Federation", "CLR").

    The logger name is prefixed with ``vibe-thinker.`` so all loggers
    share the root handler configured above.
    """
    _ensure_root_handler()
    name = f"vibe-thinker.{component}"
    if name in _loggers:
        return _loggers[name]
    logger = logging.getLogger(name)
    logger.setLevel(_NUMERIC_LEVEL)
    # Don't propagate to the root logging.getLogger() — we have our own handler.
    logger.propagate = False
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(handler)
    _loggers[name] = logger
    return logger
