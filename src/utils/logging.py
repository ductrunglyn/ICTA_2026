"""Project-wide logging helper."""

from __future__ import annotations

import logging
import sys
from typing import Optional

_CONFIGURED = False


def get_logger(name: str = "transval", level: int = logging.INFO) -> logging.Logger:
    """Return a configured :class:`logging.Logger`.

    A single stream handler is attached to the root the first time this is
    called; subsequent calls reuse the configuration.

    Args:
        name: Logger name.
        level: Logging level.

    Returns:
        Configured logger instance.
    """
    global _CONFIGURED
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stdout)
        fmt = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
        handler.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))
        root = logging.getLogger()
        root.handlers.clear()
        root.addHandler(handler)
        root.setLevel(level)
        _CONFIGURED = True
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger


def set_level(level: int, name: Optional[str] = None) -> None:
    """Set the logging level for a given logger (root by default)."""
    logging.getLogger(name).setLevel(level)
