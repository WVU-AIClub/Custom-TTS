"""Logging helpers shared across the package."""

from __future__ import annotations

import logging

_CONFIGURED = False


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging once with a concise format."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger, ensuring logging is configured."""
    setup_logging()
    return logging.getLogger(name)
