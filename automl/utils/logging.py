"""Standard process logging setup."""

from __future__ import annotations

import logging


def configure_logging(name: str, *, level: int = logging.INFO) -> logging.Logger:
    """Configure root logging and return a named logger."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    return logging.getLogger(name)


__all__ = ["configure_logging"]
