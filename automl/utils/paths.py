"""Small path helpers for local filesystem and URI-like prefixes."""

from __future__ import annotations

from pathlib import Path


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a ``Path``."""
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def join_posix(*parts: str) -> str:
    """Join path parts using forward slashes, suitable for object-store keys."""
    return "/".join(part.strip("/") for part in parts if part.strip("/"))


__all__ = ["ensure_dir", "join_posix"]
