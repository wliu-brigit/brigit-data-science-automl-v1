"""Generic content-addressed local file cache with LRU-on-write eviction.

Mechanism only — knows nothing about datasets or GCS. Callers own the key
scheme (content-address your keys: identical keys must mean identical bytes).
Atomicity: populate lands in a temp file and is renamed into place, so a
concurrent reader sees a complete file or a miss, never a partial write.
Eviction assumes a single writer (see docs/to-do/multi-runner-architecture.md).
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

_SAFE_SEGMENT_RE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class CacheEntry:
    key: str
    path: Path
    size_bytes: int
    last_used: float


class BlobCache:
    """Key -> directory of files, size-capped, least-recently-used out first."""

    def __init__(self, root: str | Path, *, max_bytes: int) -> None:
        self.root = Path(root).expanduser()
        self.max_bytes = int(max_bytes)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, key: str, filename: str) -> Path:
        return self.root / _safe_segment(key) / _safe_segment(filename)

    def get_or_populate(
        self,
        key: str,
        filename: str,
        populate: Callable[[Path], None],
    ) -> Path:
        target = self.path_for(key, filename)
        if target.exists():
            _touch(target.parent)
            return target
        tmp_dir = self.root / ".tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=tmp_dir)
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            populate(tmp_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(tmp_path, target)
            _touch(target.parent)
        finally:
            tmp_path.unlink(missing_ok=True)
        self._evict_over_cap(keep=target.parent)
        return target

    def entries(self) -> list[CacheEntry]:
        found: list[CacheEntry] = []
        for child in sorted(self.root.iterdir()) if self.root.exists() else []:
            if not child.is_dir() or child.name == ".tmp":
                continue
            size = sum(item.stat().st_size for item in child.rglob("*") if item.is_file())
            found.append(
                CacheEntry(
                    key=child.name,
                    path=child,
                    size_bytes=size,
                    last_used=child.stat().st_mtime,
                )
            )
        return found

    def total_bytes(self) -> int:
        return sum(entry.size_bytes for entry in self.entries())

    def remove(self, key: str) -> bool:
        entry_dir = self.root / _safe_segment(key)
        if not entry_dir.is_dir():
            return False
        shutil.rmtree(entry_dir, ignore_errors=True)
        return True

    def clear(self) -> int:
        removed = 0
        for entry in self.entries():
            shutil.rmtree(entry.path, ignore_errors=True)
            removed += 1
        return removed

    def prune(self, *, max_bytes: int | None = None) -> int:
        """Evict least-recently-used entries until under ``max_bytes``."""
        cap = self.max_bytes if max_bytes is None else int(max_bytes)
        return self._evict_over_cap(keep=None, cap=cap)

    def _evict_over_cap(self, *, keep: Path | None, cap: int | None = None) -> int:
        cap = self.max_bytes if cap is None else cap
        entries = sorted(self.entries(), key=lambda entry: entry.last_used)
        total = sum(entry.size_bytes for entry in entries)
        evicted = 0
        for entry in entries:
            if total <= cap:
                break
            if keep is not None and entry.path == keep:
                continue
            shutil.rmtree(entry.path, ignore_errors=True)
            total -= entry.size_bytes
            evicted += 1
        return evicted


def _safe_segment(value: str) -> str:
    cleaned = _SAFE_SEGMENT_RE.sub("_", value.strip())
    if not cleaned or cleaned in {".", ".."}:
        raise ValueError(f"cache segment must be a non-empty safe name, got {value!r}")
    return cleaned


def _touch(path: Path) -> None:
    now = time.time()
    try:
        os.utime(path, (now, now))
    except OSError:
        pass


__all__ = ["BlobCache", "CacheEntry"]
