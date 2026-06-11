from __future__ import annotations

import os
from pathlib import Path

import pytest

from automl.utils.io.blob_cache import BlobCache

pytestmark = pytest.mark.unit


def _populate_with(content: bytes):
    def populate(tmp_path: Path) -> None:
        tmp_path.write_bytes(content)

    return populate


def test_miss_populates_then_hit_skips_populate(tmp_path):
    cache = BlobCache(tmp_path / "cache", max_bytes=1_000_000)
    calls = []

    def populate(dest: Path) -> None:
        calls.append(1)
        dest.write_bytes(b"hello")

    first = cache.get_or_populate("key-1", "data.bin", populate)
    second = cache.get_or_populate("key-1", "data.bin", populate)
    assert first == second
    assert first.read_bytes() == b"hello"
    assert len(calls) == 1


def test_failed_populate_leaves_no_entry(tmp_path):
    cache = BlobCache(tmp_path / "cache", max_bytes=1_000_000)

    def explode(dest: Path) -> None:
        dest.write_bytes(b"partial")
        raise RuntimeError("network died")

    with pytest.raises(RuntimeError):
        cache.get_or_populate("key-1", "data.bin", explode)
    assert not cache.path_for("key-1", "data.bin").exists()
    # And a later populate succeeds cleanly.
    path = cache.get_or_populate("key-1", "data.bin", _populate_with(b"ok"))
    assert path.read_bytes() == b"ok"


def test_lru_eviction_at_cap_keeps_most_recent(tmp_path):
    cache = BlobCache(tmp_path / "cache", max_bytes=250)
    cache.get_or_populate("old", "data.bin", _populate_with(b"x" * 100))
    os.utime(cache.path_for("old", "data.bin").parent, (1, 1))  # force old mtime
    cache.get_or_populate("mid", "data.bin", _populate_with(b"x" * 100))
    os.utime(cache.path_for("mid", "data.bin").parent, (2, 2))
    cache.get_or_populate("new", "data.bin", _populate_with(b"x" * 100))
    # 300 bytes > 250 cap: the least-recently-used entry ("old") is evicted.
    assert not cache.path_for("old", "data.bin").exists()
    assert cache.path_for("mid", "data.bin").exists()
    assert cache.path_for("new", "data.bin").exists()


def test_hit_touches_recency(tmp_path):
    cache = BlobCache(tmp_path / "cache", max_bytes=1_000_000)
    cache.get_or_populate("key-1", "data.bin", _populate_with(b"x"))
    entry_dir = cache.path_for("key-1", "data.bin").parent
    os.utime(entry_dir, (1, 1))
    cache.get_or_populate("key-1", "data.bin", _populate_with(b"x"))
    assert entry_dir.stat().st_mtime > 1


def test_entries_remove_clear(tmp_path):
    cache = BlobCache(tmp_path / "cache", max_bytes=1_000_000)
    cache.get_or_populate("a", "data.bin", _populate_with(b"xx"))
    cache.get_or_populate("b", "data.bin", _populate_with(b"yyyy"))
    entries = {entry.key: entry for entry in cache.entries()}
    assert set(entries) == {"a", "b"}
    assert entries["b"].size_bytes == 4
    assert cache.total_bytes() == 6
    assert cache.remove("a") is True
    assert cache.remove("a") is False
    assert cache.clear() == 1
    assert cache.entries() == []


def test_keys_are_sanitized_to_safe_dirnames(tmp_path):
    cache = BlobCache(tmp_path / "cache", max_bytes=1_000_000)
    path = cache.get_or_populate("ds/01:sha256:ab", "data.bin", _populate_with(b"x"))
    assert path.parent.name == "ds_01_sha256_ab"
    assert path.parent.parent == cache.root


# ---------------------------------------------------------------------------
# Fix 2: .tmp orphan sweeping
# ---------------------------------------------------------------------------


def test_get_or_populate_sweeps_old_tmp_orphans(tmp_path):
    """Old .tmp orphans (mtime far in the past) are removed by get_or_populate."""
    cache = BlobCache(tmp_path / "cache", max_bytes=1_000_000)
    # Plant a fake orphan with an old mtime.
    tmp_dir = cache.root / ".tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    orphan = tmp_dir / "orphan_old.tmp"
    orphan.write_bytes(b"stale bytes")
    os.utime(orphan, (1, 1))  # mtime = epoch+1s, definitely old

    # Also plant a fresh tmp file that should survive.
    fresh = tmp_dir / "orphan_fresh.tmp"
    fresh.write_bytes(b"fresh bytes")
    # Leave mtime as current (just created).

    cache.get_or_populate("key-fresh", "data.bin", _populate_with(b"ok"))

    assert not orphan.exists(), "old orphan should have been swept"
    assert fresh.exists(), "fresh tmp file should survive"


def test_clear_removes_tmp_dir_contents(tmp_path):
    """clear() removes all contents of .tmp regardless of age."""
    cache = BlobCache(tmp_path / "cache", max_bytes=1_000_000)
    tmp_dir = cache.root / ".tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    orphan = tmp_dir / "leftovers.tmp"
    orphan.write_bytes(b"crash leftovers")
    os.utime(orphan, (1, 1))

    cache.get_or_populate("k", "data.bin", _populate_with(b"x"))
    cache.clear()

    assert not orphan.exists(), ".tmp orphan should be removed by clear()"
    assert not any(tmp_dir.iterdir()) if tmp_dir.exists() else True
