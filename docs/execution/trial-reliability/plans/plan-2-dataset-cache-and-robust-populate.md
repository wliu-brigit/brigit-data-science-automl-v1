# Dataset Cache + Robust Populate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every dataset read after the first becomes a local-disk read: a content-addressed file cache at the data read seam, populated once per dataset by a robust (streamed, checksummed, retried) download.

**Architecture:** Three altitudes (design §4–§5): a generic key→file cache in `automl/utils/io/blob_cache.py` (leaf — knows nothing about datasets); dataset policy in `automl/data/cache.py` + the read swap inside `data/registry.load_dataset_by_id` (key = `dataset.id` + `component_hashes.data_content`); thin `automl data cache {list,prune,clear}` CLI verbs. `validate_loaded_dataset` stays as defense in depth; a manifest mismatch evicts the cache entry so a bad entry can never wedge.

**Tech Stack:** Python 3.13, pandas, google-cloud-storage 3.11.0, pytest via `uv run`.

**Design:** `docs/execution/trial-reliability/design.md` §4–§5 and §9 (empirical checksum check).

---

### Task 1: Empirical check — what does the GCS client already validate?

**Files:** none (evidence step; paste output into the PR/commit message of Task 5).

- [ ] **Step 1: Inspect the installed client's download defaults**

Run:

```bash
uv run python -c "
import inspect
from google.cloud.storage.blob import Blob
print(inspect.signature(Blob.download_to_filename))
print(inspect.signature(Blob.download_as_bytes))
from google.cloud.storage.retry import DEFAULT_RETRY
print(type(DEFAULT_RETRY), getattr(DEFAULT_RETRY, '_timeout', None))
print([m for m in dir(DEFAULT_RETRY) if m.startswith('with_')])
"
```

Expected: both signatures show a `checksum=` parameter (note its default — `"auto"` or `"md5"` depending on version) and `retry=DEFAULT_RETRY`. Note which `with_*` modifiers exist (`with_timeout` vs deprecated `with_deadline`).

- [ ] **Step 2: Record the finding**

Whatever the defaults are, the populate in Task 4 sets `checksum="crc32c"` and `retry=` **explicitly** so behavior is pinned, not inherited. If the default already validates, say so in the Task 5 commit message — per design §9, do not claim the checksum as net-new behavior if it isn't.

---

### Task 2: Generic blob cache (`automl/utils/io/blob_cache.py`)

**Files:**
- Create: `automl/utils/io/blob_cache.py`
- Modify: `automl/utils/io/__init__.py` (add `"blob_cache"` to `__all__`)
- Test: `tests/unit/utils/test_blob_cache.py` (create)

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/utils/test_blob_cache.py`:

```python
from __future__ import annotations

import os
import time
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/utils/test_blob_cache.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'automl.utils.io.blob_cache'`.

- [ ] **Step 3: Implement `BlobCache`**

Create `automl/utils/io/blob_cache.py`:

```python
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
```

Update `automl/utils/io/__init__.py`:

```python
"""I/O helpers."""

__all__ = ["blob_cache", "gcs"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/utils/test_blob_cache.py -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add automl/utils/io/blob_cache.py automl/utils/io/__init__.py tests/unit/utils/test_blob_cache.py
git commit -m "feat(utils): generic content-addressed blob cache with LRU-on-write eviction"
```

---

### Task 3: Robust download primitive (`gcs.download_to_file`)

**Files:**
- Modify: `automl/utils/io/gcs.py` (new function + `__all__`)
- Test: `tests/unit/utils/test_gcs.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/utils/test_gcs.py` (follow the file's existing fake-client pattern; if it has one, reuse it — the shape below is self-contained):

```python
def test_download_to_file_passes_checksum_and_retry(tmp_path):
    from automl.utils.io import gcs

    captured = {}

    class _Blob:
        def download_to_filename(self, filename, *, checksum=None, retry=None):
            captured["filename"] = filename
            captured["checksum"] = checksum
            captured["retry"] = retry
            Path(filename).write_bytes(b"payload")

    class _Bucket:
        def blob(self, name):
            captured["blob_name"] = name
            return _Blob()

    class _Client:
        def bucket(self, name):
            captured["bucket"] = name
            return _Bucket()

    dest = tmp_path / "out.bin"
    gcs.download_to_file("gs://bucket-x/path/data.parquet", dest, client=_Client())
    assert captured["bucket"] == "bucket-x"
    assert captured["blob_name"] == "path/data.parquet"
    assert captured["checksum"] == "crc32c"
    assert captured["retry"] is not None
    assert dest.read_bytes() == b"payload"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/utils/test_gcs.py -q -k download_to_file`
Expected: FAIL — `AttributeError: module ... has no attribute 'download_to_file'`.

- [ ] **Step 3: Implement**

Add to `automl/utils/io/gcs.py` (near `read_bytes`), using the retry modifier confirmed in Task 1 (`with_timeout` on current versions; if only `with_deadline` exists, use that):

```python
def download_to_file(
    uri: str,
    dest: "str | os.PathLike[str]",
    *,
    client: Any | None = None,
) -> None:
    """Robust whole-object download: streamed to disk, checksummed, retried.

    Unlike ``read_bytes``/``read_parquet`` (single-shot into RAM), this uses the
    client's chunked transfer with explicit crc32c validation and an extended
    retry window, for multi-GB objects where a long-lived connection sees
    transients (design: trial-reliability §5).
    """
    from google.cloud.storage.retry import DEFAULT_RETRY

    bucket, blob_name = parse_gcs_uri(uri)
    blob = _client_or_default(client).bucket(bucket).blob(blob_name)
    blob.download_to_filename(
        str(dest),
        checksum="crc32c",
        retry=DEFAULT_RETRY.with_timeout(300.0),
    )
```

Add `"download_to_file"` to `__all__` (keep it sorted).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/utils/test_gcs.py -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add automl/utils/io/gcs.py tests/unit/utils/test_gcs.py
git commit -m "feat(utils): gcs.download_to_file — streamed, crc32c-checked, retried download"
```

---

### Task 4: Dataset cache policy (`automl/data/cache.py`)

**Files:**
- Create: `automl/data/cache.py`
- Test: `tests/unit/data/test_dataset_cache.py` (create)

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/data/test_dataset_cache.py`:

```python
from __future__ import annotations

import pytest

from automl.data import cache as dataset_cache_module
from automl.data.dataset import ComponentHashes, Dataset

pytestmark = pytest.mark.unit


def _dataset(dataset_id: str = "ds_001", content: str = "sha256:" + "ab" * 32) -> Dataset:
    return Dataset.from_dict(
        {
            "id": dataset_id,
            "identity_hash": "sha256:identity",
            "component_hashes": {
                "source_identity": "sha256:src",
                "feature_registry": "sha256:reg",
                "data_content": content,
                "schema": "sha256:schema",
            },
            "gcs_bucket": "bucket-x",
            "project_name": "proj",
            "created_at": "2026-06-10T00:00:00Z",
            "source_identity": {},
            "n_rows": 1,
            "n_columns": 1,
            "target_column": "y",
        }
    )


def test_cache_key_combines_id_and_content_hash():
    key = dataset_cache_module.cache_key(_dataset())
    assert key.startswith("ds_001-")
    assert "ab" * 8 in key  # first 16 hex chars of the content hash


def test_cache_key_changes_with_content_hash():
    left = dataset_cache_module.cache_key(_dataset(content="sha256:" + "ab" * 32))
    right = dataset_cache_module.cache_key(_dataset(content="sha256:" + "cd" * 32))
    assert left != right


def test_dataset_cache_honors_env_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOML_CACHE_DIR", str(tmp_path / "elsewhere"))
    monkeypatch.setenv("AUTOML_CACHE_MAX_BYTES", "12345")
    cache = dataset_cache_module.dataset_cache()
    assert cache.root == tmp_path / "elsewhere" / "datasets"
    assert cache.max_bytes == 12345


def test_list_prune_clear_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOML_CACHE_DIR", str(tmp_path))
    cache = dataset_cache_module.dataset_cache()
    cache.get_or_populate("k1", "data.parquet", lambda p: p.write_bytes(b"x" * 10))
    listed = dataset_cache_module.list_cache()
    assert listed[0]["key"] == "k1"
    assert listed[0]["size_bytes"] == 10
    assert dataset_cache_module.clear_cache() == 1
    assert dataset_cache_module.list_cache() == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/data/test_dataset_cache.py -q`
Expected: FAIL — `ImportError: cannot import name 'cache' from 'automl.data'`.

- [ ] **Step 3: Implement**

Create `automl/data/cache.py`:

```python
"""Local dataset-bytes cache: policy over the generic blob cache.

The cache is keyed by dataset id + manifest content hash (content-addressed:
identical key always means identical bytes, so there is no invalidation).
GCS stays the single source of truth; ``validate_loaded_dataset`` still
verifies every parsed frame against the manifest (design: trial-reliability
§4). Root is never inside the repo; override via AUTOML_CACHE_DIR.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from automl.utils.io.blob_cache import BlobCache

ENV_CACHE_DIR = "AUTOML_CACHE_DIR"
ENV_CACHE_MAX_BYTES = "AUTOML_CACHE_MAX_BYTES"
DEFAULT_CACHE_DIR = "~/.cache/brigit-automl"
DEFAULT_MAX_BYTES = 20 * 1024**3  # 20 GB — a couple of full datasets


def dataset_cache() -> BlobCache:
    root = Path(os.environ.get(ENV_CACHE_DIR) or DEFAULT_CACHE_DIR).expanduser()
    max_bytes = int(os.environ.get(ENV_CACHE_MAX_BYTES) or DEFAULT_MAX_BYTES)
    return BlobCache(root / "datasets", max_bytes=max_bytes)


def cache_key(dataset: Any) -> str:
    content = str(dataset.component_hashes.data_content).removeprefix("sha256:")
    return f"{dataset.id}-{content[:16]}"


def list_cache() -> list[dict[str, Any]]:
    return [
        {
            "key": entry.key,
            "path": str(entry.path),
            "size_bytes": entry.size_bytes,
            "last_used": datetime.fromtimestamp(entry.last_used, tz=timezone.utc).isoformat(),
        }
        for entry in dataset_cache().entries()
    ]


def prune_cache(*, max_bytes: int | None = None) -> int:
    """Evict least-recently-used entries until under the size cap."""
    return dataset_cache().prune(max_bytes=max_bytes)


def clear_cache() -> int:
    return dataset_cache().clear()


__all__ = ["cache_key", "clear_cache", "dataset_cache", "list_cache", "prune_cache"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/data/test_dataset_cache.py -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add automl/data/cache.py tests/unit/data/test_dataset_cache.py
git commit -m "feat(data): dataset cache policy — content-addressed keys, env-tunable root/cap"
```

---

### Task 5: Wire the cache into the read seam (`load_dataset_by_id`)

**Files:**
- Modify: `automl/data/registry.py` (the two read lines in `load_dataset_by_id`, ~lines 66–70)
- Test: `tests/unit/data/test_dataset_cache.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/data/test_dataset_cache.py`:

```python
import pandas as pd

from automl.data import registry as data_registry


def test_read_dataset_files_populates_once_then_hits(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOML_CACHE_DIR", str(tmp_path))
    dataset = _dataset()
    frame = pd.DataFrame({"y": [1, 2]})
    registry_frame = pd.DataFrame({"name": ["y"], "dtype": ["int64"]})
    downloads = []

    def fake_download(uri, dest, **kwargs):
        downloads.append(uri)
        if str(uri).endswith("data.parquet"):
            frame.to_parquet(dest, index=False)
        else:
            registry_frame.to_csv(dest, index=False)

    monkeypatch.setattr(data_registry.gcs, "download_to_file", fake_download)

    got_registry, got_frame = data_registry._read_dataset_files(dataset)
    pd.testing.assert_frame_equal(got_frame, frame)
    pd.testing.assert_frame_equal(got_registry, registry_frame)
    assert len(downloads) == 2

    data_registry._read_dataset_files(dataset)
    assert len(downloads) == 2  # cache hit: no new downloads


def test_read_dataset_files_wraps_failures_as_storage_error(tmp_path, monkeypatch):
    from automl.errors import StorageError

    monkeypatch.setenv("AUTOML_CACHE_DIR", str(tmp_path))

    def explode(uri, dest, **kwargs):
        raise ConnectionError("reset by peer")

    monkeypatch.setattr(data_registry.gcs, "download_to_file", explode)
    with pytest.raises(StorageError):
        data_registry._read_dataset_files(_dataset())


def test_evict_dataset_entry_removes_cached_files(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOML_CACHE_DIR", str(tmp_path))
    dataset = _dataset()

    def fake_download(uri, dest, **kwargs):
        pd.DataFrame({"y": [1]}).to_parquet(dest, index=False)

    monkeypatch.setattr(data_registry.gcs, "download_to_file", fake_download)
    data_registry._read_dataset_files(dataset)
    assert data_registry.evict_dataset_entry(dataset) is True
    assert data_registry.evict_dataset_entry(dataset) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/data/test_dataset_cache.py -q -k read_dataset_files`
Expected: FAIL — `AttributeError: ... has no attribute '_read_dataset_files'`.

- [ ] **Step 3: Implement the seam swap**

In `automl/data/registry.py`:

Add imports:

```python
from automl.data.cache import cache_key, dataset_cache
from automl.errors import StorageError
from automl.utils.io import gcs
```

Replace the two read lines inside `load_dataset_by_id`:

```python
    registry_frame = experiment_artifacts.read_registry(dataset.registry_gcs_uri)
    ...
    df = experiment_artifacts.read_dataset_frame(dataset.data_gcs_uri)
```

with:

```python
    registry_frame, df = _read_dataset_files(dataset)
```

(keep `FeatureRegistry.from_dataframe(registry_frame)` and everything after unchanged), and wrap the existing `validate_loaded_dataset(loaded, dataset)` call so a manifest mismatch can never wedge a bad cache entry:

```python
    try:
        validate_loaded_dataset(loaded, dataset)
    except DataError:
        # Defense in depth: the cached bytes failed the manifest check. Evict
        # so the next read re-populates from GCS instead of re-failing forever.
        evict_dataset_entry(dataset)
        raise
```

(add `DataError` to the imports from `automl.errors`). Then add the new module-level functions:

```python
def _read_dataset_files(dataset):
    """Read registry + frame through the local content-addressed cache.

    Non-GCS URIs (and records without a content hash) bypass the cache and
    use the direct artifact readers.
    """
    if not (
        gcs.is_gcs_uri(dataset.data_gcs_uri)
        and dataset.component_hashes.data_content
    ):
        return (
            experiment_artifacts.read_registry(dataset.registry_gcs_uri),
            experiment_artifacts.read_dataset_frame(dataset.data_gcs_uri),
        )
    import pandas as pd

    cache = dataset_cache()
    key = cache_key(dataset)
    try:
        data_path = cache.get_or_populate(
            key,
            "data.parquet",
            lambda tmp: gcs.download_to_file(dataset.data_gcs_uri, tmp),
        )
        registry_path = cache.get_or_populate(
            key,
            "feature_registry.csv",
            lambda tmp: gcs.download_to_file(dataset.registry_gcs_uri, tmp),
        )
        return pd.read_csv(registry_path), pd.read_parquet(data_path)
    except Exception as exc:
        raise StorageError(f"Failed to read dataset {dataset.id!r}") from exc


def evict_dataset_entry(dataset) -> bool:
    """Drop a dataset's cached bytes (used on manifest-verification failure)."""
    return dataset_cache().remove(cache_key(dataset))
```

- [ ] **Step 4: Run the data + utils unit suites**

Run: `uv run pytest tests/unit/data tests/unit/utils -q`
Expected: all PASS. If an existing test monkeypatched `experiment_artifacts.read_dataset_frame` to fake a load, point it at `data_registry.gcs.download_to_file` (or set `AUTOML_CACHE_DIR` to a tmp dir) instead — same fake, new seam.

- [ ] **Step 5: Run the full unit suite**

Run: `uv run pytest tests/unit -q`
Expected: all PASS (runner/eval suites exercise `load_dataset_by_id` indirectly through fakes; fix any that pinned the old read path the same way as Step 4).

- [ ] **Step 6: Commit**

```bash
git add automl/data/registry.py tests/unit/data/test_dataset_cache.py
git commit -m "feat(data): route dataset reads through the content-addressed local cache"
```

---

### Task 6: CLI — `automl data cache {list,prune,clear}`

**Files:**
- Modify: `automl/cli/data.py`
- Test: `tests/unit/cli/test_cli_catalog.py` (append) — plus run the contract suite.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/cli/test_cli_catalog.py`:

```python
def test_data_cache_verbs_exist():
    from automl.cli import build_parser

    parser = build_parser()
    cache_parser = _subparser(parser, "data", "cache")
    subparser_action = next(
        action
        for action in cache_parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    assert set(subparser_action.choices) == {"list", "prune", "clear"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/cli/test_cli_catalog.py -q -k data_cache`
Expected: FAIL — `KeyError: 'cache'`.

- [ ] **Step 3: Implement the verbs**

In `automl/cli/data.py`, extend the import and `add_parser`:

```python
from automl.data.cache import clear_cache, list_cache, prune_cache
```

Inside `add_parser`, after the `materialize` block:

```python
    cache_parser = data_sub.add_parser("cache")
    cache_sub = cache_parser.add_subparsers(dest="cache_action", required=True)

    cache_list = cache_sub.add_parser("list")
    cache_list.set_defaults(func=_cache_list)

    cache_prune = cache_sub.add_parser("prune")
    cache_prune.add_argument(
        "--max-bytes",
        type=int,
        default=None,
        help="evict least-recently-used entries until under this size (default: configured cap)",
    )
    cache_prune.set_defaults(func=_cache_prune)

    cache_clear = cache_sub.add_parser("clear")
    cache_clear.set_defaults(func=_cache_clear)
```

And the handlers (no session needed — the cache is machine-local):

```python
def _cache_list(args: argparse.Namespace) -> int:
    print_json(list_cache())
    return 0


def _cache_prune(args: argparse.Namespace) -> int:
    print_json({"evicted": prune_cache(max_bytes=args.max_bytes)})
    return 0


def _cache_clear(args: argparse.Namespace) -> int:
    print_json({"removed": clear_cache()})
    return 0
```

- [ ] **Step 4: Run the CLI + contract suites**

Run: `uv run pytest tests/unit/cli tests/contracts -q`
Expected: all PASS. If a contract/ratchet test pins the CLI verb catalog or doc phrases, add the new `data cache` verbs to the pinned shape in the same commit.

- [ ] **Step 5: Smoke the verbs by hand**

```bash
uv run automl data cache list
uv run automl data cache prune
uv run automl data cache clear
```

Expected: JSON output (`[]` / `{"evicted": 0}` / `{"removed": 0}` on an empty cache), exit 0.

- [ ] **Step 6: Commit**

```bash
git add automl/cli/data.py tests/unit/cli/test_cli_catalog.py
git commit -m "feat(cli): automl data cache list/prune/clear"
```

---

## Done criteria

- `uv run pytest tests/unit tests/contracts -q` green.
- A trial against a GCS dataset performs **one** download per object per machine (`automl data cache list` shows the entry; re-running the trial adds no downloads).
- `AUTOML_CACHE_DIR` relocates the cache for both parent and child processes (it rides `os.environ`, which `serving_validation.py` already copies into `child_env`).
- Task 1's empirical finding about default checksums is recorded in the Task 5 commit/PR message.
- Design §10's integration item (populate path + hit path): the seam tests in Task 5 cover both with a faked download; the live populate is exercised by the standard e2e pass (`AUTOML_E2E=1 uv run pytest tests/e2e -q` against `projects/example_homecredit/`, `qa/` namespace). Add a dedicated `tests/integration/` case only if that tier already fakes MLflow dataset records conveniently — do not build new MLflow fakes just for this.
