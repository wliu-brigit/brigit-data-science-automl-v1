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
