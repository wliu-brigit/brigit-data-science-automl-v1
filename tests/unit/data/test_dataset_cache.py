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
