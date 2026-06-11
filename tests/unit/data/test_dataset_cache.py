from __future__ import annotations

import contextlib
from unittest.mock import MagicMock

import pandas as pd
import pytest

from automl.data import cache as dataset_cache_module
from automl.data import registry as data_registry
from automl.data.dataset import Dataset
from automl.errors import DataError, StorageError

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
    frame = pd.DataFrame({"y": [1]})
    registry_frame = pd.DataFrame({"name": ["y"], "dtype": ["int64"]})

    def fake_download(uri, dest, **kwargs):
        if str(uri).endswith("data.parquet"):
            frame.to_parquet(dest, index=False)
        else:
            registry_frame.to_csv(dest, index=False)

    monkeypatch.setattr(data_registry.gcs, "download_to_file", fake_download)
    data_registry._read_dataset_files(dataset)
    assert data_registry.evict_dataset_entry(dataset) is True
    assert data_registry.evict_dataset_entry(dataset) is False


# ---------------------------------------------------------------------------
# Fix 1: corrupt cached file self-heals (evict-and-repopulate-once)
# ---------------------------------------------------------------------------


def _make_fake_download(frame, registry_frame, downloads):
    """Return a fake download function that records calls and writes valid bytes."""

    def fake_download(uri, dest, **kwargs):
        downloads.append(uri)
        if str(uri).endswith("data.parquet"):
            frame.to_parquet(dest, index=False)
        else:
            registry_frame.to_csv(dest, index=False)

    return fake_download


def test_corrupt_cached_parquet_self_heals(tmp_path, monkeypatch):
    """Corrupt cached parquet triggers evict + re-download; result is correct."""
    monkeypatch.setenv("AUTOML_CACHE_DIR", str(tmp_path))
    dataset = _dataset()
    frame = pd.DataFrame({"y": [10, 20]})
    registry_frame = pd.DataFrame({"name": ["y"], "dtype": ["int64"]})
    downloads = []
    monkeypatch.setattr(
        data_registry.gcs,
        "download_to_file",
        _make_fake_download(frame, registry_frame, downloads),
    )

    # Populate the cache with valid bytes.
    data_registry._read_dataset_files(dataset)
    assert len(downloads) == 2

    # Corrupt the cached parquet on disk.
    cache = dataset_cache_module.dataset_cache()
    key = dataset_cache_module.cache_key(dataset)
    cached_parquet = cache.path_for(key, "data.parquet")
    cached_parquet.write_bytes(b"not parquet")

    # The call must succeed, return the correct frame, and re-download.
    got_registry, got_frame = data_registry._read_dataset_files(dataset)
    pd.testing.assert_frame_equal(got_frame, frame)
    pd.testing.assert_frame_equal(got_registry, registry_frame)
    assert len(downloads) == 4  # 2 original + 2 self-heal re-downloads


def test_persistent_corrupt_download_raises_storage_error(tmp_path, monkeypatch):
    """If GCS keeps returning corrupt bytes, StorageError is raised (no loop)."""
    monkeypatch.setenv("AUTOML_CACHE_DIR", str(tmp_path))
    dataset = _dataset()

    def always_corrupt(uri, dest, **kwargs):
        dest.write_bytes(b"garbage")

    monkeypatch.setattr(data_registry.gcs, "download_to_file", always_corrupt)

    with pytest.raises(StorageError):
        data_registry._read_dataset_files(dataset)


# ---------------------------------------------------------------------------
# Fix 3: DataError during validate_loaded_dataset evicts + re-raises
# ---------------------------------------------------------------------------


def test_load_dataset_by_id_evicts_on_data_error(tmp_path, monkeypatch):
    """DataError from validate_loaded_dataset causes cache eviction and re-raise."""
    monkeypatch.setenv("AUTOML_CACHE_DIR", str(tmp_path))
    dataset = _dataset()
    frame = pd.DataFrame({"y": [1, 2]})
    registry_frame = pd.DataFrame({"name": ["y"], "dtype": ["int64"]})

    monkeypatch.setattr(
        data_registry.gcs,
        "download_to_file",
        _make_fake_download(frame, registry_frame, []),
    )

    # Fake experiment_artifacts.read_dataset_record to return our dataset dict.
    dataset_dict = {
        "id": dataset.id,
        "identity_hash": dataset.identity_hash,
        "component_hashes": dataset.component_hashes.to_dict(),
        "gcs_bucket": dataset.gcs_bucket,
        "project_name": dataset.project_name,
        "created_at": dataset.created_at,
        "source_identity": dataset.source_identity,
        "n_rows": dataset.n_rows,
        "n_columns": dataset.n_columns,
        "target_column": dataset.target_column,
    }
    monkeypatch.setattr(
        data_registry.experiment_artifacts,
        "read_dataset_record",
        lambda dataset_id, **_kwargs: dataset_dict,
    )

    # Fake mlflow_client.bound_for as a no-op context manager.
    @contextlib.contextmanager
    def fake_bound_for(*args, **kwargs):
        yield

    monkeypatch.setattr(data_registry.mlflow_client, "bound_for", fake_bound_for)

    # Force validate_loaded_dataset to raise DataError.
    monkeypatch.setattr(
        data_registry,
        "validate_loaded_dataset",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(DataError("forced")),
    )

    # Cache entry should not exist yet; populate it first via _read_dataset_files.
    data_registry._read_dataset_files(dataset)
    cache = dataset_cache_module.dataset_cache()
    key = dataset_cache_module.cache_key(dataset)
    assert cache.path_for(key, "data.parquet").exists()

    # load_dataset_by_id must raise DataError and evict the cache entry.
    session_mock = MagicMock()
    session_mock.active_experiment_id = "exp_001"
    session_mock.config.require_run_config.return_value.splits.resolve.return_value = None

    with pytest.raises(DataError):
        data_registry.load_dataset_by_id(dataset.id, session=session_mock)

    # Cache entry must be gone.
    assert not cache.path_for(key, "data.parquet").exists()
