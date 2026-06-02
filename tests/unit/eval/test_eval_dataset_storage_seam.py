import pandas as pd
import pytest

from automl.errors import StorageError
from automl.mlflow.experiment import eval_datasets

pytestmark = pytest.mark.unit


def test_eval_dataset_storage_seam_delegates_gcs_operations(monkeypatch):
    frame = pd.DataFrame({"row_id": [1]})
    calls = []

    monkeypatch.setattr(
        eval_datasets.gcs,
        "write_json",
        lambda uri, payload, **kwargs: calls.append(("write_json", uri, payload, kwargs)),
    )
    monkeypatch.setattr(eval_datasets.gcs, "read_json", lambda uri: {"uri": uri})
    monkeypatch.setattr(
        eval_datasets.gcs,
        "write_parquet",
        lambda uri, payload, **kwargs: calls.append(("write_parquet", uri, payload, kwargs)),
    )
    monkeypatch.setattr(eval_datasets.gcs, "read_parquet", lambda uri: frame)
    monkeypatch.setattr(eval_datasets.gcs, "blob_exists", lambda uri: uri == "gs://bucket/a")
    monkeypatch.setattr(eval_datasets.gcs, "list_blob_names", lambda uri: [uri])
    monkeypatch.setattr(eval_datasets.gcs, "list_prefixes", lambda uri: [uri.rstrip("/") + "/"])

    eval_datasets.write_manifest("gs://bucket/manifest.json", {"a": 1}, overwrite=True)
    eval_datasets.write_frame("gs://bucket/data.parquet", frame, overwrite=False)

    assert eval_datasets.read_manifest("gs://bucket/manifest.json") == {
        "uri": "gs://bucket/manifest.json"
    }
    assert eval_datasets.read_frame("gs://bucket/data.parquet").equals(frame)
    assert eval_datasets.blob_exists("gs://bucket/a") is True
    assert eval_datasets.list_blob_names("gs://bucket/prefix") == ["gs://bucket/prefix"]
    assert eval_datasets.list_prefixes("gs://bucket/prefix") == ["gs://bucket/prefix/"]
    assert calls == [
        ("write_json", "gs://bucket/manifest.json", {"a": 1}, {"overwrite": True}),
        ("write_parquet", "gs://bucket/data.parquet", frame, {"overwrite": False}),
    ]


@pytest.mark.parametrize(
    ("operation", "message"),
    [
        (lambda: eval_datasets.read_manifest("gs://bucket/manifest.json"), "read eval manifest"),
        (
            lambda: eval_datasets.write_manifest("gs://bucket/manifest.json", {}),
            "write eval manifest",
        ),
        (lambda: eval_datasets.read_frame("gs://bucket/data.parquet"), "read eval frame"),
        (
            lambda: eval_datasets.write_frame("gs://bucket/data.parquet", pd.DataFrame()),
            "write eval frame",
        ),
    ],
)
def test_eval_dataset_storage_seam_wraps_read_write_failures(
    monkeypatch,
    operation,
    message,
):
    def fail(*args, **kwargs):
        raise RuntimeError("backend failed")

    monkeypatch.setattr(eval_datasets.gcs, "read_json", fail)
    monkeypatch.setattr(eval_datasets.gcs, "write_json", fail)
    monkeypatch.setattr(eval_datasets.gcs, "read_parquet", fail)
    monkeypatch.setattr(eval_datasets.gcs, "write_parquet", fail)

    with pytest.raises(StorageError, match=message):
        operation()
