from __future__ import annotations

from types import SimpleNamespace

import pytest

from automl.data import ComponentHashes, Dataset
from automl.data.selection import (
    activate_dataset,
    resolve_active_dataset,
    resolve_active_dataset_id,
)
from automl.errors import DataError

pytestmark = pytest.mark.unit


def _record(dataset_id: str) -> dict:
    return Dataset(
        id=dataset_id,
        identity_hash="sha256:identity",
        component_hashes=ComponentHashes(
            source_identity="sha256:source",
            feature_registry="sha256:registry",
            data_content="sha256:data",
            schema="sha256:schema",
        ),
        gcs_bucket="bucket",
        gcs_prefix="root",
        project_name="demo",
        created_at="2026-06-05T00:00:00+00:00",
        source_identity={"kind": "local_csv"},
        n_rows=2,
        n_columns=3,
        target_column="target",
        split_pct_col="SPLIT_PCT",
        unique_key=("row_id",),
        split_group_key=("row_id",),
        record_uri=f"runs:/overview/datasets/{dataset_id}/dataset.json",
    ).to_dict() | {"record_uri": f"runs:/overview/datasets/{dataset_id}/dataset.json"}


def _session():
    return SimpleNamespace(active_experiment_id="exp-1")


def test_activate_dataset_validates_record_and_writes_tag_and_artifact(monkeypatch):
    calls = []

    monkeypatch.setattr(
        "automl.data.selection.mlflow_client.bound_for",
        lambda *args, **kwargs: _NullContext(),
    )
    monkeypatch.setattr(
        "automl.data.selection.experiment_artifacts.read_dataset_record",
        lambda dataset_id, experiment_id=None: (
            _record(dataset_id) if dataset_id == "v2_good" else None
        ),
    )
    monkeypatch.setattr(
        "automl.data.selection.mlflow_experiment.set_active_dataset",
        lambda dataset_id, experiment_id=None: calls.append(("tag", dataset_id, experiment_id)),
    )
    monkeypatch.setattr(
        "automl.data.selection.experiment_artifacts.write_active_dataset_pointer",
        lambda dataset_id, experiment_id=None: calls.append(
            ("artifact", dataset_id, experiment_id)
        ),
    )

    dataset = activate_dataset("v2_good", session=_session())

    assert dataset.id == "v2_good"
    assert calls == [
        ("tag", "v2_good", "exp-1"),
        ("artifact", "v2_good", "exp-1"),
    ]


def test_activate_dataset_rejects_missing_record(monkeypatch):
    monkeypatch.setattr(
        "automl.data.selection.mlflow_client.bound_for",
        lambda *args, **kwargs: _NullContext(),
    )
    monkeypatch.setattr(
        "automl.data.selection.experiment_artifacts.read_dataset_record",
        lambda dataset_id, experiment_id=None: None,
    )

    with pytest.raises(KeyError, match="dataset 'missing' not found"):
        activate_dataset("missing", session=_session())


def test_resolve_active_dataset_requires_matching_tag_artifact_and_record(monkeypatch):
    monkeypatch.setattr(
        "automl.data.selection.mlflow_client.bound_for",
        lambda *args, **kwargs: _NullContext(),
    )
    monkeypatch.setattr(
        "automl.data.selection.mlflow_experiment.get_active_dataset",
        lambda experiment_id=None: "v2_good",
    )
    monkeypatch.setattr(
        "automl.data.selection.experiment_artifacts.read_active_dataset_pointer",
        lambda experiment_id=None: {"schema_version": 1, "active_dataset_id": "v2_good"},
    )
    monkeypatch.setattr(
        "automl.data.selection.experiment_artifacts.read_dataset_record",
        lambda dataset_id, experiment_id=None: _record(dataset_id),
    )

    assert resolve_active_dataset_id(session=_session()) == "v2_good"
    assert resolve_active_dataset(session=_session()).id == "v2_good"


def test_resolve_active_dataset_rejects_missing_tag(monkeypatch):
    monkeypatch.setattr(
        "automl.data.selection.mlflow_client.bound_for",
        lambda *args, **kwargs: _NullContext(),
    )
    monkeypatch.setattr(
        "automl.data.selection.mlflow_experiment.get_active_dataset",
        lambda experiment_id=None: None,
    )
    monkeypatch.setattr(
        "automl.data.selection.experiment_artifacts.read_active_dataset_pointer",
        lambda experiment_id=None: None,
    )

    with pytest.raises(DataError, match="active dataset pointer is not set"):
        resolve_active_dataset(session=_session())


def test_resolve_active_dataset_rejects_missing_artifact(monkeypatch):
    monkeypatch.setattr(
        "automl.data.selection.mlflow_client.bound_for",
        lambda *args, **kwargs: _NullContext(),
    )
    monkeypatch.setattr(
        "automl.data.selection.mlflow_experiment.get_active_dataset",
        lambda experiment_id=None: "v2_good",
    )
    monkeypatch.setattr(
        "automl.data.selection.experiment_artifacts.read_active_dataset_pointer",
        lambda experiment_id=None: None,
    )

    with pytest.raises(DataError, match="active dataset pointer artifact is missing"):
        resolve_active_dataset(session=_session())


def test_resolve_active_dataset_rejects_tag_artifact_mismatch(monkeypatch):
    monkeypatch.setattr(
        "automl.data.selection.mlflow_client.bound_for",
        lambda *args, **kwargs: _NullContext(),
    )
    monkeypatch.setattr(
        "automl.data.selection.mlflow_experiment.get_active_dataset",
        lambda experiment_id=None: "v2_good",
    )
    monkeypatch.setattr(
        "automl.data.selection.experiment_artifacts.read_active_dataset_pointer",
        lambda experiment_id=None: {"schema_version": 1, "active_dataset_id": "v3_bad"},
    )

    with pytest.raises(DataError, match="active dataset pointer mismatch"):
        resolve_active_dataset(session=_session())


def test_resolve_active_dataset_rejects_missing_pointed_record(monkeypatch):
    monkeypatch.setattr(
        "automl.data.selection.mlflow_client.bound_for",
        lambda *args, **kwargs: _NullContext(),
    )
    monkeypatch.setattr(
        "automl.data.selection.mlflow_experiment.get_active_dataset",
        lambda experiment_id=None: "v2_good",
    )
    monkeypatch.setattr(
        "automl.data.selection.experiment_artifacts.read_active_dataset_pointer",
        lambda experiment_id=None: {"schema_version": 1, "active_dataset_id": "v2_good"},
    )
    monkeypatch.setattr(
        "automl.data.selection.experiment_artifacts.read_dataset_record",
        lambda dataset_id, experiment_id=None: None,
    )

    with pytest.raises(DataError, match="points at missing dataset"):
        resolve_active_dataset(session=_session())


class _NullContext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, traceback):
        return False
