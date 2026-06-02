from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pandas as pd
import pytest

from automl.data import ComponentHashes, Dataset, FeatureRegistry, LoadedDataset, pipeline

pytestmark = pytest.mark.unit


def _dataset() -> Dataset:
    return Dataset(
        id="v1_manifest",
        identity_hash="sha256:identity",
        component_hashes=ComponentHashes(
            source_identity="sha256:source",
            feature_registry="sha256:registry",
            data_content="sha256:data",
            schema="sha256:schema",
        ),
        gcs_bucket="automl-test-bucket",
        gcs_prefix="",
        project_name="demo",
        created_at="2026-05-27T00:00:00+00:00",
        source_identity={"kind": "local_csv"},
        n_rows=1,
        n_columns=2,
        target_column="target",
        split_id_col="SPLITID",
        hash_key=("row_id",),
    )


def test_materialize_can_return_dataset_manifest_without_rows(monkeypatch):
    active = SimpleNamespace(active_experiment_id="exp-1")
    dataset = _dataset()
    loaded = LoadedDataset(
        dataset=dataset,
        df=pd.DataFrame({"secret_row_value": ["classified"], "target": [1]}),
        registry=FeatureRegistry(),
    )

    monkeypatch.setattr(pipeline, "_session", lambda session: session)
    monkeypatch.setattr(pipeline.mlflow_client, "bound_for", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(
        pipeline,
        "_materialize_bound",
        lambda *, active, refresh_source: loaded,
    )

    assert pipeline.materialize(session=active, include_rows=False) == dataset
    assert pipeline.materialize(session=active, include_rows=True) == loaded
