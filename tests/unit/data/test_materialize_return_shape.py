from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pandas as pd
import pytest

from automl.data import ComponentHashes, Dataset, FeatureRegistry, LoadedDataset, pipeline

pytestmark = pytest.mark.unit


def _dataset() -> Dataset:
    return Dataset(
        id="v1_record",
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
        split_pct_col="SPLIT_PCT",
        unique_key=("row_id",),
        split_group_key=("row_id",),
    )


def test_materialize_forwards_flags_and_returns_bound_result(monkeypatch):
    active = SimpleNamespace(active_experiment_id="exp-1")
    dataset = _dataset()
    loaded = LoadedDataset(
        dataset=dataset,
        df=pd.DataFrame({"secret_row_value": ["classified"], "target": [1]}),
        registry=FeatureRegistry(),
    )
    calls: list[dict] = []

    def fake_bound(*, active, refresh_data, refresh_source, include_rows):
        calls.append(
            {
                "refresh_data": refresh_data,
                "refresh_source": refresh_source,
                "include_rows": include_rows,
            }
        )
        return dataset if not include_rows else loaded

    monkeypatch.setattr(pipeline, "_session", lambda session: session)
    monkeypatch.setattr(pipeline.mlflow_client, "bound_for", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(pipeline, "_materialize_bound", fake_bound)

    assert pipeline.materialize(session=active, include_rows=False) == dataset
    assert pipeline.materialize(session=active, include_rows=True) == loaded
    # refresh_source implies refresh_data
    pipeline.materialize(session=active, refresh_source=True, include_rows=False)

    assert calls == [
        {"refresh_data": False, "refresh_source": False, "include_rows": False},
        {"refresh_data": False, "refresh_source": False, "include_rows": True},
        {"refresh_data": True, "refresh_source": True, "include_rows": False},
    ]
