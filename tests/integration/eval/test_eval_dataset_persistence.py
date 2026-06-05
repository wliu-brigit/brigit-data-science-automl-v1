from pathlib import Path

import pandas as pd
import pytest

from automl.data import ComponentHashes, Dataset, DatasetIndex, FeatureRegistry, LoadedSlice
from automl.eval import load_eval_dataset, prepare_eval_dataset
from automl.project import (
    BinaryClassification,
    ModelRoute,
    ModelsConfig,
    ProjectConfig,
    RunConfig,
    Session,
    Predicate,
    Splits,
    Where,
)

pytestmark = pytest.mark.integration


def _models() -> ModelsConfig:
    route = ModelRoute("sonnet", "medium")
    return ModelsConfig(manager=route, proposer=route, coder=route)


def _session(tmp_path: Path) -> Session:
    return Session(
        config=ProjectConfig(
            project_name="demo",
            repo_root=tmp_path,
            project_dir=tmp_path / "projects" / "demo",
            config_path=tmp_path / "projects" / "demo" / "config.py",
            task=BinaryClassification(target="target"),
            run_config=RunConfig(
                experiment_id="baseline",
                splits=Splits({"train": Where("SPLIT_PCT") < 50, "test": Where("SPLIT_PCT") >= 50}),
                models=_models(),
                per_trial_seconds=120,
            ),
            gcs_bucket="bucket",
            gcs_prefix="root",
        )
    )


def _dataset() -> Dataset:
    return Dataset(
        id="dataset-v1",
        identity_hash="sha256:identity",
        component_hashes=ComponentHashes("source", "registry", "data", "schema"),
        gcs_bucket="bucket",
        gcs_prefix="root",
        project_name="demo",
        created_at="2026-05-27T00:00:00+00:00",
        source_identity={},
        n_rows=4,
        n_columns=4,
        target_column="target",
        split_pct_col="SPLIT_PCT",
        unique_key=("row_id",),
        split_group_key=("row_id",),
    )


def _loaded_slice() -> LoadedSlice:
    frame = pd.DataFrame(
        {
            "row_id": [1, 2],
            "target": [0, 1],
            "score": [0.1, 0.9],
            "SPLIT_PCT": [51, 52],
        }
    )
    return LoadedSlice(
        dataset=_dataset(),
        df=frame,
        registry=FeatureRegistry().build_from_df(frame, target_column="target"),
        split_name=None,
        predicate=Where("SPLIT_PCT") >= 50,
    )


def _fake_gcs(monkeypatch):
    json_store: dict[str, dict] = {}
    parquet_store: dict[str, pd.DataFrame] = {}

    monkeypatch.setattr(
        "automl.mlflow.experiment.eval_datasets.write_record",
        lambda uri, payload, **kwargs: json_store.setdefault(uri, payload),
    )
    monkeypatch.setattr(
        "automl.mlflow.experiment.eval_datasets.read_record",
        lambda uri, **kwargs: json_store[uri],
    )
    monkeypatch.setattr(
        "automl.mlflow.experiment.eval_datasets.write_frame",
        lambda uri, df, **kwargs: parquet_store.setdefault(uri, df.copy()),
    )
    monkeypatch.setattr(
        "automl.mlflow.experiment.eval_datasets.read_frame",
        lambda uri, **kwargs: parquet_store[uri].copy(),
    )
    monkeypatch.setattr(
        "automl.mlflow.experiment.eval_datasets.blob_exists",
        lambda uri, **kwargs: uri in json_store or uri in parquet_store,
    )
    return json_store, parquet_store


def test_split_view_prepare_writes_record_and_loads_lazily(tmp_path, monkeypatch):
    active = _session(tmp_path)
    dataset = _dataset()
    json_store, parquet_store = _fake_gcs(monkeypatch)
    load_calls = []

    monkeypatch.setattr(
        "automl.eval.prepare.data.list_datasets",
        lambda *, session=None: DatasetIndex((dataset,), active_dataset_id=dataset.id),
    )

    def fake_load_dataset_by_id(dataset_id, *, split_name=None, predicate=None, session=None):
        load_calls.append((dataset_id, split_name, predicate, session))
        return _loaded_slice()

    monkeypatch.setattr("automl.eval._load.data.load_dataset_by_id", fake_load_dataset_by_id)

    eval_dataset, cached = prepare_eval_dataset(
        session=active,
        dataset_id=dataset.id,
        split="test",
    )
    loaded = load_eval_dataset(eval_dataset.id, session=active)

    assert cached is False
    assert len(json_store) == 1
    assert parquet_store == {}
    assert loaded.df["row_id"].tolist() == [1, 2]
    assert load_calls == [(dataset.id, None, Predicate.from_dict((Where("SPLIT_PCT") >= 50).to_dict()), active)]


def test_external_prepare_writes_frame_and_loads_round_trip(tmp_path, monkeypatch):
    active = _session(tmp_path)
    json_store, parquet_store = _fake_gcs(monkeypatch)
    frame = pd.DataFrame({"row_id": [1, 2], "target": [0, 1], "score": [0.1, 0.9]})

    eval_dataset, cached = prepare_eval_dataset(
        session=active,
        kind="external",
        frame=frame,
        target_col="target",
        unique_key=("row_id",),
        provenance={"source": "unit"},
    )
    second, second_cached = prepare_eval_dataset(
        session=active,
        kind="external",
        frame=frame,
        target_col="target",
        unique_key=("row_id",),
        provenance={"source": "unit"},
    )
    loaded = load_eval_dataset(eval_dataset.id, session=active)

    assert cached is False
    assert second == eval_dataset
    assert second_cached is True
    assert eval_dataset.record_gcs_uri in json_store
    assert eval_dataset.data_gcs_uri in parquet_store
    assert loaded.df.equals(frame)
    assert loaded.row_ids.equals(frame[["row_id"]])


def test_external_load_rejects_payload_that_no_longer_matches_record(tmp_path, monkeypatch):
    active = _session(tmp_path)
    _json_store, parquet_store = _fake_gcs(monkeypatch)
    frame = pd.DataFrame({"row_id": [1, 2], "target": [0, 1], "score": [0.1, 0.9]})

    eval_dataset, _cached = prepare_eval_dataset(
        session=active,
        kind="external",
        frame=frame,
        target_col="target",
        unique_key=("row_id",),
        provenance={"source": "unit"},
    )
    parquet_store[eval_dataset.data_gcs_uri] = frame.assign(score=[0.2, 0.9])

    with pytest.raises(ValueError, match="content_hash"):
        load_eval_dataset(eval_dataset.id, session=active)
