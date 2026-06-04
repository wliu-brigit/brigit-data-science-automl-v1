import importlib
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from automl.data import ComponentHashes, Dataset, DatasetIndex, FeatureRegistry, LoadedSlice
from automl.errors import EvalError
from automl.eval import (
    Auc,
    EvalIndex,
    EvalResult,
    EvalSpec,
    evaluate,
    load_eval_dataset,
    prepare_eval_dataset,
)
from automl.eval.base import Metric
from automl.model import BaseModel
from automl.project import (
    BinaryClassification,
    ModelRoute,
    ModelsConfig,
    ProjectConfig,
    RunConfig,
    Session,
    Splits,
)

pytestmark = pytest.mark.unit


class NeedsAmount(Metric):
    name = "needs_amount"
    required_columns = ("amount",)

    def compute(self, df, y_pred, target_col):
        return float(df["amount"].mean() + y_pred.mean() + df[target_col].mean())


class ScoreModel(BaseModel):
    def fit(self, df_train, registry, seed=0):
        self.feature_registry = registry
        self.preprocessor = "identity"
        self.model = "score"
        self.name = "score"
        return self

    def transform(self, df):
        return df[["score"]].to_numpy()

    def _predict(self, X):
        return X[:, 0]


def _models() -> ModelsConfig:
    route = ModelRoute("sonnet", "medium")
    return ModelsConfig(manager=route, proposer=route, coder=route)


def _session(tmp_path: Path, spec: EvalSpec | None = None) -> Session:
    return Session(
        config=ProjectConfig(
            project_name="demo",
            repo_root=tmp_path,
            project_dir=tmp_path / "projects" / "demo",
            config_path=tmp_path / "projects" / "demo" / "config.py",
            task=BinaryClassification(target="target"),
            eval_spec=spec or EvalSpec(primary=Auc()),
            run_config=RunConfig(
                experiment_id="baseline",
                splits=Splits({"train": ((0, 50),), "test": ((50, 100),)}),
                models=_models(),
                per_trial_seconds=120,
            ),
            gcs_bucket="bucket",
            mlflow_tracking_uri="file:///tmp/mlruns",
        )
    )


def _loaded_slice() -> LoadedSlice:
    df = pd.DataFrame(
        {
            "row_id": [1, 2, 3, 4],
            "target": [0, 0, 1, 1],
            "score": [0.05, 0.4, 0.6, 0.95],
            "amount": [10.0, 20.0, 30.0, 40.0],
            "SPLIT_PCT": [51, 52, 53, 54],
        }
    )
    dataset = Dataset(
        id="dataset-1",
        identity_hash="sha256:identity",
        component_hashes=ComponentHashes("a", "b", "c", "d"),
        gcs_bucket="bucket",
        project_name="demo",
        created_at="2026-05-27T00:00:00+00:00",
        source_identity={},
        n_rows=len(df),
        n_columns=len(df.columns),
        target_column="target",
        split_pct_col="SPLIT_PCT",
        unique_key=("row_id",),
        split_group_key=("row_id",),
    )
    registry = FeatureRegistry().build_from_df(df, target_column="target")
    return LoadedSlice(
        dataset=dataset,
        df=df,
        registry=registry,
        split_name="test",
        split_ranges=((50, 100),),
    )


def _fake_eval_dataset_storage(monkeypatch):
    json_store: dict[str, dict] = {}

    monkeypatch.setattr(
        "automl.mlflow.experiment.eval_datasets.write_manifest",
        lambda uri, payload, **kwargs: json_store.__setitem__(uri, payload),
    )
    monkeypatch.setattr(
        "automl.mlflow.experiment.eval_datasets.read_manifest",
        lambda uri, **kwargs: json_store[uri],
    )
    monkeypatch.setattr(
        "automl.mlflow.experiment.eval_datasets.blob_exists",
        lambda uri, **kwargs: uri in json_store,
    )
    return json_store


def _patch_dataset_index(monkeypatch, dataset):
    monkeypatch.setattr(
        "automl.eval.prepare.data.list_datasets",
        lambda *, session=None: DatasetIndex((dataset,), active_dataset_id=dataset.id),
    )


def test_auc_computes_real_roc_auc():
    df = pd.DataFrame({"target": [0, 0, 1, 1]})
    score = Auc().compute(df, pd.Series([0.1, 0.2, 0.8, 0.9]), "target")

    assert score == pytest.approx(1.0)


def test_eval_spec_validates_columns_and_duplicate_metric_names():
    spec = EvalSpec(primary=Auc(), metrics=[NeedsAmount()])
    df = pd.DataFrame({"target": [0, 1], "amount": [10.0, 20.0]})

    assert spec.primary_name == "auc"
    assert spec.required_columns() == ("amount",)
    spec.validate_columns(df, "target")

    with pytest.raises(ValueError, match="duplicate metric"):
        EvalSpec(primary=Auc(), metrics=[Auc()])
    with pytest.raises(ValueError, match="missing required"):
        spec.validate_columns(pd.DataFrame({"target": [0, 1]}), "target")


def test_eval_result_round_trips_persisted_schema_without_runtime_fields():
    result = EvalResult(
        label="test",
        eval_dataset_id="ev_abc123",
        eval_dataset_kind="split_view",
        predictions_uri="gs://bucket/predictions.parquet",
        predictions_manifest_uri="gs://bucket/predictions.json",
        augmentations_used=("aug_a",),
        primary="auc",
        metrics=({"name": "auc", "value": 0.9, "augmentations": []},),
        computed_at="2026-05-27T00:00:00+00:00",
        cached=True,
    )

    payload = result.to_dict()
    restored = EvalResult.from_dict(payload)

    assert payload == {
        "schema_version": 1,
        "label": "test",
        "eval_dataset_id": "ev_abc123",
        "eval_dataset_kind": "split_view",
        "predictions_uri": "gs://bucket/predictions.parquet",
        "predictions_manifest_uri": "gs://bucket/predictions.json",
        "augmentations_used": ["aug_a"],
        "primary": "auc",
        "metrics": [{"name": "auc", "value": 0.9, "augmentations": []}],
        "computed_at": "2026-05-27T00:00:00+00:00",
    }
    assert restored == EvalResult(
        label="test",
        eval_dataset_id="ev_abc123",
        eval_dataset_kind="split_view",
        predictions_uri="gs://bucket/predictions.parquet",
        predictions_manifest_uri="gs://bucket/predictions.json",
        augmentations_used=("aug_a",),
        primary="auc",
        metrics=({"name": "auc", "value": 0.9, "augmentations": []},),
        computed_at="2026-05-27T00:00:00+00:00",
    )
    assert restored.cached is False


def test_prepare_eval_dataset_is_deterministic_and_loads_lazily(tmp_path, monkeypatch):
    active = _session(tmp_path)
    loaded = _loaded_slice()
    _fake_eval_dataset_storage(monkeypatch)
    _patch_dataset_index(monkeypatch, loaded.dataset)

    def explode(*args, **kwargs):
        raise AssertionError("prepare_eval_dataset must not load data")

    monkeypatch.setattr("automl.eval._load.data.load_dataset_by_id", explode)

    first, first_cached = prepare_eval_dataset(session=active, dataset_id="dataset-1", split="test")
    second, second_cached = prepare_eval_dataset(
        session=active, dataset_id="dataset-1", split="test"
    )

    assert first == second
    assert first_cached is False
    assert second_cached is True


def test_prepare_external_eval_dataset_partial_objects_raise_eval_error(tmp_path, monkeypatch):
    active = _session(tmp_path)
    frame = pd.DataFrame({"row_id": [1], "target": [0]})

    monkeypatch.setattr(
        "automl.mlflow.experiment.eval_datasets.blob_exists",
        lambda uri, **kwargs: uri.endswith("/manifest.json"),
    )

    with pytest.raises(EvalError, match="partial external eval dataset objects exist"):
        prepare_eval_dataset(
            session=active,
            kind="external",
            frame=frame,
            unique_key=("row_id",),
        )


def test_load_eval_dataset_delegates_to_data_load_dataset_by_id(tmp_path, monkeypatch):
    active = _session(tmp_path)
    loaded_slice = _loaded_slice()
    _fake_eval_dataset_storage(monkeypatch)
    _patch_dataset_index(monkeypatch, loaded_slice.dataset)
    eval_dataset, _ = prepare_eval_dataset(session=active, dataset_id="dataset-1", split="test")
    calls = []

    def fake_load_dataset_by_id(dataset_id, *, split_name=None, split_range=None, session=None):
        calls.append((dataset_id, split_name, split_range, session))
        return loaded_slice

    monkeypatch.setattr(
        "automl.eval._load.data.load_dataset_by_id",
        fake_load_dataset_by_id,
    )

    loaded = load_eval_dataset(eval_dataset.id, session=active)

    assert loaded.df["target"].tolist() == [0, 0, 1, 1]
    assert calls == [("dataset-1", None, ((50, 100),), active)]


def test_evaluate_loads_split_view_scores_injected_model_and_returns_result(tmp_path, monkeypatch):
    spec = EvalSpec(primary=Auc(), metrics=[NeedsAmount()])
    active = _session(tmp_path, spec)
    loaded = _loaded_slice()
    _fake_eval_dataset_storage(monkeypatch)
    _patch_dataset_index(monkeypatch, loaded.dataset)
    eval_dataset, _ = prepare_eval_dataset(session=active, dataset_id=loaded.id, split="test")
    model = ScoreModel().fit(loaded.df, loaded.registry)

    monkeypatch.setattr(
        "automl.eval._load.data.load_dataset_by_id",
        lambda *args, **kwargs: loaded,
    )
    evaluate_module = importlib.import_module("automl.eval.evaluate")
    monkeypatch.setattr(
        evaluate_module.artifacts,
        "load_eval",
        lambda *args, **kwargs: (_ for _ in ()).throw(KeyError("missing")),
    )
    monkeypatch.setattr(
        evaluate_module.artifacts,
        "load_predictions",
        lambda *args, **kwargs: (_ for _ in ()).throw(KeyError("missing")),
    )
    monkeypatch.setattr(
        evaluate_module.artifacts,
        "write_predictions",
        lambda *args, **kwargs: SimpleNamespace(uri="", manifest_uri=""),
    )
    monkeypatch.setattr(
        evaluate_module.artifacts,
        "write_eval",
        lambda *args, **kwargs: SimpleNamespace(path="eval/test/results.json"),
    )
    monkeypatch.setattr(
        evaluate_module.artifacts,
        "load_eval_index",
        lambda *args, **kwargs: EvalIndex(primary_label=None, evaluations=()),
    )
    monkeypatch.setattr(evaluate_module.artifacts, "write_eval_index", lambda *args, **kwargs: None)
    monkeypatch.setattr(evaluate_module.mlflow_trial, "log_metrics", lambda *args, **kwargs: None)
    monkeypatch.setattr(evaluate_module.mlflow_trial, "log_metric", lambda *args, **kwargs: None)
    monkeypatch.setattr(evaluate_module.mlflow_trial, "set_tag", lambda *args, **kwargs: None)

    result = evaluate(
        session=active,
        model_run_id="run-123",
        eval_dataset_id=eval_dataset.id,
        label="test",
        set_as_primary_label=True,
        _model=model,
        _model_feature_registry=loaded.registry,
    )

    assert isinstance(result, EvalResult)
    assert result.eval_dataset_id == eval_dataset.id
    assert result.eval_dataset_kind == "split_view"
    assert result.label == "test"
    assert result.primary == "auc"
    assert result.metrics[0]["name"] == "auc"
    assert result.metrics[0]["value"] == pytest.approx(1.0)
    assert result.predictions_uri == ""
    assert result.predictions_manifest_uri == ""
    assert result.augmentations_used == ()
    assert result.computed_at
    assert "eval_dataset_id" in result.to_dict()
