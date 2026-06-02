from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from automl.eval import Auc, EvalSpec, LogLoss, ThresholdSweep, evaluate, prepare_eval_dataset
from automl.mlflow import client, experiment, tags, trial
from automl.mlflow.trial import artifacts
from automl.project import (
    BinaryClassification,
    ModelRoute,
    ModelsConfig,
    ProjectConfig,
    RunConfig,
    Session,
    Splits,
)

pytestmark = pytest.mark.integration


class ScoreModel:
    def predict(self, context, model_input):
        del context
        return model_input["score"].to_numpy()


def _session(tmp_path: Path, spec: EvalSpec | None = None) -> Session:
    route = ModelRoute("sonnet", "medium")
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
                models=ModelsConfig(manager=route, proposer=route, coder=route),
                per_trial_seconds=120,
            ),
            gcs_bucket="automl-test-bucket",
            gcs_prefix="automl-root",
            mlflow_tracking_uri=(tmp_path / "mlruns").as_uri(),
        )
    )


def _fake_gcs(monkeypatch):
    json_store: dict[str, dict] = {}
    bytes_store: dict[str, bytes] = {}
    parquet_store: dict[str, pd.DataFrame] = {}

    monkeypatch.setattr(
        "automl.utils.io.gcs.write_json",
        lambda uri, payload, **kwargs: json_store.__setitem__(uri, payload),
    )
    monkeypatch.setattr("automl.utils.io.gcs.read_json", lambda uri, **kwargs: json_store[uri])
    monkeypatch.setattr(
        "automl.utils.io.gcs.write_bytes",
        lambda uri, payload, **kwargs: bytes_store.__setitem__(uri, payload),
    )
    monkeypatch.setattr("automl.utils.io.gcs.read_bytes", lambda uri, **kwargs: bytes_store[uri])
    monkeypatch.setattr(
        "automl.utils.io.gcs.write_parquet",
        lambda uri, df, **kwargs: parquet_store.__setitem__(uri, df.copy()),
    )
    monkeypatch.setattr(
        "automl.utils.io.gcs.read_parquet",
        lambda uri, **kwargs: parquet_store[uri].copy(),
    )
    monkeypatch.setattr(
        "automl.utils.io.gcs.blob_exists",
        lambda uri, **kwargs: uri in json_store or uri in bytes_store or uri in parquet_store,
    )
    monkeypatch.setattr("automl.utils.io.gcs.list_prefixes", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        "automl.mlflow.experiment.eval_datasets.write_manifest",
        lambda uri, payload, **kwargs: json_store.__setitem__(uri, payload),
    )
    monkeypatch.setattr(
        "automl.mlflow.experiment.eval_datasets.read_manifest",
        lambda uri, **kwargs: json_store[uri],
    )
    monkeypatch.setattr(
        "automl.mlflow.experiment.eval_datasets.write_frame",
        lambda uri, df, **kwargs: parquet_store.__setitem__(uri, df.copy()),
    )
    monkeypatch.setattr(
        "automl.mlflow.experiment.eval_datasets.read_frame",
        lambda uri, **kwargs: parquet_store[uri].copy(),
    )
    monkeypatch.setattr(
        "automl.mlflow.experiment.eval_datasets.blob_exists",
        lambda uri, **kwargs: uri in json_store or uri in bytes_store or uri in parquet_store,
    )
    monkeypatch.setattr(
        "automl.mlflow.experiment.eval_datasets.list_prefixes",
        lambda *args, **kwargs: [],
    )
    return json_store, bytes_store, parquet_store


def test_evaluate_persists_predictions_index_metrics_and_uses_cache(tmp_path, monkeypatch):
    spec = EvalSpec(
        primary=Auc(),
        metrics=[-LogLoss(), ThresholdSweep(thresholds=[0.3, 0.5, 0.7])],
    )
    active = _session(tmp_path, spec)
    _fake_gcs(monkeypatch)
    client.bind(
        tracking_uri=active.config.mlflow_tracking_uri,
        bucket=active.config.gcs_bucket,
        gcs_prefix=active.config.gcs_prefix,
        project_name=active.project_name,
        experiment_id=active.active_experiment_id,
    )
    experiment.ensure()
    frame = pd.DataFrame(
        {"row_id": [1, 2, 3, 4], "target": [0, 0, 1, 1], "score": [0.1, 0.2, 0.8, 0.9]}
    )
    eval_dataset, _ = prepare_eval_dataset(
        session=active,
        kind="external",
        frame=frame,
        target_col="target",
        hash_key=("row_id",),
    )

    with trial.active(slug="score_model", strategy="baseline") as run_id:
        artifacts.write_model(run_id, ScoreModel())
        result = evaluate(
            session=active,
            model_run_id=run_id,
            eval_dataset_id=eval_dataset.id,
            label="holdout",
            set_as_primary_label=True,
            _model=ScoreModel(),
            overwrite=False,
        )
        cached = evaluate(
            session=active,
            model_run_id=run_id,
            eval_dataset_id=eval_dataset.id,
            label="holdout",
            set_as_primary_label=True,
            _model=ScoreModel(),
            overwrite=False,
        )
        loaded_model_result = evaluate(
            session=active,
            model_run_id=run_id,
            eval_dataset_id=eval_dataset.id,
            label="holdout_model_load",
            _model=None,
            overwrite=False,
        )
        secondary = evaluate(
            session=active,
            model_run_id=run_id,
            eval_dataset_id=eval_dataset.id,
            label="secondary",
            _model=ScoreModel(),
            overwrite=False,
        )
        secondary_cached_primary = evaluate(
            session=active,
            model_run_id=run_id,
            eval_dataset_id=eval_dataset.id,
            label="secondary",
            set_as_primary_label=True,
            _model=ScoreModel(),
            overwrite=False,
        )
        secondary_index = artifacts.load_eval_index(run_id)
        overwritten = evaluate(
            session=active,
            model_run_id=run_id,
            eval_dataset_id=eval_dataset.id,
            label="holdout",
            set_as_primary_label=True,
            _model=ScoreModel(),
            overwrite=True,
        )

        run = client.raw().get_run(run_id)

    assert result.cached is False
    assert cached.cached is True
    assert overwritten.cached is False
    assert loaded_model_result.label == "holdout_model_load"
    assert secondary.cached is False
    assert secondary_cached_primary.cached is True
    assert secondary_index.primary_label == "secondary"
    assert run.data.metrics["eval.holdout.auc"] == pytest.approx(1.0)
    assert run.data.metrics["eval.holdout.negative_log_loss"] == pytest.approx(-0.164252033486018)
    assert "eval.holdout.threshold_sweep" not in run.data.metrics
    assert "auc" not in run.data.metrics
    assert run.data.tags[tags.EVAL_PRIMARY_LABEL] == "holdout"
    assert artifacts.list_eval(run_id) == [
        ("holdout", eval_dataset.id),
        ("holdout_model_load", eval_dataset.id),
        ("secondary", eval_dataset.id),
    ]
    assert artifacts.load_eval(run_id, "holdout").metrics[2]["name"] == "threshold_sweep"
    assert artifacts.load_predictions(run_id, "holdout").frame["y_pred"].tolist() == [
        0.1,
        0.2,
        0.8,
        0.9,
    ]
    index = artifacts.load_eval_index(run_id)
    assert index.primary_label == "holdout"
    assert [entry.label for entry in index.evaluations] == [
        "holdout",
        "holdout_model_load",
        "secondary",
    ]
