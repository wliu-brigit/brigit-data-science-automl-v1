from pathlib import Path

import pandas as pd
import pytest

from automl.eval import Auc, EvalSpec, Metric, prepare_eval_dataset
from automl.eval import _load as eval_load
from automl.eval import prepare as eval_prepare
from automl.project import (
    BinaryClassification,
    ModelRoute,
    ModelsConfig,
    ProjectConfig,
    RunConfig,
    Session,
    Splits,
    Where,
)

pytestmark = pytest.mark.integration


class WeightedMeanScore(Metric):
    name = "weighted_mean_score"
    required_columns = ("risk_weight",)
    required_augmentations = ("risk_weight",)

    def compute(self, df, y_pred, target_col):
        del target_col
        return float((df["risk_weight"] * y_pred).mean())


def _session(tmp_path: Path) -> Session:
    route = ModelRoute("sonnet", "medium")
    return Session(
        config=ProjectConfig(
            project_name="demo",
            repo_root=tmp_path,
            project_dir=tmp_path / "projects" / "demo",
            config_path=tmp_path / "projects" / "demo" / "config.py",
            task=BinaryClassification(target="target"),
            eval_spec=EvalSpec(primary=Auc()),
            run_config=RunConfig(
                experiment_id="baseline",
                splits=Splits({"train": Where("SPLIT_PCT") < 50, "test": Where("SPLIT_PCT") >= 50}),
                models=ModelsConfig(manager=route, proposer=route, coder=route),
                per_trial_seconds=120,
            ),
            gcs_bucket="bucket",
            gcs_prefix="root",
        )
    )


def _fake_gcs(monkeypatch):
    json_store: dict[str, dict] = {}
    parquet_store: dict[str, pd.DataFrame] = {}

    def write_json(uri, payload, **kwargs):
        del kwargs
        json_store[uri] = payload

    def write_parquet(uri, df, **kwargs):
        del kwargs
        parquet_store[uri] = df.copy()

    def blob_exists(uri, **kwargs):
        del kwargs
        return uri in json_store or uri in parquet_store

    def list_prefixes(uri_or_bucket, prefix=None):
        if prefix is None:
            base = uri_or_bucket.rstrip("/") + "/"
        else:
            base = f"gs://{uri_or_bucket}/{prefix.strip('/')}/"
        prefixes = set()
        for uri in [*json_store, *parquet_store]:
            if uri.startswith(base):
                rest = uri.removeprefix(base)
                head = rest.split("/", 1)[0]
                if head:
                    prefixes.add(base + head + "/")
        return sorted(prefixes)

    monkeypatch.setattr("automl.mlflow.experiment.eval_datasets.write_record", write_json)
    monkeypatch.setattr(
        "automl.mlflow.experiment.eval_datasets.read_record",
        lambda uri, **kwargs: json_store[uri],
    )
    monkeypatch.setattr("automl.mlflow.experiment.eval_datasets.write_frame", write_parquet)
    monkeypatch.setattr(
        "automl.mlflow.experiment.eval_datasets.read_frame",
        lambda uri, **kwargs: parquet_store[uri].copy(),
    )
    monkeypatch.setattr("automl.mlflow.experiment.eval_datasets.blob_exists", blob_exists)
    monkeypatch.setattr("automl.mlflow.experiment.eval_datasets.list_prefixes", list_prefixes)
    return json_store, parquet_store


def test_prepare_load_and_join_augmentation(tmp_path, monkeypatch):
    active = _session(tmp_path)
    _fake_gcs(monkeypatch)
    frame = pd.DataFrame({"row_id": [1, 2], "target": [0, 1], "score": [0.2, 0.8]})
    eval_dataset, _ = prepare_eval_dataset(
        session=active,
        kind="external",
        frame=frame,
        target_col="target",
        unique_key=("row_id",),
    )
    augmentation_frame = pd.DataFrame({"row_id": [1, 2], "risk_weight": [1.0, 2.0]})

    augmentation, cached = eval_prepare.prepare_eval_augmentation(
        session=active,
        eval_dataset_id=eval_dataset.id,
        frame=augmentation_frame,
        name="risk_weight",
    )
    second, second_cached = eval_prepare.prepare_eval_augmentation(
        session=active,
        eval_dataset_id=eval_dataset.id,
        frame=augmentation_frame,
        name="risk_weight",
    )
    frames, used = eval_load.load_eval_augmentations(
        eval_dataset.id,
        names=("risk_weight",),
        session=active,
    )

    report = EvalSpec(primary=Auc(), metrics=[WeightedMeanScore()]).evaluate(
        frame,
        pd.Series([0.2, 0.8]),
        "target",
        augmentation_frames=frames,
        unique_key=("row_id",),
    )

    assert cached is False
    assert second == augmentation
    assert second_cached is True
    assert frames["risk_weight"].equals(augmentation_frame)
    assert used == (
        {
            "name": "risk_weight",
            "hash8": augmentation.hash8,
            "data_uri": augmentation.data_gcs_uri,
            "record_uri": augmentation.record_gcs_uri,
        },
    )
    assert report["metrics"][1] == {
        "name": "weighted_mean_score",
        "value": pytest.approx(0.9),
        "augmentations": ["risk_weight"],
    }


def test_load_eval_augmentations_names_missing_published_items(tmp_path, monkeypatch):
    active = _session(tmp_path)
    _fake_gcs(monkeypatch)
    frame = pd.DataFrame({"row_id": [1], "target": [0]})
    eval_dataset, _ = prepare_eval_dataset(
        session=active,
        kind="external",
        frame=frame,
        target_col="target",
        unique_key=("row_id",),
    )

    with pytest.raises(ValueError, match="augmentations not published"):
        eval_load.load_eval_augmentations(eval_dataset.id, names=("risk_weight",), session=active)


def test_load_eval_augmentations_rejects_payload_that_no_longer_matches_record(
    tmp_path, monkeypatch
):
    active = _session(tmp_path)
    _json_store, parquet_store = _fake_gcs(monkeypatch)
    frame = pd.DataFrame({"row_id": [1, 2], "target": [0, 1], "score": [0.2, 0.8]})
    eval_dataset, _ = prepare_eval_dataset(
        session=active,
        kind="external",
        frame=frame,
        target_col="target",
        unique_key=("row_id",),
    )
    augmentation, _ = eval_prepare.prepare_eval_augmentation(
        session=active,
        eval_dataset_id=eval_dataset.id,
        frame=pd.DataFrame({"row_id": [1, 2], "risk_weight": [1.0, 2.0]}),
        name="risk_weight",
    )
    parquet_store[augmentation.data_gcs_uri] = pd.DataFrame(
        {"row_id": [1, 2], "risk_weight": [1.0, 3.0]}
    )

    with pytest.raises(ValueError, match="content_hash"):
        eval_load.load_eval_augmentations(eval_dataset.id, names=("risk_weight",), session=active)
