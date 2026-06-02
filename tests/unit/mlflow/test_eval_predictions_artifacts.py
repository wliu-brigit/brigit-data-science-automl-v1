from __future__ import annotations

import cloudpickle
import pandas as pd
import pytest

from automl.eval import EvalIndex, EvalIndexEntry, EvalResult, Predictions
from automl.mlflow import client, experiment, tags, trial
from automl.mlflow.trial import artifacts

pytestmark = pytest.mark.unit


@pytest.fixture
def active_run_with_fake_gcs(tmp_path, monkeypatch):
    client.clear()
    client.bind(
        tracking_uri=(tmp_path / "mlruns").as_uri(),
        bucket="automl-test-bucket",
        gcs_prefix="automl-root",
        project_name="home_credit",
        experiment_id="baseline",
    )
    experiment.ensure()
    json_store: dict[str, dict] = {}
    bytes_store: dict[str, bytes] = {}

    def fake_write_json(uri: str, payload: dict, **kwargs) -> None:
        del kwargs
        json_store[uri] = payload

    def fake_read_json(uri: str, **kwargs) -> dict:
        del kwargs
        return json_store[uri]

    def fake_write_bytes(uri: str, payload: bytes, **kwargs) -> None:
        del kwargs
        bytes_store[uri] = payload

    def fake_read_bytes(uri: str, **kwargs) -> bytes:
        del kwargs
        return bytes_store[uri]

    monkeypatch.setattr("automl.utils.io.gcs.write_json", fake_write_json)
    monkeypatch.setattr("automl.utils.io.gcs.read_json", fake_read_json)
    monkeypatch.setattr("automl.utils.io.gcs.write_bytes", fake_write_bytes)
    monkeypatch.setattr("automl.utils.io.gcs.read_bytes", fake_read_bytes)

    with trial.active(slug="baseline_lr", strategy="baseline") as run_id:
        yield run_id, json_store, bytes_store

    client.clear()


def _eval_result(label: str = "holdout") -> EvalResult:
    return EvalResult(
        label=label,
        eval_dataset_id="eval-v1",
        eval_dataset_kind="external",
        predictions_uri=f"gs://bucket/eval/{label}/predictions.parquet",
        predictions_manifest_uri=f"gs://bucket/eval/{label}/predictions.json",
        augmentations_used=(),
        primary="auc",
        metrics=({"name": "auc", "value": 0.71, "augmentations": []},),
        computed_at="2026-05-27T00:00:00+00:00",
    )


def test_write_load_and_list_eval_artifacts(active_run_with_fake_gcs):
    run_id, json_store, _ = active_run_with_fake_gcs

    holdout = artifacts.write_eval(run_id, "holdout", _eval_result("holdout"))
    train = artifacts.write_eval(run_id, "train", _eval_result("train"))

    assert holdout.path == "eval/holdout/report.json"
    assert json_store == {}
    assert artifacts.load_eval(run_id, "holdout") == _eval_result("holdout")
    assert artifacts.list_eval(run_id) == [("holdout", "eval-v1"), ("train", "eval-v1")]

    run_tags = client.raw().get_run(run_id).data.tags
    assert run_tags[tags.eval_uri("holdout")] == "eval/holdout/report.json"
    assert run_tags[tags.eval_dataset_id("train")] == "eval-v1"
    assert train.uri.endswith("eval/train/report.json")


def test_write_and_load_eval_index(active_run_with_fake_gcs):
    run_id, json_store, _ = active_run_with_fake_gcs
    index = EvalIndex(
        primary_label="holdout",
        evaluations=(
            EvalIndexEntry(
                label="holdout",
                eval_dataset_id="eval-v1",
                kind="external",
                report_path="eval/holdout/report.json",
                eval_dataset_manifest_uri="gs://bucket/eval/datasets/eval-v1/manifest.json",
                predictions_uri="gs://bucket/eval/holdout/predictions.parquet",
                predictions_manifest_uri="gs://bucket/eval/holdout/predictions.json",
                augmentations_used=(),
                computed_at="2026-05-27T00:00:00+00:00",
            ),
        ),
    )

    ref = artifacts.write_eval_index(run_id, index)

    assert ref.path == "eval/manifest.json"
    assert json_store == {}
    assert artifacts.load_eval_index(run_id) == index


def test_load_eval_index_returns_empty_when_absent(active_run_with_fake_gcs):
    run_id, _, _ = active_run_with_fake_gcs

    assert artifacts.load_eval_index(run_id) == EvalIndex(primary_label=None, evaluations=())


def test_write_load_and_list_predictions(active_run_with_fake_gcs):
    run_id, json_store, bytes_store = active_run_with_fake_gcs
    predictions = Predictions(
        trial_run_id=run_id,
        eval_dataset_id="eval-v1",
        eval_dataset_kind="external",
        label="holdout",
        hash_key=("row_id",),
        frame=pd.DataFrame({"row_id": [1, 2], "y_pred": [0.2, 0.8]}),
        augmentations_used=(),
        written_at="2026-05-27T00:00:00+00:00",
    )

    ref = artifacts.write_predictions(run_id, "holdout", predictions)
    restored = artifacts.load_predictions(run_id, "holdout")

    assert ref.path == "eval/holdout/predictions.parquet"
    assert ref.manifest_path == "eval/holdout/predictions.json"
    assert json_store == {}
    assert bytes_store[ref.uri]
    assert restored.frame.equals(predictions.frame)
    assert restored.manifest_dict() == predictions.manifest_dict()
    assert artifacts.list_predictions(run_id) == ["holdout"]

    run_tags = client.raw().get_run(run_id).data.tags
    assert run_tags[tags.eval_predictions_uri("holdout")] == ref.uri
    assert (
        run_tags[tags.eval_predictions_manifest_uri("holdout")] == "eval/holdout/predictions.json"
    )


def test_load_pickle_model_round_trips_from_mlflow(active_run_with_fake_gcs):
    run_id, json_store, bytes_store = active_run_with_fake_gcs
    payload = {"model": "plain-object"}

    artifacts.write_pickle_model(run_id, payload)

    assert json_store == {}
    assert bytes_store == {}
    assert artifacts.load_model(run_id) == payload


def test_eval_labels_reject_path_segments(active_run_with_fake_gcs):
    run_id, _, _ = active_run_with_fake_gcs

    for label in ("", ".", "..", "bad/path"):
        with pytest.raises(ValueError):
            artifacts.write_eval(run_id, label, _eval_result(label or "bad"))


def test_load_model_uses_cloudpickle_bytes(active_run_with_fake_gcs):
    run_id, _, bytes_store = active_run_with_fake_gcs
    artifacts.write_pickle_model(run_id, {"x": 1})
    uri = client.raw().get_run(run_id).data.tags[tags.MODEL_URI]

    assert bytes_store == {}
    local_path = client.raw().download_artifacts(run_id, uri.removeprefix(f"runs:/{run_id}/"))
    with open(local_path, "rb") as handle:
        assert cloudpickle.loads(handle.read()) == {"x": 1}
