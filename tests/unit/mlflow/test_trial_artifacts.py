from __future__ import annotations

import importlib
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import cloudpickle
import pytest

from automl.mlflow import client, experiment, trial
from automl.mlflow.trial import artifacts
from automl.model import BaseModel

pytestmark = pytest.mark.unit


def test_model_artifact_import_does_not_emit_mlflow_type_hint_warning():
    sys.modules.pop("automl.mlflow.trial.artifacts.model", None)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        importlib.import_module("automl.mlflow.trial.artifacts.model")

    assert not [
        warning
        for warning in caught
        if "Add type hints to the `predict` method" in str(warning.message)
    ]


@dataclass(frozen=True)
class PayloadWithToDict:
    value: int

    def to_dict(self) -> dict[str, int]:
        return {"value": self.value}


class TinyModel(BaseModel):
    def fit(self, df_train, registry, seed: int = 0):
        del df_train, registry, seed
        return self

    def transform(self, df):
        return df

    def _predict(self, X):
        del X
        return [0.25]


@pytest.fixture
def active_run_with_gcs(tmp_path, monkeypatch):
    client.clear()
    client.bind(
        tracking_uri=(tmp_path / "mlruns").as_uri(),
        bucket="automl-test-bucket",
        gcs_prefix="automl-root",
        project_name="home_credit",
        experiment_id="baseline",
    )
    experiment.ensure()
    written: list[tuple[str, dict]] = []
    written_bytes: list[tuple[str, bytes]] = []

    def fake_write_json(uri: str, payload: dict, **kwargs) -> None:
        written.append((uri, payload))

    def fake_write_bytes(uri: str, payload: bytes, **kwargs) -> None:
        written_bytes.append((uri, payload))

    monkeypatch.setattr("automl.utils.io.gcs.write_json", fake_write_json)
    monkeypatch.setattr("automl.utils.io.gcs.write_bytes", fake_write_bytes)

    with trial.active(slug="baseline_lr", strategy="baseline") as run_id:
        yield run_id, written, written_bytes

    client.clear()


def _read_json_artifact(run_id: str, path: str) -> dict:
    local_path = client.raw().download_artifacts(run_id, path)
    import json

    with open(local_path, encoding="utf-8") as handle:
        return json.load(handle)


def _artifact_paths(run_id: str) -> set[str]:
    return {item.path for item in trial.list_artifacts(run_id)}


def test_write_trial_data_contract_logs_mlflow_artifact_and_tags_path(active_run_with_gcs):
    run_id, written, _ = active_run_with_gcs

    ref = artifacts.write_trial_data_contract(run_id, PayloadWithToDict(3))

    assert written == []
    assert ref.run_id == run_id
    assert ref.path == "data/contract.json"
    assert ref.uri == f"runs:/{run_id}/data/contract.json"
    assert _read_json_artifact(run_id, "data/contract.json") == {
        "schema_version": 1,
        "value": 3,
    }
    run = client.raw().get_run(run_id)
    assert run.data.tags["data.contract_artifact"] == "data/contract.json"


def test_write_eval_sets_label_tags_and_returns_ref(active_run_with_gcs):
    run_id, written, _ = active_run_with_gcs

    ref = artifacts.write_eval(
        run_id,
        label="holdout",
        payload={"eval_dataset_id": "eval-v1", "auc": 0.71},
    )

    assert written == []
    assert ref.run_id == run_id
    assert ref.label == "holdout"
    assert ref.path == "eval/holdout/report.json"
    assert _read_json_artifact(run_id, "eval/holdout/report.json") == {
        "schema_version": 1,
        "eval_dataset_id": "eval-v1",
        "auc": 0.71,
    }
    run = client.raw().get_run(run_id)
    assert run.data.tags["eval.holdout.report_artifact"] == "eval/holdout/report.json"
    assert run.data.tags["eval.holdout.dataset_id"] == "eval-v1"


def test_singleton_json_artifact_writers_log_to_mlflow(active_run_with_gcs):
    run_id, written, written_bytes = active_run_with_gcs

    artifacts.write_manifest(run_id, PayloadWithToDict(9))

    assert written_bytes == []
    assert written == []
    assert _read_json_artifact(run_id, "manifest.json") == {
        "schema_version": 1,
        "value": 9,
    }
    run = client.raw().get_run(run_id)
    assert run.data.tags["trial.manifest_artifact"] == "manifest.json"


def test_write_model_logs_pyfunc_model_as_logged_model(active_run_with_gcs):
    run_id, _, written_bytes = active_run_with_gcs

    ref = artifacts.write_model(run_id, TinyModel())

    assert ref.path == "model"
    assert ref.uri == f"runs:/{run_id}/model"
    assert written_bytes == []
    # MLflow 3 stores the model as a standalone "logged model" under
    # ``models/<model_id>/`` rather than in the run's ``model/`` artifact path.
    run = client.raw().get_run(run_id)
    assert run.data.tags["model.uri"] == ref.uri  # back-compat tag preserved
    logged_id = run.data.tags["model.logged_model_id"]
    assert logged_id
    assert ref.logged_uri == f"models:/{logged_id}"
    # The logged model exists and is discoverable from the run alone.
    logged_model = client.raw().get_logged_model(logged_id)
    assert logged_model.artifact_location
    assert "model.source_artifact" not in run.data.tags
    assert "source/model.py" not in _artifact_paths(run_id)


def test_load_model_source_prefers_generated_trial_model_from_pyfunc_code(
    active_run_with_gcs,
    tmp_path: Path,
):
    run_id, _, _ = active_run_with_gcs
    trial_model = tmp_path / "trial_model_abc123.py"
    trial_model.write_text("# generated trial marker\nclass Model: pass\n", encoding="utf-8")
    project_model = tmp_path / "projects" / "home_credit" / "model"
    project_model.mkdir(parents=True)
    (project_model / "__init__.py").write_text("# generic project model\n", encoding="utf-8")

    artifacts.write_model(
        run_id,
        TinyModel(),
        code_paths=[str(tmp_path / "projects"), str(trial_model)],
    )

    assert "generated trial marker" in artifacts.load_model_source(run_id)


def test_write_model_rejects_missing_string_path(active_run_with_gcs):
    run_id, _, _ = active_run_with_gcs

    with pytest.raises(FileNotFoundError):
        artifacts.write_pickle_model(run_id, "/missing/model.pkl")


def test_write_model_cloudpickles_plain_objects(active_run_with_gcs):
    run_id, _, written_bytes = active_run_with_gcs
    payload = {"model": "plain-object"}

    ref = artifacts.write_pickle_model(run_id, payload)

    assert written_bytes == []
    local_path = client.raw().download_artifacts(run_id, ref.path)
    with open(local_path, "rb") as handle:
        assert cloudpickle.loads(handle.read()) == payload
