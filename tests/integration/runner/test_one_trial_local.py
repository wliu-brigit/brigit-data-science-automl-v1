from __future__ import annotations

import json
import shutil
from pathlib import Path
from uuid import uuid4

import pytest

from automl.data import materialize
from automl.mlflow import client as mlflow_client
from automl.mlflow import tags
from automl.mlflow import trial as mlflow_trial
from automl.mlflow.trial import artifacts
from automl.project import Where, clear_session, use_project
from automl.runner import run_trial
from automl.trial import TrialStatus
from automl.trial.create import create as create_trial
from automl.utils.io import gcs

pytestmark = pytest.mark.integration


class FakeBlob:
    def __init__(self, store: dict[tuple[str, str], bytes], bucket: str, name: str) -> None:
        self._store = store
        self._bucket = bucket
        self.name = name

    def upload_from_string(
        self,
        data: str | bytes,
        *,
        content_type: str | None = None,
        if_generation_match: int | None = None,
    ) -> None:
        del content_type
        if if_generation_match == 0 and (self._bucket, self.name) in self._store:
            raise FileExistsError(f"object already exists: {self._bucket}/{self.name}")
        self._store[(self._bucket, self.name)] = (
            data if isinstance(data, bytes) else data.encode("utf-8")
        )

    def upload_from_file(
        self,
        file_obj,
        *,
        content_type: str | None = None,
        if_generation_match: int | None = None,
    ) -> None:
        del content_type
        if if_generation_match == 0 and (self._bucket, self.name) in self._store:
            raise FileExistsError(f"object already exists: {self._bucket}/{self.name}")
        self._store[(self._bucket, self.name)] = file_obj.read()

    def download_as_bytes(self) -> bytes:
        return self._store[(self._bucket, self.name)]

    def download_to_filename(
        self,
        filename: str,
        *,
        checksum: str | None = None,
        retry: object = None,
    ) -> None:
        del checksum, retry
        Path(filename).write_bytes(self._store[(self._bucket, self.name)])

    def exists(self) -> bool:
        return (self._bucket, self.name) in self._store


class FakeBucket:
    def __init__(self, store: dict[tuple[str, str], bytes], name: str) -> None:
        self._store = store
        self.name = name

    def blob(self, name: str) -> FakeBlob:
        return FakeBlob(self._store, self.name, name)


class FakeGCSClient:
    def __init__(self) -> None:
        self.store: dict[tuple[str, str], bytes] = {}

    def bucket(self, name: str) -> FakeBucket:
        return FakeBucket(self.store, name)


def test_run_trial_executes_homecredit_chain_and_logs_artifacts(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[3]
    fake_gcs = FakeGCSClient()
    monkeypatch.setattr(gcs, "_gcs_client", lambda: fake_gcs)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", (tmp_path / "mlruns").as_uri())
    monkeypatch.setenv("GCS_BUCKET", "automl-test-bucket")
    monkeypatch.setenv("GCS_PREFIX", "runner-local")

    active = use_project("example_homecredit", repo_root=repo_root)
    try:
        loaded = materialize(session=active)
        runner_loads: list[str | None] = []
        eval_loads: list[tuple[str, str | None, tuple[tuple[int, int], ...] | None]] = []

        import automl.eval._load as eval_load
        import automl.runner.trial as runner_trial

        original_runner_loader = runner_trial.data.load_dataset
        original_eval_loader = eval_load.data.load_dataset_by_id

        def counting_runner_loader(*, split_name=None, predicate=None, session=None):
            runner_loads.append(split_name)
            return original_runner_loader(
                split_name=split_name,
                predicate=predicate,
                session=session,
            )

        def counting_eval_loader(dataset_id, *, split_name=None, predicate=None, session=None):
            eval_loads.append((dataset_id, split_name, predicate))
            return original_eval_loader(
                dataset_id,
                split_name=split_name,
                predicate=predicate,
                session=session,
            )

        monkeypatch.setattr(runner_trial.data, "load_dataset", counting_runner_loader)
        monkeypatch.setattr(eval_load.data, "load_dataset_by_id", counting_eval_loader)

        result = run_trial("example_homecredit", session=active)

        assert result.status == TrialStatus.FINISHED.value
        assert result.error is None
        assert result.run_id
        assert result.trial_id == f"{result.trial_number}_homecredit_logistic"
        assert result.trial_number == 1
        assert result.metrics["auc"] >= 0.0
        assert result.metrics["auc"] <= 1.0
        assert runner_loads == [active.config.require_run_config().train_split]
        assert eval_loads == [
            (loaded.id, None, Where("SPLIT_PCT") >= 80),
            (loaded.id, None, Where("SPLIT_PCT") < 80),
            (loaded.id, None, None),
            (loaded.id, "test", None),
        ]

        run = mlflow_client.raw().get_run(result.run_id)
        assert run.data.metrics["eval.test.auc"] == result.metrics["auc"]
        assert "eval.train.auc" in run.data.metrics
        assert run.data.tags[tags.TRIAL_NUMBER] == "1"
        assert run.data.tags[tags.TRIAL_SLUG] == "homecredit_logistic"
        assert run.data.tags["trial.id"] == "1_homecredit_logistic"
        assert run.data.params["trial.strategy"] == "homecredit_logistic"
        assert "proposal.schema_version" not in run.data.params
        assert run.data.tags[tags.DATA_CONTRACT_URI] == "data/contract.json"
        assert run.data.tags[tags.MODEL_URI] == f"runs:/{result.run_id}/model"
        assert run.data.tags[tags.eval_dataset_id("test")]
        assert run.data.tags[tags.eval_dataset_id("train")]
        assert run.data.tags[tags.eval_predictions_uri("test")]
        assert run.data.tags[tags.eval_predictions_uri("train")]
        assert run.data.tags[tags.EVAL_INDEX_URI]
        assert not any(key.startswith("automl.trial") for key in run.data.tags)

        artifact_paths = {item.path for item in mlflow_trial.list_artifacts(result.run_id)}
        # MLflow 3 stores the model as a standalone logged model (under
        # ``models/<id>/``), not in the run's ``model/`` artifact path. It is
        # discoverable from the run via the persisted ``model.logged_model_id`` tag.
        logged_id = run.data.tags[tags.MODEL_LOGGED_ID]
        assert logged_id
        assert mlflow_client.raw().get_logged_model(logged_id).artifact_location
        assert {
            "data/contract.json",
            "eval/manifest.json",
            "eval/test/report.json",
            "eval/train/report.json",
            "features/dataset_feature_registry.csv",
            "features/feature_registry.csv",
            "manifest.json",
            "timing/summary.json",
            "trial/issues.json",
            "validation/data/input.csv",
            "validation/data/input.parquet",
            "validation/data/expected.parquet",
            "validation/data/input_schema.json",
            "validation/latency_detail.json",
            "validation/report.json",
        }.issubset(artifact_paths)
        assert "source/model.py" not in artifact_paths
        assert not any("/__pycache__/" in path or path.endswith(".pyc") for path in artifact_paths)
        assert not any("/experiments/" in path for path in artifact_paths)
        assert not any("/notebooks/" in path for path in artifact_paths)

        contract_path = mlflow_client.raw().download_artifacts(
            result.run_id,
            run.data.tags[tags.DATA_CONTRACT_URI],
        )
        eval_path = mlflow_client.raw().download_artifacts(
            result.run_id,
            run.data.tags[tags.eval_uri("test")],
        )
        train_eval_path = mlflow_client.raw().download_artifacts(
            result.run_id,
            run.data.tags[tags.eval_uri("train")],
        )

        contract = json.loads(Path(contract_path).read_text(encoding="utf-8"))
        eval_result = json.loads(Path(eval_path).read_text(encoding="utf-8"))
        train_eval_result = json.loads(Path(train_eval_path).read_text(encoding="utf-8"))

        assert contract["trial"]["run_id"] == result.run_id
        assert contract["trial"]["trial_id"] == result.trial_id
        assert contract["dataset"]["id"] == loaded.id
        assert set(contract["splits"]) == {"train", "test"}
        assert [item["name"] for item in contract["slices"]] == ["train", "test"]
        assert (
            run.data.tags["data.slice.train.content_hash"] == contract["slices"][0]["content_hash"]
        )
        assert (
            run.data.tags["data.slice.test.content_hash"] == contract["slices"][1]["content_hash"]
        )
        assert eval_result["label"] == "test"
        assert eval_result["metrics"][0]["name"] == "auc"
        assert eval_result["metrics"][0]["value"] == result.metrics["auc"]
        assert train_eval_result["label"] == "train"
        assert train_eval_result["metrics"][0]["name"] == "auc"
        assert artifacts.list_eval(result.run_id) == [
            ("test", run.data.tags[tags.eval_dataset_id("test")]),
            ("train", run.data.tags[tags.eval_dataset_id("train")]),
        ]
        index = artifacts.load_eval_index(result.run_id)
        assert index.primary_label == "test"
        assert [entry.label for entry in index.evaluations] == ["test", "train"]
        predictions = artifacts.load_predictions(result.run_id, "test")
        assert predictions.frame.shape[0] > 0
        assert "columns" not in predictions.manifest_dict()
        timing_path = mlflow_client.raw().download_artifacts(
            result.run_id,
            "timing/summary.json",
        )
        timing = json.loads(Path(timing_path).read_text(encoding="utf-8"))
        runner_phases = timing["phase_details"]["runner"]["phases"]
        assert timing["schema_version"] == 2
        assert list(timing["phases"]) == ["runner"]
        assert {
            "pre_fit_validation",
            "mlflow_setup",
            "model_import",
            "fit",
            "contract_validation",
            "local_artifacts",
            "evaluation",
            "mlflow_pyfunc_log",
            "validation_fixture",
            "validation",
            "validation_publish",
        }.issubset(runner_phases)
        manifest_path = mlflow_client.raw().download_artifacts(
            result.run_id,
            "manifest.json",
        )
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        assert list(manifest) == [
            "schema_version",
            "run",
            "data",
            "model",
            "evaluation",
            "validation",
            "timing",
            "artifacts",
        ]
        assert manifest["run"]["run_id"] == result.run_id
        assert manifest["run"]["trial_id"] == result.trial_id
        assert manifest["data"]["contract_artifact"] == "data/contract.json"
        assert manifest["evaluation"]["manifest_artifact"] == "eval/manifest.json"
        assert manifest["timing"]["schema_version"] == timing["schema_version"]
        assert manifest["timing"]["unit"] == timing["unit"]
        assert manifest["timing"]["total_seconds"] >= timing["total_seconds"]
        assert set(manifest["timing"]["phases"]) == set(timing["phases"])
        assert set(manifest["timing"]["phase_details"]["runner"]["phases"]) == set(
            timing["phase_details"]["runner"]["phases"]
        )
        assert manifest["validation"]["report_artifact"] == "validation/report.json"
        assert {item["path"] for item in manifest["artifacts"]} == {
            "data/contract.json",
            "eval/manifest.json",
            "features/dataset_feature_registry.csv",
            "features/feature_registry.csv",
            "model/MLmodel",
            "timing/summary.json",
            "trial/issues.json",
            "validation/data/input.csv",
            "validation/data/input.parquet",
            "validation/data/expected.parquet",
            "validation/latency_detail.json",
            "validation/report.json",
        }
        validation_path = mlflow_client.raw().download_artifacts(
            result.run_id,
            "validation/report.json",
        )
        validation_report = json.loads(Path(validation_path).read_text(encoding="utf-8"))
        assert validation_report["checks"]["input_schema"]["status"] == "passed"
        assert validation_report["checks"]["parquet_roundtrip"]["status"] == "passed"
        assert validation_report["checks"]["csv_string_roundtrip"]["status"] == "passed"
        assert validation_report["files"] == {
            "input_csv": "validation/data/input.csv",
            "input_parquet": "validation/data/input.parquet",
            "expected_parquet": "validation/data/expected.parquet",
            "latency_detail": "validation/latency_detail.json",
        }
        latency_path = mlflow_client.raw().download_artifacts(
            result.run_id,
            "validation/latency_detail.json",
        )
        latency = json.loads(Path(latency_path).read_text(encoding="utf-8"))
        assert latency["method"] == "fresh_process_loaded_pyfunc_single_row_repeated"
        assert latency["sample_count"] == 300
        assert len(latency["groups"]) == 3
        assert not any(blob.endswith("/model.pkl") for _, blob in fake_gcs.store)
    finally:
        clear_session()
        mlflow_client.clear()


def test_run_trial_rebinds_explicit_session_and_restores_prior_binding(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[3]
    tracking_uri = (tmp_path / "mlruns").as_uri()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    monkeypatch.setenv("GCS_BUCKET", "automl-test-bucket")
    monkeypatch.setenv("GCS_PREFIX", "prefix-a")
    session_a = use_project("example_homecredit", repo_root=repo_root)
    monkeypatch.setenv("GCS_PREFIX", "prefix-b")
    use_project("example_homecredit", repo_root=repo_root)

    observed: dict[str, str] = {}

    import automl.runner.trial as runner_trial

    def stop_after_bind(*, split_name=None, predicate=None, session=None):
        del split_name, predicate, session
        observed["prefix_at_load"] = mlflow_client.bound().gcs_prefix
        raise RuntimeError("stop after binding check")

    monkeypatch.setattr(runner_trial.data, "load_dataset", stop_after_bind)

    try:
        result = run_trial("example_homecredit", session=session_a)

        assert result.status == TrialStatus.FAILED.value
        assert observed["prefix_at_load"] == "prefix-a"
        assert mlflow_client.bound().gcs_prefix == "prefix-b"
    finally:
        clear_session()
        mlflow_client.clear()


def test_run_trial_logs_failure_report_and_traceback_artifacts(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[3]
    monkeypatch.setenv("MLFLOW_TRACKING_URI", (tmp_path / "mlruns").as_uri())

    # Unique route per run (like the sibling import-failure test): the dataset
    # record lives in this run's fresh MLflow, so re-using a fixed GCS route
    # across runs would trip materialize's refuse-to-overwrite guard.
    active = use_project(
        "example_homecredit",
        repo_root=repo_root,
        dry_run=True,
        namespace=f"qa_failure_{uuid4().hex}",
    )

    import automl.runner.trial as runner_trial

    def fail_contract_validation(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("forced contract failure")

    monkeypatch.setattr(
        runner_trial,
        "validate_fitted_model",
        fail_contract_validation,
    )

    try:
        materialize(session=active)
        result = run_trial("example_homecredit", session=active)

        assert result.status == TrialStatus.FAILED.value
        assert result.run_id
        assert result.error == "RuntimeError: forced contract failure"

        run = mlflow_client.raw().get_run(result.run_id)
        assert run.data.tags[tags.TRIAL_STATUS] == "FAILED"
        assert "trial.error_type" not in run.data.tags
        assert "trial.error_artifact" not in run.data.tags

        artifact_paths = {item.path for item in mlflow_trial.list_artifacts(result.run_id)}
        assert "logs/errors/report.json" in artifact_paths
        assert "logs/errors/traceback.txt" in artifact_paths

        report_path = mlflow_client.raw().download_artifacts(
            result.run_id,
            "logs/errors/report.json",
        )
        traceback_path = mlflow_client.raw().download_artifacts(
            result.run_id,
            "logs/errors/traceback.txt",
        )
        report = json.loads(Path(report_path).read_text(encoding="utf-8"))
        traceback_text = Path(traceback_path).read_text(encoding="utf-8")

        assert report["schema_version"] == 1
        assert report["status"] == "failed"
        assert report["runner_kind"] == "trial"
        assert report["phase"] == "contract_validation"
        assert report["error_class"] == "RuntimeError"
        assert report["message"] == "forced contract failure"
        assert report["trial_id"] == result.trial_id
        assert report["run_id"] == result.run_id
        assert report["traceback_artifact"] == "logs/errors/traceback.txt"
        assert "forced contract failure" in traceback_text
        assert "fail_contract_validation" in traceback_text
    finally:
        clear_session()
        mlflow_client.clear()


def test_run_trial_logs_generated_trial_import_failure_artifacts(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[3]
    monkeypatch.setenv("MLFLOW_TRACKING_URI", (tmp_path / "mlruns").as_uri())

    trial_path: Path | None = None
    active = use_project(
        "example_homecredit",
        repo_root=repo_root,
        dry_run=True,
        namespace=f"qa_failure_{uuid4().hex}",
    )
    bad_model = tmp_path / "bad_model.py"
    bad_model.write_text("class BrokenModel(\n", encoding="utf-8")

    try:
        trial_path = create_trial(
            slug="broken_import",
            strategy="syntax_check",
            hypothesis="Verify generated trial import failures are durable.",
            model_source=bad_model,
            proposal={
                "schema_version": 2,
                "slug": "broken_import",
                "strategy": "syntax_check",
                "hypothesis": "Verify generated trial import failures are durable.",
            },
            session=active,
        )

        result = run_trial(trial_path, session=active)

        assert result.status == TrialStatus.FAILED.value
        assert result.run_id
        assert result.trial_id == f"{result.trial_number}_broken_import"
        assert "SyntaxError" in (result.error or "")

        artifact_paths = {item.path for item in mlflow_trial.list_artifacts(result.run_id)}
        assert "agent/proposer/proposal.json" in artifact_paths
        assert "logs/errors/report.json" in artifact_paths
        assert "logs/errors/traceback.txt" in artifact_paths

        report_path = mlflow_client.raw().download_artifacts(
            result.run_id,
            "logs/errors/report.json",
        )
        report = json.loads(Path(report_path).read_text(encoding="utf-8"))
        assert report["runner_kind"] == "trial"
        assert report["phase"] == "model_import"
        assert report["trial_slug"] == "broken_import"
        assert report["trial_strategy"] == "syntax_check"
        assert report["proposal_artifact"] == "agent/proposer/proposal.json"
    finally:
        if trial_path is not None:
            shutil.rmtree(trial_path.parent.parent.parent.parent, ignore_errors=True)
        clear_session()
        mlflow_client.clear()
