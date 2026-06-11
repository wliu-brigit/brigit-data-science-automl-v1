import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from automl.errors import ValidationError
from automl.project import ProjectConfig, Session
from automl.runner import artifacts, serving_validation

pytestmark = pytest.mark.unit


class _FakeConfig:
    mlflow_tracking_uri = "http://127.0.0.1:9"
    repo_root = Path(".")
    run_config = None


class _FakeSession:
    config = _FakeConfig()


def _session(tmp_path):
    return Session(
        config=ProjectConfig(
            project_name="demo",
            repo_root=tmp_path,
            project_dir=tmp_path / "projects" / "demo",
            config_path=tmp_path / "projects" / "demo" / "config.py",
            gcs_bucket="bucket",
            mlflow_tracking_uri="file:///tmp/mlruns",
        )
    )


def test_validation_requires_eval_rows_raises_validation_error(tmp_path, monkeypatch):
    active = _session(tmp_path)
    loaded = SimpleNamespace(df=pd.DataFrame({"feature": []}))
    monkeypatch.setattr(
        serving_validation.data,
        "load_dataset_by_id",
        lambda *args, **kwargs: loaded,
    )

    with pytest.raises(ValidationError, match="validation requires at least one eval row"):
        artifacts.log_validation_artifacts(
            run_id="run-1",
            active=active,
            model=object(),
            dataset_id="dataset-1",
            eval_split="test",
            model_registry=None,
        )


def test_pyfunc_validation_rejects_non_object_report_with_validation_error(
    tmp_path,
    monkeypatch,
):
    active = _session(tmp_path)
    report_path = tmp_path / "report.json"
    report_path.write_text("[]", encoding="utf-8")

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 0, stdout="", stderr="")

    monkeypatch.setattr(serving_validation.subprocess, "run", fake_run)

    with pytest.raises(ValidationError, match="validation report must be a JSON object"):
        serving_validation._run_pyfunc_validation(
            run_id="run-1",
            active=active,
            input_parquet=tmp_path / "input.parquet",
            input_csv=tmp_path / "input.csv",
            expected_parquet=tmp_path / "expected.parquet",
            input_schema=tmp_path / "input_schema.json",
            report_path=report_path,
            tolerance=1e-10,
        )


def test_validation_publish_uses_local_artifact_writer(tmp_path, monkeypatch):
    active = _session(tmp_path)
    loaded = SimpleNamespace(df=pd.DataFrame({"feature": [1.0]}))
    writes = []

    class Model:
        def predict(self, context, model_input):
            return pd.Series([0.25] * len(model_input))

    monkeypatch.setattr(
        serving_validation.data,
        "load_dataset_by_id",
        lambda *args, **kwargs: loaded,
    )
    monkeypatch.setattr(
        serving_validation,
        "_run_pyfunc_validation",
        lambda **kwargs: {
            "schema_version": 1,
            "status": "passed",
            "row_count": 1,
            "max_abs_diff": 0.0,
            "tolerance": kwargs["tolerance"],
            "checks": {},
            "latency": {"latency_ms": {}},
        },
    )
    monkeypatch.setattr(
        serving_validation,
        "_log_validation_tags_and_metrics",
        lambda run_id, report: None,
    )
    monkeypatch.setattr(
        serving_validation.runner_artifacts,
        "write_local_file",
        lambda run_id, artifact_path, local_path: writes.append(
            (run_id, artifact_path, local_path.exists())
        ),
    )

    serving_validation.log_validation_artifacts(
        run_id="run-1",
        active=active,
        model=Model(),
        dataset_id="dataset-1",
        eval_split="test",
        model_registry=None,
    )

    # Fixtures (input + expected + schema) are published up front, before the
    # benchmark subprocess, so a failed/timed-out validation still leaves them
    # behind. Latency + report are published afterward.
    assert writes == [
        ("run-1", "validation/data/input.csv", True),
        ("run-1", "validation/data/input.parquet", True),
        ("run-1", "validation/data/expected.parquet", True),
        ("run-1", "validation/data/input_schema.json", True),
        ("run-1", "validation/latency_detail.json", True),
        ("run-1", "validation/report.json", True),
    ]


def test_timeout_report_serializes_with_bytes_stderr(tmp_path, monkeypatch):
    def _raise_timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=["python"], timeout=7, stderr=b"\xff boom \xfe")

    monkeypatch.setattr(serving_validation.subprocess, "run", _raise_timeout)
    report_path = tmp_path / "report.json"
    report = serving_validation._run_pyfunc_validation(
        run_id="run123",
        active=_FakeSession(),
        input_parquet=tmp_path / "input.parquet",
        input_csv=tmp_path / "input.csv",
        expected_parquet=tmp_path / "expected.parquet",
        input_schema=tmp_path / "input_schema.json",
        report_path=report_path,
        tolerance=1e-10,
    )
    assert report["status"] == "failed"
    assert report["error_class"] == "TimeoutExpired"
    assert isinstance(report["stderr_tail"], str)
    assert json.loads(report_path.read_text(encoding="utf-8"))["status"] == "failed"


def test_validation_timeout_uses_run_config(tmp_path, monkeypatch):
    captured = {}

    def _capture_run(*args, **kwargs):
        captured["timeout"] = kwargs.get("timeout")
        raise subprocess.TimeoutExpired(cmd=["python"], timeout=kwargs.get("timeout"))

    monkeypatch.setattr(serving_validation.subprocess, "run", _capture_run)

    class _RunConfig:
        serving_validation_seconds = 77

    class _Config(_FakeConfig):
        run_config = _RunConfig()

    class _Session:
        config = _Config()

    report = serving_validation._run_pyfunc_validation(
        run_id="run123",
        active=_Session(),
        input_parquet=tmp_path / "input.parquet",
        input_csv=tmp_path / "input.csv",
        expected_parquet=tmp_path / "expected.parquet",
        input_schema=tmp_path / "input_schema.json",
        report_path=tmp_path / "report.json",
        tolerance=1e-10,
    )
    assert captured["timeout"] == 77
    assert "77" in report["error"]
