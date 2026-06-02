from __future__ import annotations

import json
from pathlib import Path

import pytest

from automl.mlflow import client, experiment, trial
from automl.mlflow.trial.artifacts import runner as runner_artifacts
from automl.trial.metadata import TimingReport

pytestmark = pytest.mark.unit


@pytest.fixture
def active_run(tmp_path):
    client.clear()
    client.bind(
        tracking_uri=(tmp_path / "mlruns").as_uri(),
        bucket="",
        gcs_prefix="",
        project_name="home_credit",
        experiment_id="baseline",
    )
    experiment.ensure()
    with trial.active(slug="baseline_lr", strategy="baseline") as run_id:
        yield run_id
    client.clear()


def _read_json_artifact(run_id: str, path: str) -> dict:
    local_path = client.raw().download_artifacts(run_id, path)
    with open(local_path, encoding="utf-8") as handle:
        return json.load(handle)


def test_write_local_file_logs_to_exact_artifact_path(
    active_run: str,
    tmp_path: Path,
):
    local_path = tmp_path / "report.json"
    local_path.write_text('{"ok": true}', encoding="utf-8")

    runner_artifacts.write_local_file(
        active_run,
        "runner/report.json",
        local_path,
    )

    downloaded = client.raw().download_artifacts(active_run, "runner/report.json")
    assert Path(downloaded).read_text(encoding="utf-8") == '{"ok": true}'


def test_write_timing_logs_summary_json_with_timing_report_shape(active_run: str):
    timing = {
        "schema_version": 1,
        "unit": "seconds",
        "total_seconds": 3.25,
        "phases": {
            "fit": 1.5,
            "evaluation": 0.75,
        },
    }

    runner_artifacts.write_timing(active_run, timing)

    assert _read_json_artifact(active_run, "timing/summary.json") == (
        TimingReport.from_dict(timing).to_dict()
    )
