from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def test_runner_failure_report_serializes_exception_and_runner_context(tmp_path):
    from automl.runner.failures import ExceptionSnapshot, RunnerFailureReport

    try:
        raise RuntimeError("forced failure")
    except RuntimeError as exc:
        snapshot = ExceptionSnapshot.from_exception(exc)

    report = RunnerFailureReport(
        runner_kind="trial",
        phase="fit",
        exception=snapshot,
        run_id="run-1",
        project_name="demo",
        experiment_id="exp",
        trial_id="3_tree",
        trial_number=3,
        trial_slug="tree",
        trial_strategy="boosted_tree",
        trial_dir=tmp_path / "trial",
        timing={"schema_version": 1, "phases": {"fit": 1.2}},
    )

    payload = report.to_dict(
        traceback_artifact="logs/errors/traceback.txt",
        proposal_artifact="agent/proposer/proposal.json",
    )

    assert payload["schema_version"] == 1
    assert payload["status"] == "failed"
    assert payload["runner_kind"] == "trial"
    assert payload["phase"] == "fit"
    assert payload["error_class"] == "RuntimeError"
    assert payload["message"] == "forced failure"
    assert payload["run_id"] == "run-1"
    assert payload["project_name"] == "demo"
    assert payload["experiment_id"] == "exp"
    assert payload["trial_id"] == "3_tree"
    assert payload["trial_number"] == 3
    assert payload["trial_slug"] == "tree"
    assert payload["trial_strategy"] == "boosted_tree"
    assert payload["trial_dir"] == str(tmp_path / "trial")
    assert payload["timing"] == {"schema_version": 1, "phases": {"fit": 1.2}}
    assert payload["traceback_artifact"] == "logs/errors/traceback.txt"
    assert payload["proposal_artifact"] == "agent/proposer/proposal.json"
    assert any("forced failure" in line for line in payload["traceback_tail"])
    assert "traceback" not in payload
    assert "forced failure" in snapshot.traceback_text
