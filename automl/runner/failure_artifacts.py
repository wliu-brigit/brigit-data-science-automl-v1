"""Runner failure artifact publishing helpers."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from automl.mlflow.trial.artifacts import runner as runner_artifacts
from automl.mlflow.trial.artifacts.failure import (
    ERROR_REPORT_ARTIFACT,
    ERROR_TRACEBACK_ARTIFACT,
)
from automl.runner.failures import RunnerFailureReport


def log_failure_artifacts(
    *,
    failure: RunnerFailureReport,
    has_agent_proposal: bool = False,
) -> None:
    proposal_artifact = "agent/proposer/proposal.json" if has_agent_proposal else ""
    report = failure.to_dict(
        traceback_artifact=ERROR_TRACEBACK_ARTIFACT,
        proposal_artifact=proposal_artifact,
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        report_path = root / "report.json"
        traceback_path = root / "traceback.txt"
        report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        traceback_path.write_text(failure.exception.traceback_text, encoding="utf-8")
        runner_artifacts.write_local_file(
            failure.run_id,
            ERROR_REPORT_ARTIFACT,
            report_path,
        )
        runner_artifacts.write_local_file(
            failure.run_id,
            ERROR_TRACEBACK_ARTIFACT,
            traceback_path,
        )


__all__ = ["log_failure_artifacts"]
