"""Trial issue-ledger publishing helpers."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from automl.mlflow import tags as mlflow_tags
from automl.mlflow import trial as mlflow_trial
from automl.mlflow.trial.artifacts import runner as runner_artifacts
from automl.runner.issues import IssueRecorder

ISSUES_ARTIFACT = "trial/issues.json"


def log_issue_artifacts(run_id: str, issues: IssueRecorder) -> None:
    """Publish the ledger + count tag. Called on BOTH trial exit paths."""
    if not run_id:
        return
    payload = {"schema_version": 1, "issues": issues.snapshot()}
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "issues.json"
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        runner_artifacts.write_local_file(run_id, ISSUES_ARTIFACT, path)
    mlflow_trial.set_tags(run_id, {mlflow_tags.TRIAL_ISSUE_COUNT: issues.count})


__all__ = ["ISSUES_ARTIFACT", "log_issue_artifacts"]
