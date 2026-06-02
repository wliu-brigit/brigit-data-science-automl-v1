"""MLflow seam for runner-produced trial artifacts.

This module keeps runner artifact writes behind trial MLflow verbs; it is not a
new public runner artifact concept.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from automl.mlflow import client
from automl.mlflow.trial.logging import log_json
from automl.trial.metadata import TimingReport


def write_local_file(run_id: str, artifact_path: str, local_path: Path) -> None:
    client.log_artifact_file(run_id, artifact_path, local_path)


def write_timing(run_id: str, timing: Mapping[str, object]) -> None:
    report = TimingReport.from_dict(timing).to_dict()
    log_json(run_id, "timing/summary", report)


__all__ = ["write_local_file", "write_timing"]
