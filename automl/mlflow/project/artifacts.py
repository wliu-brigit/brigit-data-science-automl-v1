"""Project-scoped artifact helpers.

Dataset artifacts moved to the experiment seam
(:mod:`automl.mlflow.experiment.artifacts`) — datasets are produced by an
experiment's pipeline, not by the project. The project overview run is
reserved for future project-level consolidated learnings.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from automl.errors import StorageError
from automl.mlflow import client
from automl.mlflow.artifact_paths import json_artifact_path
from automl.mlflow.project import overview as project_overview


def log_json(name: str, payload: dict) -> None:
    """Log a loose-tier JSON artifact on the project overview run."""
    artifact_path = json_artifact_path(name)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / Path(artifact_path).name
        tmp_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        _log_artifact(
            tmp_path,
            artifact_path=str(Path(artifact_path).parent)
            if Path(artifact_path).parent.as_posix() != "."
            else None,
        )


def _log_artifact(local_path: Path, *, artifact_path: str | None) -> None:
    run_id = project_overview._ensure_overview_run_id()
    try:
        client.raw().log_artifact(run_id, str(local_path), artifact_path=artifact_path)
    except Exception as exc:
        raise StorageError(f"Failed to log project artifact {local_path.name!r}") from exc


__all__ = ["log_json"]
