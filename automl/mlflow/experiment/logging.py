"""Experiment-level loose JSON logging."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from automl.errors import StorageError
from automl.mlflow import client
from automl.mlflow import tags
from automl.mlflow.artifact_paths import json_artifact_path
from automl.mlflow.experiment.lifecycle import ensure_overview, mlflow_experiment_id


def log_json(name: str, payload: dict, *, experiment_id: str | None = None) -> None:
    """Log a loose-tier JSON artifact on the experiment overview run."""

    artifact_path = json_artifact_path(name)
    ensure_overview(experiment_id)
    overview_run_id = _overview_run_id(experiment_id)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / Path(artifact_path).name
        tmp_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        try:
            client.raw().log_artifact(
                overview_run_id,
                str(tmp_path),
                artifact_path=str(Path(artifact_path).parent)
                if Path(artifact_path).parent.as_posix() != "."
                else None,
            )
        except Exception as exc:
            raise StorageError(f"Failed to log experiment JSON artifact {artifact_path!r}") from exc


def _overview_run_id(experiment_id: str | None = None) -> str:
    numeric_experiment_id = mlflow_experiment_id(experiment_id)
    if numeric_experiment_id is None:
        raise StorageError("Failed to resolve MLflow experiment for overview JSON")
    try:
        runs = client.raw().search_runs(
            [numeric_experiment_id],
            filter_string=f"tags.{tags.RUN_KIND} = 'experiment_overview'",
            max_results=1,
        )
    except Exception as exc:
        raise StorageError("Failed to read MLflow experiment overview run") from exc
    rows = list(runs)
    if not rows:
        raise StorageError("MLflow experiment overview run not found")
    return str(rows[0].info.run_id)


__all__ = ["log_json"]
