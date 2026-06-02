"""Trial-level metric, param, tag, and loose JSON logging."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Mapping

from automl.errors import StorageError
from automl.mlflow import client
from automl.mlflow.artifact_paths import json_artifact_path


def log_metric(run_id: str, key: str, value: float, step: int | None = None) -> None:
    try:
        client.raw().log_metric(run_id, key, float(value), step=step)
    except Exception as exc:
        raise StorageError(f"Failed to log MLflow metric {key!r}") from exc


def log_metrics(run_id: str, metrics: Mapping[str, float], step: int | None = None) -> None:
    for key, value in metrics.items():
        log_metric(run_id, key, value, step=step)


def log_param(run_id: str, key: str, value: object) -> None:
    try:
        client.raw().log_param(run_id, key, str(value))
    except Exception as exc:
        raise StorageError(f"Failed to log MLflow param {key!r}") from exc


def log_params(run_id: str, params: Mapping[str, object]) -> None:
    for key, value in params.items():
        log_param(run_id, key, value)


def set_tag(run_id: str, key: str, value: object) -> None:
    try:
        client.raw().set_tag(run_id, key, str(value))
    except Exception as exc:
        raise StorageError(f"Failed to set MLflow tag {key!r}") from exc


def set_tags(run_id: str, tags: Mapping[str, object]) -> None:
    for key, value in tags.items():
        set_tag(run_id, key, value)


def get_tags(run_id: str) -> dict[str, str]:
    try:
        return dict(client.raw().get_run(run_id).data.tags)
    except Exception as exc:
        raise StorageError(f"Failed to read MLflow tags for run {run_id!r}") from exc


def log_json(run_id: str, name: str, payload: dict) -> None:
    """Log a loose-tier JSON artifact under the run."""
    artifact_path = json_artifact_path(name)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / Path(artifact_path).name
        tmp_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        try:
            client.raw().log_artifact(
                run_id,
                str(tmp_path),
                artifact_path=str(Path(artifact_path).parent)
                if Path(artifact_path).parent.as_posix() != "."
                else None,
            )
        except Exception as exc:
            raise StorageError(f"Failed to log JSON artifact {artifact_path!r}") from exc


__all__ = [
    "log_json",
    "log_metric",
    "log_metrics",
    "log_param",
    "log_params",
    "get_tags",
    "set_tag",
    "set_tags",
]
