"""Trial failure diagnostic artifact readers."""

from __future__ import annotations

import json

from automl.errors import StorageError
from automl.mlflow import client


ERROR_REPORT_ARTIFACT = "logs/errors/report.json"
ERROR_TRACEBACK_ARTIFACT = "logs/errors/traceback.txt"


def load_error_report(run_id: str) -> dict:
    try:
        path = client.download_artifact(run_id, ERROR_REPORT_ARTIFACT, required=True)
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        raise StorageError(
            f"Failed to read MLflow failure artifact {ERROR_REPORT_ARTIFACT!r}"
        ) from exc
    if not isinstance(payload, dict):
        raise StorageError(f"Expected JSON object at {ERROR_REPORT_ARTIFACT!r}")
    return payload


__all__ = [
    "ERROR_REPORT_ARTIFACT",
    "ERROR_TRACEBACK_ARTIFACT",
    "load_error_report",
]
