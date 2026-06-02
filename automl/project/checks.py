"""Project domain validation checks."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from automl.validate.base import Issue


def config_required_fields(*, config: Any) -> Iterable[Issue]:
    required_fields = {
        "task": "TASK",
        "data_spec": "DATA",
        "eval_spec": "EVAL",
        "run_config": "RUN_CONFIG",
    }
    for attr, public_name in required_fields.items():
        if getattr(config, attr) is None:
            yield Issue(
                level="error",
                check=f"project.config.{attr}",
                message=f"{public_name} is missing from project config",
                location=str(config.config_path) if config.config_path else None,
            )


def environment_fields(*, config: Any) -> Iterable[Issue]:
    env_fields = {
        "gcs_bucket": "GCS_BUCKET",
        "gcs_prefix": "GCS_PREFIX",
        "mlflow_tracking_uri": "MLFLOW_TRACKING_URI",
    }
    for attr, env_name in env_fields.items():
        if not getattr(config, attr):
            yield Issue(
                level="error",
                check=f"project.env.{attr}",
                message=f"{env_name} is required",
            )


__all__ = ["config_required_fields", "environment_fields"]
