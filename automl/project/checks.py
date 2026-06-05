"""Project domain validation checks and recipe."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from automl.validate.base import Issue, ValidationReport, run_check


def validate_project(*, session=None, live: bool = False) -> ValidationReport:
    """Validate the active project.

    Structural checks always run. ``live=True`` adds service connectivity
    probes (GCS round-trip, MLflow query); the CLI always passes it so one
    verb reports the whole picture, while library callers and unit tests
    stay offline by default.
    """
    active = session
    if active is None:
        from automl.project.session import session as active_project_session

        active = active_project_session()
    config = active.config
    issues: list[Issue] = []
    issues.extend(
        run_check(
            "project.config_required_fields",
            config_required_fields,
            config=config,
        )
    )
    issues.extend(
        run_check(
            "project.environment_fields",
            environment_fields,
            config=config,
        )
    )
    issues.extend(
        run_check(
            "project.placeholder_values",
            placeholder_values,
            config=config,
        )
    )
    if live:
        issues.extend(
            run_check("project.connections.gcs", gcs_connection, config=config)
        )
        issues.extend(
            run_check(
                "project.connections.mlflow",
                mlflow_connection,
                config=config,
            )
        )
        issues.extend(
            run_check(
                "project.connections.snowflake",
                snowflake_connection,
                config=config,
            )
        )
    return ValidationReport(issues=issues)


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


def placeholder_values(*, config: Any) -> Iterable[Issue]:
    """Flag scaffold ``TBD_`` placeholders left in config.py and SQL files."""
    paths = []
    if config.config_path and config.config_path.exists():
        paths.append(config.config_path)
    source = getattr(config.data_spec, "source", None)
    if getattr(source, "kind", "") == "snowflake":
        for attr in ("base_table_sql", "training_data_sql"):
            raw = getattr(source, attr, None)
            if not raw:
                continue
            path = Path(raw)
            if not path.is_absolute():
                path = config.project_dir / path
            if path.exists():
                paths.append(path)
    for path in paths:
        if "TBD_" in path.read_text(encoding="utf-8"):
            yield Issue(
                level="error",
                check="project.placeholders",
                message="scaffold TBD_ placeholder values remain; fill them in",
                location=str(path),
            )


def gcs_connection(*, config: Any) -> Iterable[Issue]:
    """Probe GCS with a write/read/delete round-trip under the project prefix."""
    if not config.gcs_bucket or not config.gcs_prefix:
        return  # environment_fields already reports the missing variables
    from automl.utils.io import gcs

    probe_uri = gcs.join_uri(
        f"gs://{config.gcs_bucket}", config.gcs_prefix, ".validate", "probe.json"
    )
    try:
        gcs.write_json(probe_uri, {"check": "gcs_connection"}, overwrite=True)
        gcs.read_json(probe_uri)
        gcs.delete_prefix(probe_uri)
    except Exception as exc:  # noqa: BLE001 - surface the service error verbatim
        yield Issue(
            level="error",
            check="project.connections.gcs",
            message=f"GCS probe failed: {type(exc).__name__}: {exc}",
            location=probe_uri,
        )


def mlflow_connection(*, config: Any) -> Iterable[Issue]:
    """Probe the MLflow tracking server with one cheap query."""
    if not config.mlflow_tracking_uri:
        return  # environment_fields already reports the missing variable
    from automl.mlflow.client import check_connection

    try:
        check_connection(config.mlflow_tracking_uri)
    except Exception as exc:  # noqa: BLE001 - surface the service error verbatim
        yield Issue(
            level="error",
            check="project.connections.mlflow",
            message=f"MLflow tracking server check failed: {type(exc).__name__}: {exc}",
            location=config.mlflow_tracking_uri,
        )


def snowflake_connection(*, config: Any) -> Iterable[Issue]:
    """Live Snowflake probe: env vars present, SQL files on disk, SELECT 1 connects."""
    source = getattr(config.data_spec, "source", None)
    if getattr(source, "kind", "") != "snowflake":
        return
    from automl.utils.io import snowflake as sf

    missing = sf.missing_env()
    if missing:
        yield Issue(
            level="error",
            check="project.connections.snowflake",
            message=f"missing Snowflake environment variable(s): {', '.join(missing)}",
        )
        return
    for label, sql_path in (
        ("base_table_sql", source.base_table_sql),
        ("training_data_sql", source.training_data_sql),
    ):
        path = Path(sql_path)
        resolved = path if path.is_absolute() else config.project_dir / path
        if not resolved.exists():
            yield Issue(
                level="error",
                check="project.connections.snowflake",
                message=f"{label} file not found: {resolved}",
            )
    try:
        sf.check_connection()
    except Exception as exc:  # noqa: BLE001 - driver errors surface verbatim
        yield Issue(
            level="error",
            check="project.connections.snowflake",
            message=f"Snowflake connection failed: {exc}",
        )


__all__ = [
    "config_required_fields",
    "environment_fields",
    "gcs_connection",
    "mlflow_connection",
    "placeholder_values",
    "snowflake_connection",
    "validate_project",
]
