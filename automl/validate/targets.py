"""Validate orchestrators.

Target orchestrators lazily import per-domain checks and wrap them with
``_safe()`` so validation failures are reported as issues. Domain check modules
may import ``automl.validate.base`` value types, but must not import these
validate target orchestrators.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from automl.validate.base import Issue, ValidationReport


def model(cls: type[Any], *, df, registry, session=None) -> ValidationReport:
    from automl.model.checks import (
        check_required_transformers,
        fit_succeeds,
        post_fit_attrs_set,
        predict_succeeds,
        subclass_basemodel,
    )

    issues: list[Issue] = []
    issues.extend(_safe("model.subclass_basemodel", subclass_basemodel, cls=cls))
    if any(issue.level == "error" for issue in issues):
        return ValidationReport(issues=issues)

    instance, error, error_stage = _try_fit(cls, df, registry, seed=0)
    issues.extend(
        _safe(
            "model.fit_succeeds",
            fit_succeeds,
            cls=cls,
            instance=instance,
            error=error,
            error_stage=error_stage,
        )
    )
    if error is None:
        issues.extend(
            _safe("model.predict_succeeds", predict_succeeds, instance=instance, df=df)
        )
    if not any(issue.level == "error" for issue in issues):
        issues.extend(
            _safe("model.post_fit_attrs_set", post_fit_attrs_set, cls=cls, instance=instance)
        )
        issues.extend(
            _safe(
                "model.required_transformers",
                check_required_transformers,
                instance=instance,
                session=session,
            )
        )
    return ValidationReport(issues=issues)


def project(*args, live: bool = False, **kwargs) -> ValidationReport:
    """Validate the active project.

    Structural checks always run. ``live=True`` adds service connectivity
    probes (GCS round-trip, MLflow query); the CLI always passes it so one
    verb reports the whole picture, while library callers and unit tests
    stay offline by default.
    """
    del args
    from automl.project import checks as project_checks

    active = kwargs.get("session")
    if active is None:
        from automl.project import session as active_project_session

        active = active_project_session()
    config = active.config
    issues: list[Issue] = []
    issues.extend(
        _safe(
            "project.config_required_fields",
            project_checks.config_required_fields,
            config=config,
        )
    )
    issues.extend(
        _safe(
            "project.environment_fields",
            project_checks.environment_fields,
            config=config,
        )
    )
    issues.extend(
        _safe(
            "project.placeholder_values",
            project_checks.placeholder_values,
            config=config,
        )
    )
    if live:
        issues.extend(
            _safe("project.connections.gcs", project_checks.gcs_connection, config=config)
        )
        issues.extend(
            _safe(
                "project.connections.mlflow",
                project_checks.mlflow_connection,
                config=config,
            )
        )
        issues.extend(
            _safe(
                "project.connections.snowflake",
                project_checks.snowflake_connection,
                config=config,
            )
        )
    return ValidationReport(issues=issues)


def proposal(*, proposal: dict, session=None) -> ValidationReport:
    from automl.agent.checks import proposal_schema

    return ValidationReport(
        issues=_safe("proposal.schema", proposal_schema, proposal=proposal, session=session)
    )


def _safe(name: str, fn: Callable[..., Iterable[Issue]], **kwargs) -> list[Issue]:
    try:
        return list(fn(**kwargs))
    except Exception as exc:  # noqa: BLE001 - validation must report crashed checks
        return [
            Issue(
                level="error",
                check=f"{name}.crashed",
                message=f"check {name!r} raised {type(exc).__name__}: {exc}",
            )
        ]


def _try_fit(cls: type[Any], df, registry, *, seed: int) -> tuple[Any | None, BaseException | None, str | None]:
    try:
        instance = cls()
    except Exception as exc:  # noqa: BLE001
        return None, exc, "construct"
    try:
        fitted = instance.fit(df, registry, seed=seed)
        if fitted is not None:
            instance = fitted
    except Exception as exc:  # noqa: BLE001
        return instance, exc, "fit"
    return instance, None, None


__all__ = ["model", "project", "proposal"]
