"""Validation checks and the validation recipe for model classes."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from automl.model.base import BaseModel
from automl.validate.base import Issue, ValidationReport, run_check


REQUIRED_POST_FIT_ATTRS = ("feature_registry", "preprocessor", "model", "name")


def validate_model(cls: type[Any], *, df, registry, session=None) -> ValidationReport:
    """Validate a model class end to end: contract, fit, predict, post-fit state."""
    issues: list[Issue] = []
    issues.extend(run_check("model.subclass_basemodel", subclass_basemodel, cls=cls))
    if any(issue.level == "error" for issue in issues):
        return ValidationReport(issues=issues)

    instance, error, error_stage = _try_fit(cls, df, registry, seed=0)
    issues.extend(
        run_check(
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
            run_check("model.predict_succeeds", predict_succeeds, instance=instance, df=df)
        )
    if not any(issue.level == "error" for issue in issues):
        issues.extend(
            run_check("model.post_fit_attrs_set", post_fit_attrs_set, cls=cls, instance=instance)
        )
        issues.extend(
            run_check(
                "model.required_transformers",
                check_required_transformers,
                instance=instance,
                session=session,
            )
        )
    return ValidationReport(issues=issues)


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


def subclass_basemodel(*, cls: type[Any]) -> Iterable[Issue]:
    if not isinstance(cls, type) or not issubclass(cls, BaseModel):
        return [
            Issue(
                level="error",
                check="model.subclass_basemodel",
                message="model class must subclass automl.model.BaseModel",
            )
        ]
    return []


def fit_succeeds(
    *,
    cls: type[Any],
    instance: Any | None,
    error: BaseException | None,
    error_stage: str | None,
) -> Iterable[Issue]:
    del cls, instance
    if error is None:
        return []
    stage = error_stage or "fit"
    return [
        Issue(
            level="error",
            check="model.fit_succeeds",
            message=f"model {stage} failed: {type(error).__name__}: {error}",
        )
    ]


def predict_succeeds(*, instance: Any, df) -> Iterable[Issue]:
    try:
        instance.predict(context=None, model_input=df)
    except Exception as exc:  # noqa: BLE001
        return [
            Issue(
                level="error",
                check="model.predict_succeeds",
                message=f"model predict failed: {type(exc).__name__}: {exc}",
            )
        ]
    return []


def post_fit_attrs_set(*, cls: type[Any], instance: Any) -> Iterable[Issue]:
    del cls
    missing = [
        name
        for name in REQUIRED_POST_FIT_ATTRS
        if not hasattr(instance, name) or getattr(instance, name) is None
    ]
    if not missing:
        return []
    return [
        Issue(
            level="error",
            check="model.post_fit_attrs_set",
            message=f"model fit must set post-fit attr(s): {', '.join(missing)}",
        )
    ]


def check_required_transformers(*, instance: Any, session: Any | None = None) -> Iterable[Issue]:
    from sklearn.compose import ColumnTransformer

    from automl.model.preprocessing import _requirements

    declared = list(_requirements(session))
    if not declared:
        return []
    requirements = [
        {"name": requirement.name, "columns": list(requirement.input_cols)}
        for requirement in declared
    ]

    preprocessor = getattr(instance, "preprocessor", None)
    if not isinstance(preprocessor, ColumnTransformer):
        return [
            Issue(
                level="error",
                check="model.required_transformers",
                message=(
                    "required transformers require instance.preprocessor to be a "
                    "top-level sklearn.compose.ColumnTransformer"
                ),
            )
        ]
    fitted_entries = getattr(preprocessor, "transformers_", None)
    if not fitted_entries:
        return [
            Issue(
                level="error",
                check="model.required_transformers",
                message="required transformers require a fitted ColumnTransformer",
            )
        ]

    by_name = {str(name): (transformer, columns) for name, transformer, columns in fitted_entries}
    issues: list[Issue] = []
    declared_by_name = {requirement.name: requirement for requirement in declared}

    for requirement in requirements:
        name = requirement["name"]
        if name not in by_name:
            issues.append(
                Issue(
                    level="error",
                    check="model.required_transformers",
                    message=f"required transformer {name!r} is missing from preprocessor",
                )
            )
            continue
        fitted_transformer, fitted_columns = by_name[name]
        declared_requirement = declared_by_name[name]
        if not isinstance(fitted_transformer, type(declared_requirement.transformer)):
            issues.append(
                Issue(
                    level="error",
                    check="model.required_transformers",
                    message=(
                        f"required transformer {name!r} must be "
                        f"{type(declared_requirement.transformer).__name__}, "
                        f"got {type(fitted_transformer).__name__}"
                    ),
                )
            )
            continue
        fitted_column_set = _named_column_set(fitted_columns)
        required_columns = set(requirement["columns"])
        if not required_columns.issubset(fitted_column_set):
            issues.append(
                Issue(
                    level="error",
                    check="model.required_transformers",
                    message=(
                        f"required transformer {name!r} must include columns "
                        f"{sorted(required_columns)}, got {sorted(fitted_column_set)}"
                    ),
                )
            )
    return issues


def _named_column_set(columns: object) -> set[str]:
    if isinstance(columns, str):
        return {columns}
    try:
        values = list(columns)  # type: ignore[arg-type]
    except TypeError:
        return set()
    if any(not isinstance(column, str) for column in values):
        return set()
    return set(values)


__all__ = [
    "REQUIRED_POST_FIT_ATTRS",
    "check_required_transformers",
    "fit_succeeds",
    "post_fit_attrs_set",
    "predict_succeeds",
    "subclass_basemodel",
    "validate_model",
]
