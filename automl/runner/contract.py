"""Runner-side contract checks for trial execution."""

from __future__ import annotations

from automl.errors import RunnerError
from automl.model import BaseModel


def require_validation_passed(report) -> None:
    """Raise if a validation report contains blocking errors."""

    if getattr(report, "passed", False):
        return
    messages = [
        f"{issue.check}: {issue.message}"
        for issue in getattr(report, "issues", ())
        if getattr(issue, "level", "") == "error"
    ]
    detail = "; ".join(messages) if messages else "unknown validation failure"
    raise RunnerError(f"model pre-fit validation failed: {detail}")


def validate_fitted_model(model, *, sample, registry) -> None:
    """Minimal post-fit contract checks before eval/logging."""

    missing = [
        name
        for name in ("feature_registry", "preprocessor", "model", "name")
        if not hasattr(model, name) or getattr(model, name) is None
    ]
    if missing:
        raise RunnerError(f"fitted model is missing required attribute(s): {missing}")
    _validate_method_ownership(model)
    _validate_method_contract(model, sample)
    _validate_registry_compatibility(getattr(model, "feature_registry"), registry)
    _validate_model_feature_declaration(model, getattr(model, "feature_registry"), registry)


def _validate_method_contract(model, sample) -> None:
    try:
        model.transform(sample)
        predictions = model.predict(context=None, model_input=sample)
    except Exception as exc:
        raise RunnerError(f"fitted model transform/predict contract failed: {exc}") from exc
    try:
        n_predictions = len(predictions)
    except TypeError as exc:
        raise RunnerError("fitted model predict must return a sized prediction vector") from exc
    if n_predictions != len(sample):
        raise RunnerError(
            "fitted model predict returned "
            f"{n_predictions} predictions for {len(sample)} input rows"
        )


def _validate_method_ownership(model) -> None:
    if not isinstance(model, BaseModel):
        raise RunnerError("fitted model must subclass automl.model.BaseModel")
    for name in ("predict", "predict_transformed"):
        owner = _method_owner(type(model), name)
        if owner is not BaseModel:
            raise RunnerError(f"fitted model must inherit BaseModel.{name} without overriding it")


def _method_owner(cls: type, name: str) -> type | None:
    for base in cls.__mro__:
        if name in base.__dict__:
            return base
    return None


def _validate_registry_compatibility(model_registry, fit_registry) -> None:
    fit_frame = fit_registry.to_dataframe()
    model_frame = model_registry.to_dataframe()
    missing_columns = sorted(set(fit_frame.columns) - set(model_frame.columns))
    if missing_columns:
        raise RunnerError(f"fitted model registry is missing registry columns: {missing_columns}")

    fit_names = set(fit_frame["name"])
    model_names = set(model_frame["name"])
    missing_entries = sorted(fit_names - model_names)
    if missing_entries:
        raise RunnerError(f"fitted model registry is missing registry entries: {missing_entries}")

    fit_targets = sorted(fit_frame.loc[fit_frame["target"].astype(bool), "name"])
    model_targets = sorted(model_frame.loc[model_frame["target"].astype(bool), "name"])
    if model_targets != fit_targets or len(model_targets) != 1:
        raise RunnerError(
            "fitted model registry target columns must match the fit registry: "
            f"expected {fit_targets}, found {model_targets}"
        )


def _validate_model_feature_declaration(model, model_registry, fit_registry) -> None:
    if model_registry is fit_registry:
        raise RunnerError(
            "fitted model feature_registry must be a deep-copy annotated for model use, "
            "not the dataset registry object"
        )

    feature_cols = getattr(model, "feature_cols", None)
    if feature_cols is None:
        raise RunnerError("fitted model must set feature_cols to the consumed raw input columns")
    if isinstance(feature_cols, str):
        raise RunnerError("fitted model feature_cols must be a list or tuple of column names")
    try:
        declared_cols = [str(column) for column in feature_cols]
    except TypeError as exc:
        raise RunnerError("fitted model feature_cols must be iterable") from exc
    if not declared_cols:
        raise RunnerError("fitted model feature_cols must not be empty")

    fit_names = set(fit_registry.to_dataframe()["name"])
    unknown_cols = sorted(set(declared_cols) - fit_names)
    if unknown_cols:
        raise RunnerError(f"fitted model feature_cols include unknown column(s): {unknown_cols}")

    flagged_cols = model_registry.get_by_flag("model")
    if sorted(declared_cols) != flagged_cols:
        raise RunnerError(
            "fitted model registry columns flagged model=True must exactly match "
            f"feature_cols: expected {sorted(declared_cols)}, found {flagged_cols}"
        )


__all__ = ["require_validation_passed", "validate_fitted_model"]
