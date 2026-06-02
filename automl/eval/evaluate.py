"""Evaluation verbs for the public eval path."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

import pandas as pd

from automl.eval._load import load_eval_augmentations, load_eval_dataset
from automl.eval.base import scalar_metric_records
from automl.eval.results import EvalIndex, EvalIndexEntry, Predictions
from automl.eval.results import EvalResult
from automl.mlflow import tags as mlflow_tags
from automl.mlflow import trial as mlflow_trial
from automl.mlflow.trial import artifacts
from automl.project import Session
from automl.project import session as active_project_session


def evaluate(
    *,
    session: Session | None = None,
    model_run_id: str,
    eval_dataset_id: str,
    label: str,
    eval_spec=None,
    set_as_primary_label: bool = False,
    overwrite: bool = False,
    _model=None,
    _model_feature_registry=None,
) -> EvalResult:
    active = _session(session)
    if not overwrite:
        cached = _load_cached_result(
            model_run_id=model_run_id,
            eval_dataset_id=eval_dataset_id,
            label=label,
        )
        if cached is not None:
            result, index = cached
            if set_as_primary_label:
                artifacts.write_eval_index(
                    model_run_id,
                    EvalIndex(primary_label=label, evaluations=index.evaluations),
                )
                _log_scalar_metrics(model_run_id, result, set_as_primary_label=True)
            return result
    if _model is None:
        _model = artifacts.load_model(model_run_id)

    loaded = load_eval_dataset(eval_dataset_id, session=active)
    if _model_feature_registry is not None and not hasattr(_model, "feature_registry"):
        _model.feature_registry = _model_feature_registry
    target_col = loaded.target_column or active.config.target_column
    spec = eval_spec or active.config.require_eval_spec()
    y_pred = _predict_model(_model, loaded.df)
    y_pred = pd.Series(y_pred, index=loaded.df.index)
    augmentation_frames, augmentations_used = _load_required_augmentations(
        eval_dataset_id,
        spec=spec,
        session=active,
    )
    evaluated = spec.evaluate(
        loaded.df,
        y_pred,
        target_col,
        augmentation_frames=augmentation_frames,
        hash_key=loaded.hash_key,
    )
    computed_at = datetime.now(UTC).isoformat()
    predictions = Predictions(
        trial_run_id=model_run_id,
        eval_dataset_id=eval_dataset_id,
        eval_dataset_kind=loaded.dataset.kind,
        label=label,
        hash_key=loaded.hash_key,
        frame=_prediction_frame(loaded.row_ids, y_pred),
        augmentations_used=augmentations_used,
        written_at=computed_at,
    )
    prediction_ref = artifacts.write_predictions(model_run_id, label, predictions, overwrite=overwrite)
    result = EvalResult(
        label=label,
        eval_dataset_id=eval_dataset_id,
        eval_dataset_kind=loaded.dataset.kind,
        predictions_uri=prediction_ref.uri,
        predictions_manifest_uri=prediction_ref.manifest_uri,
        augmentations_used=augmentations_used,
        primary=str(evaluated["primary"]),
        metrics=evaluated["metrics"],
        computed_at=computed_at,
    )
    eval_ref = artifacts.write_eval(model_run_id, label, result, overwrite=overwrite)
    _write_eval_index(
        model_run_id,
        label=label,
        result=result,
        report_path=eval_ref.path,
        eval_dataset_manifest_uri=loaded.dataset.manifest_gcs_uri,
        set_as_primary_label=set_as_primary_label,
    )
    _log_scalar_metrics(model_run_id, result, set_as_primary_label=set_as_primary_label)
    return result


def evaluate_frame(*, y_pred, df, spec, target_col: str, session: Session | None = None):
    del session
    return spec.evaluate(df, y_pred, target_col)


def _session(explicit: Session | None) -> Session:
    return explicit if explicit is not None else active_project_session()


def _load_cached_result(
    *,
    model_run_id: str,
    eval_dataset_id: str,
    label: str,
) -> tuple[EvalResult, EvalIndex] | None:
    try:
        result = artifacts.load_eval(model_run_id, label)
        predictions = artifacts.load_predictions(model_run_id, label)
        index = artifacts.load_eval_index(model_run_id)
    except Exception:
        return None
    if result.eval_dataset_id != eval_dataset_id:
        return None
    if predictions.eval_dataset_id != eval_dataset_id:
        return None
    if not any(
        entry.label == label and entry.eval_dataset_id == eval_dataset_id
        for entry in index.evaluations
    ):
        return None
    return replace(result, cached=True), index


def _load_required_augmentations(eval_dataset_id: str, *, spec, session: Session):
    names = tuple(spec.required_augmentations())
    if not names:
        return {}, ()
    return load_eval_augmentations(eval_dataset_id, names=names, session=session)


def _prediction_frame(row_ids: pd.DataFrame, y_pred: pd.Series) -> pd.DataFrame:
    frame = row_ids.reset_index(drop=True).copy()
    frame["y_pred"] = y_pred.reset_index(drop=True)
    return frame


def _predict_model(model, frame: pd.DataFrame):
    try:
        return model.predict(context=None, model_input=frame)
    except TypeError:
        return model.predict(frame)


def _write_eval_index(
    run_id: str,
    *,
    label: str,
    result: EvalResult,
    report_path: str,
    eval_dataset_manifest_uri: str,
    set_as_primary_label: bool,
) -> None:
    existing = artifacts.load_eval_index(run_id)
    entry = EvalIndexEntry(
        label=label,
        eval_dataset_id=result.eval_dataset_id,
        kind=result.eval_dataset_kind,
        report_path=report_path,
        eval_dataset_manifest_uri=eval_dataset_manifest_uri,
        predictions_uri=result.predictions_uri,
        predictions_manifest_uri=result.predictions_manifest_uri,
        augmentations_used=result.augmentations_used,
        computed_at=result.computed_at,
    )
    entries = list(existing.evaluations)
    for index, current in enumerate(entries):
        if current.label == label:
            entries[index] = entry
            break
    else:
        entries.append(entry)
    primary_label = label if set_as_primary_label else existing.primary_label
    artifacts.write_eval_index(
        run_id,
        EvalIndex(primary_label=primary_label, evaluations=tuple(entries)),
    )


def _log_scalar_metrics(
    run_id: str,
    result: EvalResult,
    *,
    set_as_primary_label: bool,
) -> None:
    scalar_metrics = scalar_metric_records({"primary": result.primary, "metrics": result.metrics})
    mlflow_trial.log_metrics(
        run_id,
        {f"eval.{result.label}.{name}": value for name, value in scalar_metrics.items()},
    )
    if not set_as_primary_label:
        return
    mlflow_trial.set_tag(run_id, mlflow_tags.EVAL_PRIMARY_LABEL, result.label)


__all__ = ["evaluate", "evaluate_frame"]
