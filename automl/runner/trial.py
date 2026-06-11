"""Straight-line trial runner."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from dataclasses import dataclass, field
from json import JSONDecodeError
from pathlib import Path
from typing import Any

import automl.data as data
from automl.data import DatasetRef, SliceContract, TrialDataContract, TrialRef
from automl.eval import evaluate, prepare_eval_dataset
from automl.eval.base import scalar_metric_records
from automl.errors import ProjectError, format_error_chain
from automl.mlflow import client as mlflow_client
from automl.mlflow import experiment as mlflow_experiment
from automl.mlflow import tags as mlflow_tags
from automl.mlflow import trial as mlflow_trial
from automl.model import validate_model
from automl.project import Session, find_repo_root, session as active_project_session, use_project
from automl.trial.metadata import TrialMetadata
from automl.trial.paths import trial_slug, verify_trial_dir
from automl.trial.timing_summary import build_runner_timing_summary
from automl.utils.hashing import dataframe_content_hash

from .artifacts import (
    TimingRecorder,
    log_agent_proposal,
    log_data_contract,
    log_failure_artifacts,
    log_feature_artifacts,
    log_manifest,
    log_model,
    log_timing,
    log_validation_artifacts,
)
from .contract import require_validation_passed, validate_fitted_model
from .failures import ExceptionSnapshot, RunnerFailureReport


@dataclass(frozen=True)
class TrialResult:
    status: str
    run_id: str = ""
    trial_id: str = ""
    trial_number: int | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    error: str | None = None


@dataclass(frozen=True)
class TrialExecutionContext:
    session: Session
    trial_dir: Path | None = None
    metadata: TrialMetadata | None = None
    dataset_id: str | None = None


def run_trial(
    path_or_project: str | Path,
    *,
    session: Session | None = None,
    dataset_id: str | None = None,
) -> TrialResult:
    """Run one project model through the data->fit->eval->log chain."""

    # MLflow's HTTP retry budget is capped once, seam-wide, at import of
    # automl.mlflow.client (HTTP_MAX_RETRIES) — no runner-specific override.
    context = _execution_context(path_or_project, session=session, dataset_id=dataset_id)
    with mlflow_client.bound_for(
        context.session,
        experiment_id=_active_experiment_id_or_none(context.session),
    ):
        return _run_trial(context)


def _run_trial(context: TrialExecutionContext) -> TrialResult:
    active = context.session
    run_config = active.config.require_run_config()
    eval_spec = active.config.require_eval_spec()
    target_col = active.config.target_column
    run_id = ""
    trial_number: int | None = None
    trial_id = ""
    slug = ""
    strategy = ""
    model_cls: type[Any] | None = None
    timing = TimingRecorder()

    try:
        with timing.phase("model_import"):
            model_cls = _load_model_class(context)
        with timing.phase("data_load"):
            if context.dataset_id:
                loaded_fit = data.load_dataset_by_id(
                    context.dataset_id,
                    split_name=run_config.train_split,
                    session=active,
                )
            else:
                loaded_fit = data.load_dataset(split_name=run_config.train_split, session=active)
        sample = loaded_fit.df.head(200)
        with timing.phase("pre_fit_validation"):
            require_validation_passed(
                validate_model(
                    model_cls,
                    df=sample,
                    registry=loaded_fit.registry,
                    session=active,
                )
            )
            eval_spec.validate_columns(loaded_fit.df, target_col)

        with timing.phase("mlflow_setup"):
            mlflow_experiment.ensure(experiment_id=active.active_experiment_id)
            trial_number = mlflow_experiment.next_trial_number(
                experiment_id=active.active_experiment_id
            )
            slug = _trial_slug(context, model_cls)
            strategy = _trial_strategy(context, slug)
            trial_id = f"{trial_number}_{slug}"

        with mlflow_trial.active(
            slug=slug,
            strategy=strategy,
            experiment_id=active.active_experiment_id,
        ) as active_run_id:
            run_id = active_run_id
            run_tags = {
                mlflow_tags.TRIAL_NUMBER: trial_number,
                mlflow_tags.TRIAL_ID: trial_id,
            }
            if context.metadata is not None:
                run_tags.update(
                    {
                        mlflow_tags.TRIAL_TRAINING_ORIGIN: context.metadata.training_origin,
                    }
                )
            else:
                run_tags[mlflow_tags.TRIAL_TRAINING_ORIGIN] = "project"
            mlflow_trial.set_tags(run_id, run_tags)
            mlflow_trial.log_param(
                run_id,
                mlflow_tags.TRIAL_HYPOTHESIS,
                _trial_hypothesis(context, slug),
            )
            model = model_cls()
            with timing.phase("fit"):
                fitted = model.fit(loaded_fit.df, loaded_fit.registry, seed=0)
            if fitted is not None:
                model = fitted
            with timing.phase("contract_validation"):
                validate_fitted_model(
                    model,
                    sample=sample.head(10),
                    registry=loaded_fit.registry,
                )

            model_registry = getattr(model, "feature_registry", loaded_fit.registry)
            has_agent_proposal = False
            with timing.phase("local_artifacts"):
                has_agent_proposal = log_agent_proposal(
                    run_id=run_id,
                    trial_dir=context.trial_dir,
                )
                log_feature_artifacts(
                    run_id=run_id,
                    dataset_registry=loaded_fit.registry,
                    model_registry=model_registry,
                    model=model,
                )
            with timing.phase("mlflow_pyfunc_log"):
                model_ref = log_model(
                    run_id=run_id,
                    active=active,
                    trial_dir=context.trial_dir,
                    model=model,
                    sample=sample,
                )
            with timing.phase("evaluation"):
                eval_dataset, _ = prepare_eval_dataset(
                    session=active,
                    dataset_id=loaded_fit.id,
                    split=run_config.eval_split,
                )
                eval_result = evaluate(
                    session=active,
                    model_run_id=run_id,
                    eval_dataset_id=eval_dataset.id,
                    label=run_config.eval_split,
                    set_as_primary_label=True,
                    _model=model,
                    _model_feature_registry=model_registry,
                )
                _try_log_train_eval(
                    run_id=run_id,
                    active=active,
                    model=model,
                    dataset_id=loaded_fit.id,
                    train_split=run_config.train_split,
                    feature_registry=model_registry,
                )
            contract = _trial_data_contract(
                active=active,
                run_id=run_id,
                trial_id=trial_id,
                loaded_fit=loaded_fit,
            )
            log_data_contract(run_id, contract)
            validation_report = log_validation_artifacts(
                run_id=run_id,
                active=active,
                model=model,
                dataset_id=loaded_fit.id,
                eval_split=run_config.eval_split,
                model_registry=model_registry,
                timing=timing,
                model_uri=model_ref.logged_uri,
            )
            timing_summary = build_runner_timing_summary(timing.snapshot())
            log_timing(run_id, timing_summary)
            log_manifest(
                run_id=run_id,
                active=active,
                trial_id=trial_id,
                trial_number=trial_number,
                slug=slug,
                strategy=strategy,
                contract=contract,
                eval_result=eval_result,
                validation_report=validation_report,
                timing=timing_summary,
                has_agent_proposal=has_agent_proposal,
            )
            return TrialResult(
                status="FINISHED",
                run_id=run_id,
                trial_id=trial_id,
                trial_number=trial_number,
                metrics=scalar_metric_records(eval_result.to_dict()),
            )
    except Exception as exc:  # noqa: BLE001 - runner returns a typed failure result
        if not run_id:
            failure_run = _start_failure_run(
                context,
                active=active,
                model_cls=model_cls,
                slug=slug,
                strategy=strategy,
                trial_number=trial_number,
            )
            if failure_run is not None:
                run_id, trial_id, trial_number, slug, strategy = failure_run
                try:
                    _publish_failure_artifacts(
                        run_id=run_id,
                        active=active,
                        trial_id=trial_id,
                        trial_number=trial_number,
                        slug=slug,
                        strategy=strategy,
                        timing=timing,
                        exc=exc,
                        trial_dir=context.trial_dir,
                    )
                finally:
                    mlflow_trial.end(run_id, "FAILED")
        else:
            _publish_failure_artifacts(
                run_id=run_id,
                active=active,
                trial_id=trial_id,
                trial_number=trial_number,
                slug=slug,
                strategy=strategy,
                timing=timing,
                exc=exc,
                trial_dir=context.trial_dir,
            )
        return TrialResult(
            status="FAILED",
            run_id=run_id,
            trial_id=trial_id,
            trial_number=trial_number,
            error=format_error_chain(exc),
        )


def _publish_failure_artifacts(
    *,
    run_id: str,
    active: Session,
    trial_id: str,
    trial_number: int | None,
    slug: str,
    strategy: str,
    timing: TimingRecorder,
    exc: BaseException,
    trial_dir: Path | None,
) -> None:
    has_agent_proposal = log_agent_proposal(run_id=run_id, trial_dir=trial_dir)
    log_failure_artifacts(
        failure=RunnerFailureReport(
            runner_kind="trial",
            phase=timing.last_phase or "unknown",
            exception=ExceptionSnapshot.from_exception(exc),
            run_id=run_id,
            project_name=active.project_name,
            experiment_id=active.active_experiment_id,
            trial_id=trial_id,
            trial_number=trial_number,
            trial_slug=slug,
            trial_strategy=strategy,
            trial_dir=trial_dir,
            timing=timing.snapshot(),
        ),
        has_agent_proposal=has_agent_proposal,
    )


def _start_failure_run(
    context: TrialExecutionContext,
    *,
    active: Session,
    model_cls: type[Any] | None,
    slug: str,
    strategy: str,
    trial_number: int | None,
) -> tuple[str, str, int | None, str, str] | None:
    slug = _failure_slug(context, model_cls=model_cls, slug=slug)
    if not slug:
        return None
    strategy = _failure_strategy(context, slug=slug, strategy=strategy)
    mlflow_experiment.ensure(experiment_id=active.active_experiment_id)
    trial_number = trial_number or mlflow_experiment.next_trial_number(
        experiment_id=active.active_experiment_id
    )
    trial_id = f"{trial_number}_{slug}"
    run_id = mlflow_trial.start(
        slug=slug,
        strategy=strategy,
        experiment_id=active.active_experiment_id,
    )
    run_tags = {
        mlflow_tags.TRIAL_NUMBER: trial_number,
        mlflow_tags.TRIAL_ID: trial_id,
    }
    if context.metadata is not None:
        run_tags[mlflow_tags.TRIAL_TRAINING_ORIGIN] = context.metadata.training_origin
    else:
        run_tags[mlflow_tags.TRIAL_TRAINING_ORIGIN] = "project"
    mlflow_trial.set_tags(run_id, run_tags)
    mlflow_trial.log_param(run_id, mlflow_tags.TRIAL_HYPOTHESIS, _trial_hypothesis(context, slug))
    return run_id, trial_id, trial_number, slug, strategy


def _failure_slug(
    context: TrialExecutionContext,
    *,
    model_cls: type[Any] | None,
    slug: str,
) -> str:
    if slug:
        return slug
    if context.metadata is not None and context.metadata.slug:
        return context.metadata.slug
    if model_cls is not None:
        return _trial_slug(context, model_cls)
    return ""


def _failure_strategy(
    context: TrialExecutionContext,
    *,
    slug: str,
    strategy: str,
) -> str:
    if strategy:
        return strategy
    if context.metadata is not None and context.metadata.strategy:
        return context.metadata.strategy
    return slug


def _trial_hypothesis(context: TrialExecutionContext, slug: str) -> str:
    if context.metadata is not None and context.metadata.hypothesis:
        return context.metadata.hypothesis
    return f"{slug} project baseline"


def _execution_context(
    path_or_project: str | Path,
    *,
    session: Session | None = None,
    dataset_id: str | None = None,
) -> TrialExecutionContext:
    path = Path(path_or_project) if path_or_project not in (None, "") else None
    if path is not None and path.exists() and _should_execute_path(path, session):
        if session is not None:
            active = session
            verified = verify_trial_dir(active, _trial_path(path))
        else:
            metadata = _read_trial_metadata(path)
            active = _session_from_trial_path(path, metadata)
            verified = verify_trial_dir(active, _trial_path(path))
        metadata = _read_trial_metadata(verified)
        return TrialExecutionContext(
            session=active,
            trial_dir=verified,
            metadata=metadata,
            dataset_id=dataset_id,
        )
    return TrialExecutionContext(
        session=_resolve_session(path_or_project, session=session),
        dataset_id=dataset_id,
    )


def _should_execute_path(path: Path, session: Session | None) -> bool:
    trial_path = _trial_path(path)
    if session is None:
        return (trial_path / "metadata.json").exists()
    if trial_path.parent.name == "projects" and trial_path.name == session.project_name:
        return False
    return True


def _trial_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate.parent if candidate.is_file() else candidate


def _read_trial_metadata(path: str | Path) -> TrialMetadata:
    metadata_path = _trial_path(path) / "metadata.json"
    if not metadata_path.is_file():
        raise ProjectError(f"trial metadata not found at {metadata_path}")
    try:
        return TrialMetadata.read(metadata_path)
    except JSONDecodeError as exc:
        raise ProjectError(f"invalid trial metadata JSON at {metadata_path}: {exc}") from exc
    except ValueError as exc:
        message = str(exc)
        if message == "trial metadata must be a JSON object":
            raise ProjectError(f"{message}: {metadata_path}") from exc
        raise ProjectError(f"invalid trial metadata at {metadata_path}: {message}") from exc


def _session_from_trial_path(path: Path, metadata: TrialMetadata) -> Session:
    root = find_repo_root(path)
    dry_run, namespace = _route_flags_from_trial_path(path, metadata)
    return use_project(
        metadata.project_name,
        repo_root=root,
        dry_run=dry_run,
        namespace=namespace,
        experiment_id=metadata.experiment_id,
    )


def _route_flags_from_trial_path(path: Path, metadata: TrialMetadata) -> tuple[bool, str]:
    trial_path = _trial_path(path).resolve()
    project_dir = find_repo_root(trial_path) / "projects" / metadata.project_name
    try:
        route_parts = trial_path.relative_to(project_dir / "experiments").parts[:-1]
    except ValueError as exc:
        raise ProjectError(
            f"trial directory {trial_path} is not under {project_dir / 'experiments'}"
        ) from exc
    dry_run = "dry_run" in route_parts
    prefix = []
    for part in route_parts:
        if part == "dry_run":
            break
        if part == metadata.project_name:
            break
        prefix.append(part)
    return dry_run, "/".join(prefix)


def _resolve_session(path_or_project: str | Path, *, session: Session | None) -> Session:
    if session is not None:
        return session
    if path_or_project in (None, ""):
        return active_project_session()
    path = Path(path_or_project)
    if path.exists():
        project_dir = path.parent if path.name == "config.py" else path
        if project_dir.parent.name != "projects":
            raise ProjectError(f"cannot infer project root from {path}")
        return use_project(project_dir.name, repo_root=project_dir.parents[1])
    return use_project(str(path_or_project))


def _active_experiment_id_or_none(active: Session) -> str | None:
    try:
        return active.active_experiment_id
    except ProjectError:
        return None


def _load_model_class(context: TrialExecutionContext | Session) -> type[Any]:
    if isinstance(context, Session):
        return _load_project_model_class(context)
    if context.trial_dir is not None:
        return _load_trial_model_class(context)
    return _load_project_model_class(context.session)


def _load_project_model_class(active: Session) -> type[Any]:
    module_name = f"{active.config.project_package}.model"
    root = str(active.config.repo_root.resolve())
    old_path = list(sys.path)
    try:
        sys.path[:] = [root, *[item for item in sys.path if item != root]]
        importlib.invalidate_caches()
        module = importlib.import_module(module_name)
    finally:
        sys.path[:] = old_path
    model_cls = getattr(module, "MODEL_CLASS", None)
    if model_cls is None:
        raise ProjectError(f"{module_name} must define MODEL_CLASS")
    if not isinstance(model_cls, type):
        raise TypeError(f"{module_name}.MODEL_CLASS must be a class")
    return model_cls


def _load_trial_model_class(context: TrialExecutionContext) -> type[Any]:
    if context.trial_dir is None:
        raise ProjectError("trial_dir is required to load a folder model")
    model_path = context.trial_dir / "model.py"
    if not model_path.is_file():
        raise ProjectError(f"trial model not found at {model_path}")
    module_name = f"_automl_trial_model_{abs(hash(model_path.resolve()))}"
    spec = importlib.util.spec_from_file_location(module_name, model_path)
    if spec is None or spec.loader is None:
        raise ProjectError(f"failed to load trial model module from {model_path}")
    module = importlib.util.module_from_spec(spec)
    root = str(context.session.config.repo_root.resolve())
    old_path = list(sys.path)
    try:
        sys.path[:] = [root, *[item for item in sys.path if item != root]]
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = old_path
    model_cls = getattr(module, "MODEL_CLASS", None) or getattr(module, "Model", None)
    if model_cls is None:
        raise ProjectError(f"{model_path} must define MODEL_CLASS or Model")
    if not isinstance(model_cls, type):
        raise TypeError(f"{model_path} model export must be a class")
    return model_cls


def _trial_slug(context: TrialExecutionContext, model_cls: type[Any]) -> str:
    if context.metadata is not None and context.metadata.slug:
        return context.metadata.slug
    return trial_slug(model_cls())


def _trial_strategy(context: TrialExecutionContext, slug: str) -> str:
    if context.metadata is not None and context.metadata.strategy:
        return context.metadata.strategy
    return slug


def _trial_data_contract(
    *,
    active: Session,
    run_id: str,
    trial_id: str,
    loaded_fit,
) -> TrialDataContract:
    run_config = active.config.require_run_config()
    # One full-frame load (a local cache hit once plan 2 lands); every split's
    # slice is derived in memory. Hash semantics are unchanged: hash the
    # *sliced* frame, exactly as the per-split loads did.
    full = data.load_dataset_by_id(loaded_fit.id, session=active)
    slices: list[SliceContract] = []
    for name, predicate in run_config.splits.predicates.items():
        sliced = full.df[predicate.mask(full.df)].reset_index(drop=True)
        slices.append(
            SliceContract(
                name=name,
                predicate=predicate.to_dict(),
                n_rows=len(sliced),
                content_hash=dataframe_content_hash(sliced),
            )
        )
        # Rebinding alone would keep the previous slice alive while the next
        # mask/copy evaluates; the del caps the loop at one resident slice.
        del sliced
    contract = TrialDataContract(
        trial=TrialRef(
            project_name=active.project_name,
            experiment_id=active.active_experiment_id,
            trial_id=trial_id,
            run_id=run_id,
        ),
        dataset=DatasetRef.from_dataset(full.dataset),
        splits={
            name: predicate.to_dict() for name, predicate in run_config.splits.predicates.items()
        },
        slices=tuple(slices),
    )
    del full
    return contract


def _try_log_train_eval(
    *,
    run_id: str,
    active: Session,
    model,
    dataset_id: str,
    train_split: str,
    feature_registry,
) -> None:
    try:
        eval_dataset, _ = prepare_eval_dataset(
            session=active,
            dataset_id=dataset_id,
            split=train_split,
        )
        evaluate(
            session=active,
            model_run_id=run_id,
            eval_dataset_id=eval_dataset.id,
            label=train_split,
            set_as_primary_label=False,
            _model=model,
            _model_feature_registry=feature_registry,
        )
    except Exception:
        return


__all__ = ["TrialResult", "run_trial"]
