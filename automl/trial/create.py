"""Trial folder creation."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from automl.errors import ProjectError, StorageError
from automl.mlflow import experiment as mlflow_experiment
from automl.mlflow import trial as mlflow_trial
from automl.project import Session, session as active_project_session
from automl.trial.metadata import ModelSource, SeedSelection, TrialMetadata
from automl.trial.paths import trial_dir
from automl.trial.template import TEMPLATE
from automl.utils.slug import SLUG_RE


@dataclass(frozen=True)
class _ResolvedSeed:
    selection: SeedSelection
    source: str


def create(
    slug: str | None = None,
    strategy: str | None = None,
    *,
    hypothesis: str = "",
    seed: str | None = None,
    model_source: Path | None = None,
    training_origin: str = "automl",
    proposal: dict[str, Any] | None = None,
    session: Session | None = None,
) -> Path:
    """Create a routed local trial folder for later runner execution."""

    resolved_slug = slug or _proposal_value(proposal, "slug", default=None)
    resolved_strategy = strategy or _proposal_value(proposal, "strategy", default=None)
    resolved_hypothesis = hypothesis or _proposal_value(proposal, "hypothesis", default="")
    resolved_seed = seed or _proposal_value(proposal, "seed_hint", default=None)
    if not resolved_slug:
        raise ValueError("trial create requires slug or proposal_json.slug")
    if not resolved_strategy:
        raise ValueError("trial create requires --strategy or proposal_json.strategy")
    return _create_resolved(
        slug=resolved_slug,
        strategy=resolved_strategy,
        hypothesis=resolved_hypothesis or "",
        seed=resolved_seed,
        model_source=model_source,
        training_origin=training_origin,
        proposal=proposal,
        session=session,
    )


def _create_resolved(
    slug: str,
    strategy: str,
    *,
    hypothesis: str = "",
    seed: str | None = None,
    model_source: Path | None = None,
    training_origin: str = "automl",
    proposal: dict[str, Any] | None = None,
    session: Session | None = None,
) -> Path:
    active = session or active_project_session()
    slug = _validate_slug(slug)
    strategy = strategy.strip()
    if not strategy:
        raise ValueError("trial create requires --strategy or proposal_json.strategy")
    training_origin = _normalize_training_origin(training_origin)
    _ensure_config_exists(active)

    if model_source is not None and seed is not None:
        raise ValueError("use either model_source or seed, not both")

    target = trial_dir(active, slug)
    if target.exists():
        raise FileExistsError(
            f"local trial draft already exists: {target}. "
            "Choose a different slug or remove the existing draft."
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir()

    try:
        (target / "run.py").write_text(TEMPLATE, encoding="utf-8")
        resolved_seed = _resolve_seed(active, seed) if seed is not None else None
        if model_source is not None:
            shutil.copy2(Path(model_source), target / "model.py")
        elif resolved_seed is not None:
            (target / "model.py").write_text(resolved_seed.source, encoding="utf-8")

        metadata = TrialMetadata(
            slug=slug,
            strategy=strategy,
            hypothesis=hypothesis,
            training_origin=training_origin,
            created_at=datetime.now(UTC).isoformat(),
            project_name=active.project_name,
            project_package=active.config.project_package,
            experiment_id=active.active_experiment_id,
            seed=resolved_seed.selection if resolved_seed is not None else None,
        )
        metadata.write(target / "metadata.json")
        if proposal is not None:
            proposal_dir = target / "proposal"
            proposal_dir.mkdir()
            (proposal_dir / "proposal.json").write_text(
                json.dumps(proposal, indent=2, default=str),
                encoding="utf-8",
            )
    except Exception:
        shutil.rmtree(target, ignore_errors=True)
        raise

    return target


def _proposal_value(
    proposal: dict[str, Any] | None,
    key: str,
    *,
    default: str | None = "",
) -> str | None:
    if proposal is None:
        return default
    value = proposal.get(key)
    if value is None:
        return default
    text = str(value).strip()
    return text or default


def _validate_slug(slug: str) -> str:
    value = slug.strip()
    if not SLUG_RE.fullmatch(value):
        raise ValueError("slug must be lowercase snake_case and start with a letter")
    return value


def _normalize_training_origin(training_origin: str) -> str:
    value = training_origin.strip().lower()
    if value not in {"automl", "human"}:
        raise ValueError("training_origin must be 'automl' or 'human'")
    return value


def _ensure_config_exists(active: Session) -> None:
    if not active.config.config_path.is_file():
        raise ProjectError(f"{active.config.config_path} not found")


def _resolve_seed(active: Session, requested: str | None) -> _ResolvedSeed | None:
    selector = (requested or "auto").strip() or "auto"
    if selector not in {"auto", "best", "latest"} and not (
        selector.startswith("strategy:") and selector.split(":", 1)[1].strip()
    ):
        raise ValueError("seed must be one of 'auto', 'best', 'latest', or 'strategy:<name>'")

    summary = _select_seed_summary(active, selector)
    if summary is None:
        return None
    try:
        source = mlflow_trial.artifacts.load_model_source(summary.run_id)
    except StorageError as exc:
        raise FileNotFoundError(
            "selected seed run has no recoverable model.py source: "
            f"run_id={summary.run_id} trial_id={summary.slug}"
        ) from exc

    source_ref = ModelSource(source="mlflow", artifact_path="model/code")
    selection = SeedSelection(
        selector=selector,
        run_id=summary.run_id,
        trial_id=summary.slug,
        metric_name=summary.primary_metric_name,
        metric_value=summary.primary_metric_value,
        strategy=summary.strategy,
        model_source=source_ref,
    )
    return _ResolvedSeed(selection=selection, source=source)


def _select_seed_summary(active: Session, selector: str):
    experiment_id = active.active_experiment_id
    if selector in {"auto", "best"}:
        metric = _primary_metric_or_default(active)
        rows = mlflow_experiment.top_n_by_metric(metric, n=1, experiment_id=experiment_id)
        return rows[0] if rows else None
    if selector == "latest":
        rows = mlflow_experiment.list_trials(
            experiment_id=experiment_id,
            limit=1,
            status="FINISHED",
        )
        return rows[0] if rows else None
    strategy = selector.split(":", 1)[1]
    rows = mlflow_experiment.search_trials(
        f"params.trial.strategy = '{strategy}' and tags.trial.status = 'FINISHED'",
        experiment_id=experiment_id,
        max_results=1,
    )
    return rows[0] if rows else None


def _primary_metric_or_default(active: Session) -> str:
    try:
        run_config = active.config.require_run_config()
        return f"eval.{run_config.eval_split}.{active.config.primary_metric}"
    except Exception:
        return "eval.test.auc"


__all__ = ["create"]
