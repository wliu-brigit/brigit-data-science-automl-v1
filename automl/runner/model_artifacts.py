"""Runner model artifact publishing helpers."""

from __future__ import annotations

import fnmatch
import hashlib
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

from automl.mlflow.trial import artifacts as trial_artifacts
from automl.mlflow.trial.artifacts import runner as runner_artifacts
from automl.project import Session


_CODE_BUNDLE_EXCLUDES = {
    ".git",
    ".venv",
    ".cache",
    "mlruns",
    "experiments",
    "__pycache__",
    "notebooks",
    "data",
}
_CODE_BUNDLE_EXCLUDE_PATTERNS = ("*.pyc", "*.parquet", "*.ipynb")


def log_model(
    *,
    run_id: str,
    active: Session,
    trial_dir: Path | None,
    model,
    sample,
) -> "trial_artifacts.ModelRef":
    input_example = _model_input_example(model, sample)
    with _model_code_paths(active, trial_dir) as code_paths:
        return trial_artifacts.write_model(
            run_id,
            model,
            input_example=input_example,
            code_paths=code_paths,
            metadata={
                "run_id": run_id,
                "project_name": active.project_name,
                "experiment_id": active.active_experiment_id,
            },
        )


def log_agent_proposal(*, run_id: str, trial_dir: Path | None) -> bool:
    """Mirror the generated proposal rationale under the agent domain."""
    if trial_dir is None:
        return False
    proposal_path = _proposal_path(trial_dir)
    if proposal_path is None:
        return False
    runner_artifacts.write_local_file(
        run_id,
        "agent/proposer/proposal.json",
        proposal_path,
    )
    return True


def _model_input_example(model, sample):
    registry = getattr(model, "feature_registry", None)
    if registry is not None and hasattr(registry, "select"):
        try:
            return registry.select(sample).head(min(10, len(sample)))
        except Exception:
            pass
    return sample.head(min(10, len(sample)))


@contextmanager
def _model_code_paths(active: Session, trial_dir: Path | None):
    repo_root = active.config.repo_root
    with tempfile.TemporaryDirectory() as tmp_dir:
        bundle_root = Path(tmp_dir) / "code"
        bundle_root.mkdir()
        paths: list[str] = []
        for root_name in ("automl", "projects"):
            source = repo_root / root_name
            if not source.exists():
                continue
            target = bundle_root / root_name
            shutil.copytree(
                source,
                target,
                ignore=_code_bundle_ignore,
                symlinks=False,
            )
            paths.append(str(target))
        trial_model = trial_dir / "model.py" if trial_dir is not None else None
        if trial_model is not None and trial_model.is_file():
            payload = trial_model.read_bytes()
            digest = hashlib.sha256(payload).hexdigest()[:12]
            staged = bundle_root / f"trial_model_{digest}.py"
            staged.write_bytes(payload)
            paths.append(str(staged))
        yield paths


def _code_bundle_ignore(directory: str, names: list[str]) -> set[str]:
    del directory
    ignored: set[str] = set()
    for name in names:
        if name in _CODE_BUNDLE_EXCLUDES:
            ignored.add(name)
            continue
        if any(
            fnmatch.fnmatchcase(name, pattern)
            for pattern in _CODE_BUNDLE_EXCLUDE_PATTERNS
        ):
            ignored.add(name)
    return ignored


def _proposal_path(trial_dir: Path) -> Path | None:
    for candidate in (
        trial_dir / "proposal" / "proposal.json",
        trial_dir / "proposal" / "trial_proposal.json",
    ):
        if candidate.is_file():
            return candidate
    return None


__all__ = ["log_agent_proposal", "log_model"]
