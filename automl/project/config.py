"""Project configuration loading."""

from __future__ import annotations

import os
import re
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from automl.errors import ProjectError

from ._import import (
    INSTRUCTIONS_FILENAME,
    PROJECT_FILENAME,
    PROJECTS_DIR,
    import_project_config,
)
from .run_config import RunConfig
from .task import BinaryClassification, Multiclass, Regression, Task


PACKAGE_COMPONENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
RECIPE_FIELDS = ("task", "data_spec", "eval_spec", "run_config")


def find_repo_root(start: Path | None = None) -> Path:
    """Walk up from ``start`` (default cwd) to the directory containing ``projects/``.

    The single repo-root resolver: the CLI, skill glue scripts, and the runner
    all resolve the root through this walk so they agree from any cwd.
    """
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / PROJECTS_DIR).is_dir():
            return candidate
    raise ProjectError(f"could not find repo root containing {PROJECTS_DIR}/ from {current}")


def _load_env(repo_root: Path) -> None:
    env_path = repo_root / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(env_path, override=False)


def _normalize_column(name: str) -> str:
    normalized = re.sub(r"[^0-9A-Za-z_]+", "_", name.strip().lower())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized


@dataclass(frozen=True)
class ProjectConfig:
    """Immutable, eagerly loaded view of a project recipe and environment."""

    project_name: str = ""
    repo_root: Path = Path()
    project_dir: Path = Path()
    project_package: str = ""
    config_path: Path = Path()
    instructions_path: Path = Path()
    task: Task | None = None
    data_spec: Any | None = None
    eval_spec: Any | None = None
    run_config: RunConfig | None = None
    required_transformers: list[Any] | None = None
    gcs_bucket: str = ""
    gcs_prefix: str = ""
    mlflow_tracking_uri: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "repo_root", Path(self.repo_root))
        object.__setattr__(self, "project_dir", Path(self.project_dir))
        object.__setattr__(self, "config_path", Path(self.config_path))
        object.__setattr__(self, "instructions_path", Path(self.instructions_path))
        object.__setattr__(
            self,
            "required_transformers",
            _normalize_required_transformers(
                self.required_transformers,
                field_name="required_transformers",
                config_path=self.config_path,
            ),
        )

    @classmethod
    def partial(
        cls,
        *,
        task: Task | None = None,
        data_spec: Any | None = None,
        eval_spec: Any | None = None,
        run_config: RunConfig | None = None,
        required_transformers: list[Any] | None = None,
    ) -> "ProjectConfig":
        """Build a recipe-only config object for ``PROJECT_CONFIG`` symbols."""

        return cls(
            task=task,
            data_spec=data_spec,
            eval_spec=eval_spec,
            run_config=run_config,
            required_transformers=required_transformers,
        )

    @classmethod
    def load(cls, name: str, *, repo_root: Path | None = None) -> "ProjectConfig":
        if not name or not PACKAGE_COMPONENT_RE.fullmatch(name):
            raise ProjectError(f"invalid project name: {name!r}")

        root = Path(repo_root).resolve() if repo_root is not None else find_repo_root()
        project_dir = root / PROJECTS_DIR / name
        if not project_dir.is_dir():
            raise ProjectError(f"project {name!r} not found at {project_dir}")

        config_path = project_dir / PROJECT_FILENAME
        instructions_path = project_dir / INSTRUCTIONS_FILENAME
        _load_env(root)
        env = {
            "gcs_bucket": os.environ.get("GCS_BUCKET", ""),
            "gcs_prefix": os.environ.get("GCS_PREFIX", ""),
            "mlflow_tracking_uri": os.environ.get("MLFLOW_TRACKING_URI", ""),
        }
        if not env["mlflow_tracking_uri"]:
            warnings.warn(
                "MLFLOW_TRACKING_URI not set; MLflow will silently fall back to ./mlruns",
                stacklevel=2,
            )

        recipe = {
            "task": None,
            "data_spec": None,
            "eval_spec": None,
            "run_config": None,
            "required_transformers": [],
        }
        if config_path.exists():
            module = _import_config(root, name, config_path)
            recipe.update(_extract_recipe(module))

        _check_recipe_types(recipe, config_path)
        return cls(
            project_name=name,
            repo_root=root,
            project_dir=project_dir,
            project_package=f"{PROJECTS_DIR}.{name}",
            config_path=config_path,
            instructions_path=instructions_path,
            **recipe,
            **env,
        )

    def is_complete(self) -> bool:
        return all(getattr(self, field) is not None for field in RECIPE_FIELDS)

    def missing_fields(self) -> list[str]:
        return [field.upper() for field in RECIPE_FIELDS if getattr(self, field) is None]

    def require_task(self) -> Task:
        if self.task is None:
            raise ProjectError(f"TASK missing from {self.config_path}")
        return self.task

    def require_data_spec(self) -> Any:
        if self.data_spec is None:
            raise ProjectError(f"DATA_SPEC missing from {self.config_path}")
        return self.data_spec

    def require_eval_spec(self) -> Any:
        if self.eval_spec is None:
            raise ProjectError(f"EVAL_SPEC missing from {self.config_path}")
        return self.eval_spec

    def require_run_config(self) -> RunConfig:
        if self.run_config is None:
            raise ProjectError(f"RUN_CONFIG missing from {self.config_path}")
        return self.run_config

    @property
    def raw_target_column(self) -> str:
        return self.require_task().target

    @property
    def target_column(self) -> str:
        return _normalize_column(self.raw_target_column)

    @property
    def primary_metric(self) -> str:
        spec = self.require_eval_spec()
        if hasattr(spec, "primary_name"):
            return str(spec.primary_name)
        primary = getattr(spec, "primary", None)
        if primary is not None:
            return str(getattr(primary, "name", primary.__class__.__name__.lower()))
        raise ProjectError(f"EVAL_SPEC missing primary metric from {self.config_path}")

    @property
    def per_trial_seconds(self) -> int:
        return self.require_run_config().per_trial_seconds

    @property
    def models(self) -> Any:
        return self.require_run_config().models


def _import_config(repo_root: Path, project_name: str, config_path: Path) -> ModuleType:
    try:
        return import_project_config(repo_root, project_name)
    except Exception as exc:
        raise ProjectError(f"failed to import {config_path}: {exc}") from exc


def _extract_recipe(module: ModuleType) -> dict[str, Any]:
    project_config = getattr(module, "PROJECT_CONFIG", None)
    if project_config is None:
        raise ProjectError(f"{PROJECT_FILENAME} must define PROJECT_CONFIG")
    return _recipe_from_project_config(project_config)


def _recipe_from_project_config(value: object) -> dict[str, Any]:
    if isinstance(value, ProjectConfig):
        return {
            "task": value.task,
            "data_spec": value.data_spec,
            "eval_spec": value.eval_spec,
            "run_config": value.run_config,
            "required_transformers": value.required_transformers or [],
        }
    if isinstance(value, Mapping):
        return {
            "task": value.get("task"),
            "data_spec": value.get("data_spec"),
            "eval_spec": value.get("eval_spec"),
            "run_config": value.get("run_config"),
            "required_transformers": value.get("required_transformers", []),
        }
    raise TypeError(f"PROJECT_CONFIG must be a ProjectConfig or mapping, got {type(value).__name__}")


def _check_recipe_types(recipe: Mapping[str, Any], config_path: Path) -> None:
    task = recipe["task"]
    if task is not None and not isinstance(task, (BinaryClassification, Regression, Multiclass)):
        raise TypeError(f"{config_path} TASK must be a Task instance, got {type(task).__name__}")
    run_config = recipe["run_config"]
    if run_config is not None and not isinstance(run_config, RunConfig):
        raise TypeError(
            f"{config_path} RUN_CONFIG must be a RunConfig instance, got {type(run_config).__name__}"
        )
    required_transformers = recipe["required_transformers"]
    recipe["required_transformers"] = _normalize_required_transformers(
        required_transformers,
        field_name="REQUIRED_TRANSFORMERS",
        config_path=config_path,
    )


def _normalize_required_transformers(
    value: Any,
    *,
    field_name: str,
    config_path: Path,
) -> list[Any]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise TypeError(
            f"{config_path} {field_name} must be a list, got {type(value).__name__}"
        )
    invalid = [item for item in value if not _looks_like_required_transformer(item)]
    if invalid:
        raise TypeError(
            f"{config_path} {field_name} entries must be RequiredTransformer instances"
        )
    return list(value)


def _looks_like_required_transformer(value: object) -> bool:
    value_type = type(value)
    return (
        value_type.__module__ == "automl.model.preprocessing"
        and value_type.__name__ == "RequiredTransformer"
    )


__all__ = ["ProjectConfig"]
