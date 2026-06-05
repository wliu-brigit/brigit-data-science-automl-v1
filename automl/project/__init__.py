"""Project domain exports."""

from importlib import import_module
from typing import Any

from .checks import validate_project
from .config import ProjectConfig
from .dependencies import allowed_dependencies, parse_dependency_name
from .metadata import find_repo_root, infer_project_name, list_projects, project_metadata
from .overview import ProjectOverview
from .predicates import Predicate, Where
from .run_config import ModelRoute, ModelsConfig, RunConfig, Splits
from .scaffold import create_project
from .session import Session, active_session, clear_session, session, update_session, use_project
from .task import BinaryClassification, Multiclass, Regression, Task

__all__ = [
    "BinaryClassification",
    "ModelRoute",
    "ModelsConfig",
    "Multiclass",
    "Predicate",
    "ProjectConfig",
    "ProjectOverview",
    "Regression",
    "RunConfig",
    "Session",
    "Splits",
    "Task",
    "Where",
    "allowed_dependencies",
    "active_session",
    "clear_session",
    "create_project",
    "delete",
    "find_repo_root",
    "infer_project_name",
    "list_projects",
    "parse_dependency_name",
    "project_metadata",
    "session",
    "update_session",
    "use_project",
    "validate_project",
]


def __getattr__(name: str) -> Any:
    if name == "delete":
        value = getattr(import_module("automl.project.cleanup"), "delete")
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
