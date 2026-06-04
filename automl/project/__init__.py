"""Project domain exports."""

from .checks import validate_project
from .config import ProjectConfig
from .dependencies import allowed_dependencies, parse_dependency_name
from .metadata import find_repo_root, infer_project_name, list_projects, project_metadata
from .overview import ProjectOverview
from .run_config import ModelRoute, ModelsConfig, RunConfig, Splits
from .scaffold import create_project
from .session import Session, active_session, clear_session, session, update_session, use_project
from .task import BinaryClassification, Multiclass, Regression, Task

__all__ = [
    "BinaryClassification",
    "ModelRoute",
    "ModelsConfig",
    "Multiclass",
    "ProjectConfig",
    "ProjectOverview",
    "Regression",
    "RunConfig",
    "Session",
    "Splits",
    "Task",
    "allowed_dependencies",
    "active_session",
    "clear_session",
    "create_project",
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
