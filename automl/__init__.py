"""brigit-automl — an agent-driven AutoML library for tabular ML."""

from importlib import import_module
from typing import Any

__version__ = "0.0.0.dev0"  # set properly at cutover

_LAZY_EXPORTS = {
    "Dataset": ("automl.data.dataset", "Dataset"),
    "Experiment": ("automl.experiment.store", "Experiment"),
    "Model": ("automl.model.base", "BaseModel"),
    "ProjectConfig": ("automl.project", "ProjectConfig"),
    "Proposal": ("automl.agent.proposal", "Proposal"),
    "Session": ("automl.project", "Session"),
    "active_session": ("automl.project", "active_session"),
    "clear_session": ("automl.project", "clear_session"),
    "session": ("automl.project", "session"),
    "update_session": ("automl.project", "update_session"),
    "use_project": ("automl.project", "use_project"),
}

__all__ = [
    "Dataset",
    "ProjectConfig",
    "Proposal",
    "Model",
    "Session",
    "active_session",
    "clear_session",
    "Experiment",
    "session",
    "update_session",
    "use_project",
]


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
