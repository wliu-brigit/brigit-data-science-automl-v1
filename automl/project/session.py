"""Active project session management."""

from __future__ import annotations

import dataclasses
import importlib
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from automl.errors import ProjectError

from .config import ProjectConfig
from .metadata import infer_project_name


@dataclass(frozen=True)
class Session:
    """Active state for this Python process."""

    config: ProjectConfig
    dry_run: bool = False
    namespace: str = ""
    experiment_id: str | None = None

    @property
    def active_experiment_id(self) -> str:
        if self.experiment_id is not None:
            return self.experiment_id
        if self.config.run_config is None:
            raise ProjectError(
                "no experiment_id set: RUN_CONFIG missing from "
                f"{self.config.config_path} and no experiment override given"
            )
        return self.config.run_config.experiment_id

    @property
    def project_name(self) -> str:
        return self.config.project_name

    def mlflow_experiment_url(self) -> str:
        """Return *this* session's MLflow experiment UI URL.

        Computed from this session (not the globally active one), so it never
        goes stale relative to the object. Returns ``""`` for local tracking
        stores, exploration sessions with no experiment, or before the
        experiment exists in MLflow.
        """
        from automl.mlflow import client

        with client.bound_for(self, experiment_id=_bind_experiment_id(self)):
            return client.experiment_url()

    def mlflow_project_url(self) -> str:
        """Return *this* session's MLflow project (overview) UI URL.

        Computed from this session; returns ``""`` for local stores or before
        the project's overview experiment exists.
        """
        from automl.mlflow import client

        with client.bound_for(self, experiment_id=_bind_experiment_id(self)):
            return client.project_url()


_ACTIVE_SESSION: ContextVar[Session | None] = ContextVar("automl_active_session", default=None)


def use_project(
    name: str | None = None,
    *,
    repo_root: Path | None = None,
    dry_run: bool = False,
    namespace: str = "",
    experiment_id: str | None = None,
) -> Session:
    if name is None:
        name = infer_project_name(repo_root=repo_root)
    config = ProjectConfig.load(name, repo_root=repo_root)
    active = Session(
        config=config,
        dry_run=dry_run,
        namespace=namespace,
        experiment_id=experiment_id,
    )
    _ACTIVE_SESSION.set(active)
    _bind_mlflow_for(active)
    return active


def session() -> Session:
    active = _ACTIVE_SESSION.get()
    if active is None:
        raise ProjectError("no active project; call automl.use_project() first")
    return active


@contextmanager
def active_session(
    name: str | None = None,
    *,
    repo_root: Path | None = None,
    dry_run: bool = False,
    namespace: str = "",
    experiment_id: str | None = None,
) -> Iterator[Session]:
    if name is None:
        name = infer_project_name(repo_root=repo_root)
    config = ProjectConfig.load(name, repo_root=repo_root)
    active = Session(
        config=config,
        dry_run=dry_run,
        namespace=namespace,
        experiment_id=experiment_id,
    )
    token = _ACTIVE_SESSION.set(active)
    _bind_mlflow_for(active)
    try:
        yield active
    finally:
        _ACTIVE_SESSION.reset(token)
        prior = _ACTIVE_SESSION.get()
        if prior is not None:
            _bind_mlflow_for(prior)
        else:
            _clear_mlflow_binding()


def clear_session() -> None:
    _ACTIVE_SESSION.set(None)
    _clear_mlflow_binding()


def update_session(**kwargs: Any) -> Session:
    updated = dataclasses.replace(session(), **kwargs)
    _ACTIVE_SESSION.set(updated)
    _bind_mlflow_for(updated)
    return updated


def _bind_mlflow_for(active: Session) -> None:
    try:
        client = importlib.import_module("automl.mlflow.client")
    except ModuleNotFoundError as exc:
        if exc.name == "automl.mlflow.client":
            return
        raise
    bind = getattr(client, "bind", None)
    if bind is None:
        return
    bind(
        tracking_uri=active.config.mlflow_tracking_uri,
        bucket=active.config.gcs_bucket,
        gcs_prefix=active.config.gcs_prefix,
        project_name=active.config.project_name,
        experiment_id=_bind_experiment_id(active),
        dry_run=active.dry_run,
        namespace=active.namespace,
    )


def _bind_experiment_id(active: Session) -> str | None:
    if active.experiment_id is not None:
        return active.experiment_id
    if active.config.run_config is not None:
        return active.config.run_config.experiment_id
    return None


def _clear_mlflow_binding() -> None:
    try:
        client = importlib.import_module("automl.mlflow.client")
    except ModuleNotFoundError as exc:
        if exc.name == "automl.mlflow.client":
            return
        raise
    clear = getattr(client, "clear", None)
    if clear is not None:
        clear()


__all__ = [
    "Session",
    "active_session",
    "clear_session",
    "session",
    "update_session",
    "use_project",
]
