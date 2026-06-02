"""Experiment cleanup wrapper."""

from __future__ import annotations

from automl.project import Session
from automl.project import cleanup as project_cleanup


def delete(
    experiment_id: str,
    *,
    apply: bool = False,
    hard_delete: bool = False,
    backend_store_uri: str = "",
    artifacts_destination: str = "",
    session: Session | None = None,
):
    return project_cleanup.delete(
        experiment_id,
        scope="experiment",
        apply=apply,
        hard_delete=hard_delete,
        backend_store_uri=backend_store_uri,
        artifacts_destination=artifacts_destination,
        session=session,
    )


__all__ = ["delete"]
