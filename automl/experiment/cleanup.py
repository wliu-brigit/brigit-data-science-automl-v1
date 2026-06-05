"""Experiment cleanup wrapper."""

from __future__ import annotations

from automl.project import Session
from automl.project import cleanup as project_cleanup


def delete(
    experiment_id: str,
    *,
    apply: bool = False,
    session: Session | None = None,
):
    return project_cleanup.delete(
        experiment_id,
        scope="experiment",
        apply=apply,
        session=session,
    )


__all__ = ["delete"]
