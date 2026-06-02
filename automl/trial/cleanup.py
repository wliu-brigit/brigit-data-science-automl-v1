"""Trial cleanup wrapper."""

from __future__ import annotations

from automl.mlflow import client as mlflow_client
from automl.mlflow import routing as mlflow_routing
from automl.mlflow import trial as mlflow_trial
from automl.project import Session, session as active_project_session
from automl.project import cleanup as project_cleanup


def delete(
    run_id: str,
    *,
    apply: bool = False,
    hard_delete: bool = False,
    backend_store_uri: str = "",
    artifacts_destination: str = "",
    session: Session | None = None,
):
    active = session if session is not None else active_project_session()
    with mlflow_client.bound_for(active, experiment_id=active.active_experiment_id):
        parent = mlflow_trial.get_parent_experiment(run_id)
        expected_route = mlflow_routing.experiment_route_for(
            project_name=active.project_name,
            experiment_id=parent.experiment_id,
            namespace=active.namespace,
            dry_run=active.dry_run,
        )
    if parent.project_name != active.project_name or parent.mlflow_experiment_name != expected_route:
        raise ValueError("trial run is not in the current session universe")
    return project_cleanup.delete(
        run_id,
        scope="trial",
        apply=apply,
        hard_delete=hard_delete,
        backend_store_uri=backend_store_uri,
        artifacts_destination=artifacts_destination,
        session=active,
        parent_experiment=parent,
    )


__all__ = ["delete"]
