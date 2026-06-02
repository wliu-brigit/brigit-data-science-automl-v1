from __future__ import annotations

from types import SimpleNamespace

import pytest

from automl.mlflow import client
from automl.mlflow import routing as mlflow_routing
from automl.project import ProjectConfig, Session
from automl.runner import session_lock

pytestmark = pytest.mark.unit


def test_session_lock_acquire_release_and_context(tmp_path):
    lock_id = session_lock.acquire(
        project_root=tmp_path,
        route="qa/dry_run/demo/session-exp",
        session_id="session-1",
    )

    assert lock_id
    assert session_lock.is_locked(project_root=tmp_path, route="qa/dry_run/demo/session-exp")
    with pytest.raises(RuntimeError, match="LOCKED"):
        session_lock.acquire(
            project_root=tmp_path,
            route="qa/dry_run/demo/session-exp",
            session_id="session-2",
        )

    session_lock.release(project_root=tmp_path, session_id="session-1", lock_id=lock_id)
    assert not session_lock.is_locked(project_root=tmp_path, route="qa/dry_run/demo/session-exp")

    with session_lock.session_lock(
        project_root=tmp_path,
        route="demo/session-exp",
        session_id="session-3",
    ) as context_lock_id:
        assert context_lock_id
        assert session_lock.is_locked(project_root=tmp_path, route="demo/session-exp")
    assert not session_lock.is_locked(project_root=tmp_path, route="demo/session-exp")


@pytest.fixture(autouse=True)
def clear_mlflow_binding():
    client.clear()
    yield
    client.clear()


def test_route_for_session_composes_namespace_and_dry_run():
    active = SimpleNamespace(
        namespace="qa",
        dry_run=True,
        project_name="demo",
        active_experiment_id="session-exp",
    )
    assert session_lock.route_for_session(active) == "qa/dry_run/demo/session-exp"


def test_session_lock_acquire_for_session_builds_route_payload(tmp_path):
    active = SimpleNamespace(
        namespace="qa",
        dry_run=True,
        project_name="demo",
        active_experiment_id="exp",
        config=SimpleNamespace(repo_root=tmp_path),
    )

    payload = session_lock.acquire_for_session(active, session_id="session-1")

    assert payload["status"] == "acquired"
    assert payload["session_id"] == "session-1"
    assert payload["route"] == "qa/dry_run/demo/exp"
    assert payload["lock_id"]

    session_lock.release_for_session(
        active,
        session_id="session-1",
        lock_id=payload["lock_id"],
    )


def test_route_for_session_matches_current_mlflow_experiment_route(tmp_path):
    active = Session(
        config=ProjectConfig(
            project_name="demo",
            repo_root=tmp_path,
            project_dir=tmp_path / "projects" / "demo",
            gcs_bucket="automl-test-bucket",
            gcs_prefix="automl-root",
            mlflow_tracking_uri=(tmp_path / "mlruns").as_uri(),
        ),
        experiment_id="session-exp",
        namespace="qa",
        dry_run=True,
    )
    client.bind(
        tracking_uri=active.config.mlflow_tracking_uri,
        bucket=active.config.gcs_bucket,
        gcs_prefix=active.config.gcs_prefix,
        project_name=active.project_name,
        experiment_id=active.active_experiment_id,
        dry_run=active.dry_run,
        namespace=active.namespace,
    )

    assert session_lock.route_for_session(active) == mlflow_routing.experiment_route()
    assert session_lock.route_for_session(active) == "qa/dry_run/demo/session-exp"
