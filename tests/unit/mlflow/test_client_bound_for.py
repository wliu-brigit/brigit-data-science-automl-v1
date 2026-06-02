from __future__ import annotations

import pytest

from automl.errors import StorageError
from automl.mlflow import client
from automl.project import ModelRoute, ModelsConfig, ProjectConfig, RunConfig, Session

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def clear_mlflow_binding():
    client.clear()
    yield
    client.clear()


def _models() -> ModelsConfig:
    route = ModelRoute("sonnet", "medium")
    return ModelsConfig(manager=route, proposer=route, coder=route)


def _session(tmp_path, *, experiment_id: str | None = "cli-exp") -> Session:
    project_dir = tmp_path / "projects" / "demo"
    project_dir.mkdir(parents=True)
    return Session(
        config=ProjectConfig(
            project_name="demo",
            repo_root=tmp_path,
            project_dir=project_dir,
            config_path=project_dir / "config.py",
            run_config=RunConfig(
                experiment_id="config-exp",
                models=_models(),
                per_trial_seconds=120,
            ),
            gcs_bucket="automl-test-bucket",
            gcs_prefix="automl-root",
            mlflow_tracking_uri=(tmp_path / "mlruns").as_uri(),
        ),
        dry_run=True,
        namespace="qa",
        experiment_id=experiment_id,
    )


def _bind_prior() -> client.Bound:
    client.bind(
        tracking_uri="file:///prior/mlruns",
        bucket="prior-bucket",
        gcs_prefix="prior-root",
        project_name="prior-project",
        experiment_id="prior-exp",
        dry_run=False,
        namespace="prior-namespace",
    )
    return client.bound()


def test_bound_for_none_does_not_create_mlflow_binding():
    with client.bound_for(None):
        with pytest.raises(StorageError, match="MLflow not bound"):
            client.bound()

    with pytest.raises(StorageError, match="MLflow not bound"):
        client.bound()


def test_bound_for_none_preserves_existing_mlflow_binding():
    prior = _bind_prior()

    with client.bound_for(None):
        assert client.bound() == prior

    assert client.bound() == prior


def test_bound_for_session_binds_project_scope_without_default_experiment(tmp_path):
    active = _session(tmp_path)

    with client.bound_for(active):
        bound = client.bound()
        assert bound.tracking_uri == active.config.mlflow_tracking_uri
        assert bound.bucket == active.config.gcs_bucket
        assert bound.gcs_prefix == active.config.gcs_prefix
        assert bound.project_name == active.project_name
        assert bound.dry_run is active.dry_run
        assert bound.namespace == active.namespace
        assert bound.experiment_id is None

    with pytest.raises(StorageError, match="MLflow not bound"):
        client.bound()


@pytest.mark.parametrize(
    ("session_experiment_id", "expected"),
    [
        ("cli-exp", "cli-exp"),
        (None, "config-exp"),
    ],
)
def test_bound_for_session_binds_active_experiment_when_requested(
    tmp_path,
    session_experiment_id,
    expected,
):
    active = _session(tmp_path, experiment_id=session_experiment_id)

    with client.bound_for(active, experiment_id=active.active_experiment_id):
        bound = client.bound()
        assert bound.project_name == active.project_name
        assert bound.experiment_id == expected


def test_bound_for_session_binds_explicit_override_and_restores_prior(tmp_path):
    prior = _bind_prior()
    active = _session(tmp_path, experiment_id=None)

    with client.bound_for(active, experiment_id="override-exp"):
        bound = client.bound()
        assert bound.project_name == active.project_name
        assert bound.experiment_id == "override-exp"

    assert client.bound() == prior
