import sys
import types

import pytest

import automl
from automl.errors import ProjectError, StorageError
from automl.mlflow import client as mlflow_client
from automl.project import active_session, clear_session, session, update_session, use_project

pytestmark = pytest.mark.unit


def _write_project(tmp_path, name="demo"):
    project_dir = tmp_path / "projects" / name
    project_dir.mkdir(parents=True)
    (tmp_path / "projects" / "__init__.py").write_text("")
    (project_dir / "__init__.py").write_text("")
    (project_dir / "config.py").write_text(
        """
from automl.project import ProjectConfig, RunConfig, Splits, ModelRoute, ModelsConfig

PROJECT_CONFIG = ProjectConfig.partial(
    run_config=RunConfig(
        experiment_id="demo-exp",
        splits=Splits(train=[(0, 80)], test=[(80, 100)]),
        models=ModelsConfig(
            manager=ModelRoute("sonnet", "medium"),
            proposer=ModelRoute("sonnet", "medium"),
            coder=ModelRoute("sonnet", "medium"),
        ),
        per_trial_seconds=120,
    ),
)
"""
    )


@pytest.fixture(autouse=True)
def clean_session_and_fake_mlflow(monkeypatch):
    clear_session()
    calls = []
    fake_client = types.ModuleType("automl.mlflow.client")

    def bind(**kwargs):
        calls.append(kwargs)
        mlflow_client.bind(**kwargs)

    fake_client.bind = bind
    fake_client.clear = mlflow_client.clear
    monkeypatch.setitem(sys.modules, "automl.mlflow.client", fake_client)
    yield calls
    clear_session()


def test_session_raises_before_use_project():
    with pytest.raises(ProjectError, match="no active project"):
        session()


def test_use_project_sets_active_session_and_binds_mlflow(tmp_path, clean_session_and_fake_mlflow):
    _write_project(tmp_path)

    active = use_project(
        "demo",
        repo_root=tmp_path,
        dry_run=True,
        namespace="qa",
        experiment_id="override-exp",
    )

    assert session() is active
    assert automl.session() is active
    assert active.project_name == "demo"
    assert active.dry_run is True
    assert active.namespace == "qa"
    assert active.active_experiment_id == "override-exp"
    assert clean_session_and_fake_mlflow[-1]["project_name"] == "demo"
    assert clean_session_and_fake_mlflow[-1]["experiment_id"] == "override-exp"
    assert clean_session_and_fake_mlflow[-1]["dry_run"] is True
    assert clean_session_and_fake_mlflow[-1]["namespace"] == "qa"
    assert "artifacts_destination" not in clean_session_and_fake_mlflow[-1]


def test_update_session_replaces_active_session_and_rebinds(
    tmp_path, clean_session_and_fake_mlflow
):
    _write_project(tmp_path)
    original = use_project("demo", repo_root=tmp_path)

    updated = update_session(dry_run=True, namespace="qa")

    assert updated is not original
    assert session() is updated
    assert updated.dry_run is True
    assert updated.namespace == "qa"
    assert clean_session_and_fake_mlflow[-1]["dry_run"] is True
    assert clean_session_and_fake_mlflow[-1]["namespace"] == "qa"


def test_active_session_restores_previous_session(tmp_path, clean_session_and_fake_mlflow):
    _write_project(tmp_path, name="outer")
    _write_project(tmp_path, name="inner")
    outer = use_project("outer", repo_root=tmp_path, experiment_id="outer-exp")

    with active_session("inner", repo_root=tmp_path, experiment_id="inner-exp") as inner:
        assert session() is inner
        assert inner.active_experiment_id == "inner-exp"

    assert session() is outer
    assert clean_session_and_fake_mlflow[-1]["project_name"] == "outer"
    assert clean_session_and_fake_mlflow[-1]["experiment_id"] == "outer-exp"


def test_clear_session_removes_active_session(tmp_path):
    _write_project(tmp_path)
    use_project("demo", repo_root=tmp_path)

    clear_session()

    with pytest.raises(ProjectError):
        session()
    with pytest.raises(StorageError):
        mlflow_client.bound()


def test_active_session_without_outer_session_clears_mlflow_binding_on_exit(
    tmp_path, clean_session_and_fake_mlflow
):
    _write_project(tmp_path)

    with active_session("demo", repo_root=tmp_path):
        assert session().project_name == "demo"
        assert mlflow_client.bound().project_name == "demo"

    with pytest.raises(ProjectError):
        session()
    with pytest.raises(StorageError):
        mlflow_client.bound()


def test_session_uses_run_config_experiment_when_no_override(
    tmp_path, clean_session_and_fake_mlflow
):
    _write_project(tmp_path)

    active = use_project("demo", repo_root=tmp_path)

    assert active.active_experiment_id == "demo-exp"
    assert clean_session_and_fake_mlflow[-1]["experiment_id"] == "demo-exp"


def test_use_project_without_name_infers_single_project(tmp_path):
    _write_project(tmp_path, name="solo")

    active = use_project(repo_root=tmp_path)

    assert session() is active
    assert active.project_name == "solo"


def test_use_project_without_name_infers_from_cwd_when_multiple(tmp_path, monkeypatch):
    _write_project(tmp_path, name="alpha")
    _write_project(tmp_path, name="beta")
    monkeypatch.chdir(tmp_path / "projects" / "beta")

    active = use_project(repo_root=tmp_path)

    assert active.project_name == "beta"


def test_use_project_without_name_raises_when_ambiguous(tmp_path, monkeypatch):
    _write_project(tmp_path, name="alpha")
    _write_project(tmp_path, name="beta")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ProjectError, match="multiple projects"):
        use_project(repo_root=tmp_path)


def test_active_session_without_name_infers_single_project(tmp_path):
    _write_project(tmp_path, name="solo")

    with active_session(repo_root=tmp_path) as active:
        assert active.project_name == "solo"
        assert session() is active


def test_session_from_args_delegates_inference_to_use_project(tmp_path):
    import argparse

    from automl.cli._common import session_from_args

    _write_project(tmp_path, name="solo")
    args = argparse.Namespace(
        project=None,
        project_root=tmp_path,
        dry_run=False,
        namespace="",
        experiment_id=None,
    )

    active = session_from_args(args)

    assert active.project_name == "solo"


def test_use_project_allows_partial_config_without_experiment_for_exploration(
    tmp_path, clean_session_and_fake_mlflow
):
    project_dir = tmp_path / "projects" / "partial"
    project_dir.mkdir(parents=True)
    (tmp_path / "projects" / "__init__.py").write_text("")
    (project_dir / "__init__.py").write_text("")
    (project_dir / "config.py").write_text(
        """
from automl.project import ProjectConfig

PROJECT_CONFIG = ProjectConfig.partial()
"""
    )

    active = use_project("partial", repo_root=tmp_path)

    assert session() is active
    assert not active.config.is_complete()
    assert clean_session_and_fake_mlflow[-1]["experiment_id"] is None
