from __future__ import annotations

from pathlib import Path

import pytest

from automl.data import DataSpec, LocalCSVSource
from automl.eval import Auc, EvalSpec
from automl.project import (
    BinaryClassification,
    ModelRoute,
    ModelsConfig,
    ProjectConfig,
    RunConfig,
    Session,
    validate_project,
)

pytestmark = pytest.mark.unit


def _session(tmp_path, *, missing_env: bool = False, partial: bool = False) -> Session:
    project_dir = tmp_path / "projects" / "demo"
    project_dir.mkdir(parents=True, exist_ok=True)
    config = ProjectConfig(
        project_name="demo",
        repo_root=tmp_path,
        project_dir=project_dir,
        config_path=project_dir / "config.py",
        instructions_path=project_dir / "PROJECT_INSTRUCTIONS.md",
        task=None if partial else BinaryClassification(target="TARGET"),
        data_spec=None
        if partial
        else DataSpec(source=LocalCSVSource(csv_path=Path("data.csv"), unique_key="id")),
        eval_spec=None if partial else EvalSpec(primary=Auc()),
        run_config=None
        if partial
        else RunConfig(
            experiment_id="baseline",
            models=ModelsConfig(
                manager=ModelRoute("sonnet", "medium"),
                proposer=ModelRoute("sonnet", "medium"),
                coder=ModelRoute("sonnet", "medium"),
            ),
            per_trial_seconds=60,
        ),
        gcs_bucket="" if missing_env else "bucket",
        gcs_prefix="" if missing_env else "prefix",
        mlflow_tracking_uri="" if missing_env else "file:///tmp/mlruns",
    )
    return Session(config=config, experiment_id="baseline")


def test_validate_project_requires_active_project():
    from automl.errors import ProjectError

    with pytest.raises(ProjectError, match="no active project"):
        validate_project()


def test_validate_project_passes_complete_config(tmp_path):
    report = validate_project(session=_session(tmp_path))

    assert report.passed is True
    assert report.issues == []


def test_validate_project_reports_missing_recipe_fields(tmp_path):
    report = validate_project(session=_session(tmp_path, partial=True))

    checks = {issue.check for issue in report.issues}
    assert report.passed is False
    assert "project.config.task" in checks
    assert "project.config.data_spec" in checks
    assert "project.config.eval_spec" in checks
    assert "project.config.run_config" in checks


def test_validate_project_reports_missing_storage_env(tmp_path):
    report = validate_project(session=_session(tmp_path, missing_env=True))

    checks = {issue.check for issue in report.issues}
    assert report.passed is False
    assert "project.env.gcs_bucket" in checks
    assert "project.env.gcs_prefix" in checks
    assert "project.env.mlflow_tracking_uri" in checks


def test_validate_project_reports_leftover_tbd_placeholders(tmp_path):

    session = _session(tmp_path)
    session.config.config_path.write_text(
        'TASK = BinaryClassification(target="<TBD_target_column>")\n'
    )

    report = validate_project(session=session)

    checks = {issue.check for issue in report.issues}
    assert report.passed is False
    assert "project.placeholders" in checks


def test_validate_project_offline_by_default_skips_connection_probes(monkeypatch, tmp_path):
    from automl.project import checks as project_checks

    def boom(**kwargs):
        raise AssertionError("connection probe must not run without live=True")

    monkeypatch.setattr(project_checks, "gcs_connection", boom)
    monkeypatch.setattr(project_checks, "mlflow_connection", boom)
    monkeypatch.setattr(project_checks, "snowflake_connection", boom)

    report = validate_project(session=_session(tmp_path))

    assert report.passed is True


def test_validate_project_live_reports_unreachable_services(monkeypatch, tmp_path):
    from automl.mlflow import client as mlflow_client
    from automl.utils.io import gcs

    def gcs_down(*args, **kwargs):
        raise ConnectionError("bucket unreachable")

    def mlflow_down(tracking_uri):
        raise ConnectionError("tracking server unreachable")

    monkeypatch.setattr(gcs, "write_json", gcs_down)
    monkeypatch.setattr(mlflow_client, "check_connection", mlflow_down)

    report = validate_project(session=_session(tmp_path), live=True)

    issues = {issue.check: issue for issue in report.issues}
    assert report.passed is False
    assert "bucket unreachable" in issues["project.connections.gcs"].message
    assert "tracking server unreachable" in issues["project.connections.mlflow"].message


def test_validate_project_live_passes_when_probes_succeed(monkeypatch, tmp_path):
    from automl.mlflow import client as mlflow_client
    from automl.utils.io import gcs

    monkeypatch.setattr(gcs, "write_json", lambda *args, **kwargs: None)
    monkeypatch.setattr(gcs, "read_json", lambda *args, **kwargs: {})
    monkeypatch.setattr(gcs, "delete_prefix", lambda *args, **kwargs: 0)
    monkeypatch.setattr(mlflow_client, "check_connection", lambda tracking_uri: None)

    report = validate_project(session=_session(tmp_path), live=True)

    assert report.passed is True
    assert report.issues == []


def test_validate_project_live_marks_snowflake_pending(monkeypatch, tmp_path):
    from automl.data import SnowflakeSource
    from automl.mlflow import client as mlflow_client
    from automl.utils.io import gcs

    monkeypatch.setattr(gcs, "write_json", lambda *args, **kwargs: None)
    monkeypatch.setattr(gcs, "read_json", lambda *args, **kwargs: {})
    monkeypatch.setattr(gcs, "delete_prefix", lambda *args, **kwargs: 0)
    monkeypatch.setattr(mlflow_client, "check_connection", lambda tracking_uri: None)

    session = _session(tmp_path)
    snowflake_spec = DataSpec(
        source=SnowflakeSource(
            base_table="demo.table",
            base_table_sql="data/queries/base_table.sql",
            training_data_sql="data/queries/training_data.sql",
            unique_key="row_id",
        )
    )
    object.__setattr__(session.config, "data_spec", snowflake_spec)

    report = validate_project(session=session, live=True)

    issues = {issue.check: issue for issue in report.issues}
    assert report.passed is True  # warning, not error
    assert issues["project.connections.snowflake"].level == "warning"
    assert "pending" in issues["project.connections.snowflake"].message


def test_validate_project_wraps_crashed_domain_checks(monkeypatch, tmp_path):
    from automl.project import checks as project_checks

    def boom(**kwargs):
        raise RuntimeError("project check exploded")

    monkeypatch.setattr(project_checks, "config_required_fields", boom)

    report = validate_project(session=_session(tmp_path))

    assert report.passed is False
    assert report.issues[0].check == "project.config_required_fields.crashed"
    assert "project check exploded" in report.issues[0].message
