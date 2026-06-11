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
from automl.project.checks import snowflake_connection

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


def _snowflake_session(monkeypatch, tmp_path, *, with_sql_files: bool = True):
    """A live-validation session over a Snowflake-backed project, GCS/MLflow mocked."""
    from automl.data import SnowflakeSource
    from automl.mlflow import client as mlflow_client
    from automl.utils.io import gcs

    monkeypatch.setattr(gcs, "write_json", lambda *args, **kwargs: None)
    monkeypatch.setattr(gcs, "read_json", lambda *args, **kwargs: {})
    monkeypatch.setattr(gcs, "delete_prefix", lambda *args, **kwargs: 0)
    monkeypatch.setattr(mlflow_client, "check_connection", lambda tracking_uri: None)

    session = _session(tmp_path)
    if with_sql_files:
        queries = session.config.project_dir / "data" / "queries"
        queries.mkdir(parents=True, exist_ok=True)
        (queries / "base_table.sql").write_text("SELECT 1\n", encoding="utf-8")
        (queries / "training_data.sql").write_text("SELECT 1\n", encoding="utf-8")
    snowflake_spec = DataSpec(
        source=SnowflakeSource(
            base_table="demo.table",
            base_table_sql="data/queries/base_table.sql",
            training_data_sql="data/queries/training_data.sql",
            unique_key="row_id",
        )
    )
    object.__setattr__(session.config, "data_spec", snowflake_spec)
    return session


def test_validate_project_live_snowflake_missing_env_errors_listing_names(
    monkeypatch, tmp_path
):
    for name in ("SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER", "SNOWFLAKE_PASSWORD"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SNOWFLAKE_USER", "u")
    session = _snowflake_session(monkeypatch, tmp_path)

    report = validate_project(session=session, live=True)

    issues = {issue.check: issue for issue in report.issues}
    snowflake = issues["project.connections.snowflake"]
    assert report.passed is False
    assert snowflake.level == "error"
    assert "SNOWFLAKE_ACCOUNT" in snowflake.message
    assert "SNOWFLAKE_PASSWORD" in snowflake.message
    assert "SNOWFLAKE_USER" not in snowflake.message


def test_validate_project_live_snowflake_probe_ok_yields_no_issue(monkeypatch, tmp_path):
    for name in ("SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER", "SNOWFLAKE_PASSWORD"):
        monkeypatch.setenv(name, "x")
    monkeypatch.setattr("automl.utils.io.snowflake.check_connection", lambda: None)
    session = _snowflake_session(monkeypatch, tmp_path)

    report = validate_project(session=session, live=True)

    assert "project.connections.snowflake" not in {issue.check for issue in report.issues}


def test_validate_project_live_snowflake_connection_failure_surfaces_driver_error(
    monkeypatch, tmp_path
):
    for name in ("SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER", "SNOWFLAKE_PASSWORD"):
        monkeypatch.setenv(name, "x")

    def boom():
        raise RuntimeError("250001: account is disabled")

    monkeypatch.setattr("automl.utils.io.snowflake.check_connection", boom)
    session = _snowflake_session(monkeypatch, tmp_path)

    report = validate_project(session=session, live=True)

    issues = {issue.check: issue for issue in report.issues}
    snowflake = issues["project.connections.snowflake"]
    assert snowflake.level == "error"
    assert "250001: account is disabled" in snowflake.message  # driver error verbatim


def test_validate_project_live_snowflake_missing_sql_file_errors(monkeypatch, tmp_path):
    for name in ("SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER", "SNOWFLAKE_PASSWORD"):
        monkeypatch.setenv(name, "x")
    monkeypatch.setattr("automl.utils.io.snowflake.check_connection", lambda: None)
    session = _snowflake_session(monkeypatch, tmp_path, with_sql_files=False)

    report = validate_project(session=session, live=True)

    issues = [issue for issue in report.issues if issue.check == "project.connections.snowflake"]
    assert {issue.level for issue in issues} == {"error"}
    messages = "\n".join(issue.message for issue in issues)
    assert "base_table_sql" in messages
    assert "training_data_sql" in messages


class _SourceSf:
    kind = "snowflake"
    base_table_sql = "sql/base.sql"
    training_data_sql = "sql/train.sql"


class _DataSpecSf:
    source = _SourceSf()


class _RunConfigSkip:
    skip_snowflake_live_check = True


class _SkippingConfig:
    data_spec = _DataSpecSf()
    run_config = _RunConfigSkip()
    project_dir = Path(".")


def _sf_issues(config, monkeypatch, *, probe=None, sql_exists=True):
    from automl.utils.io import snowflake as sf

    monkeypatch.setattr(sf, "missing_env", lambda: [])
    monkeypatch.setattr(Path, "exists", lambda self: sql_exists)
    calls = []
    monkeypatch.setattr(sf, "check_connection", lambda: calls.append(1))
    found = list(snowflake_connection(config=config, probe=probe))
    return found, calls


def test_config_flag_skips_probe_with_warning(monkeypatch):
    issues, probe_calls = _sf_issues(_SkippingConfig(), monkeypatch)
    assert probe_calls == []
    assert any(
        issue.level == "warning" and "skipped" in issue.message for issue in issues
    )


def test_probe_true_overrides_config_flag(monkeypatch):
    issues, probe_calls = _sf_issues(_SkippingConfig(), monkeypatch, probe=True)
    assert probe_calls == [1]
    assert not any("skipped" in issue.message for issue in issues)


def test_probe_false_skips_even_without_config_flag(monkeypatch):
    class _NoFlagConfig(_SkippingConfig):
        run_config = None

    issues, probe_calls = _sf_issues(_NoFlagConfig(), monkeypatch, probe=False)
    assert probe_calls == []
    assert any("skipped" in issue.message for issue in issues)


def test_env_and_sql_checks_still_run_when_skipping(monkeypatch):
    issues, _ = _sf_issues(_SkippingConfig(), monkeypatch, sql_exists=False)
    assert any(
        issue.level == "error" and "file not found" in issue.message for issue in issues
    )


def test_validate_project_wraps_crashed_domain_checks(monkeypatch, tmp_path):
    from automl.project import checks as project_checks

    def boom(**kwargs):
        raise RuntimeError("project check exploded")

    monkeypatch.setattr(project_checks, "config_required_fields", boom)

    report = validate_project(session=_session(tmp_path))

    assert report.passed is False
    assert report.issues[0].check == "project.config_required_fields.crashed"
    assert "project check exploded" in report.issues[0].message
