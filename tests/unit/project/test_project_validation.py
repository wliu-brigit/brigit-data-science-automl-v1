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


def _snowflake_session(
    monkeypatch,
    tmp_path,
    *,
    with_sql_files: bool = True,
    skip_snowflake_probe: bool = False,
):
    """A live-validation session over a Snowflake-backed project, GCS/MLflow mocked."""
    from automl.data import SnowflakeSource
    from automl.mlflow import client as mlflow_client
    from automl.utils.io import gcs

    monkeypatch.setattr(gcs, "write_json", lambda *args, **kwargs: None)
    monkeypatch.setattr(gcs, "read_json", lambda *args, **kwargs: {})
    monkeypatch.setattr(gcs, "delete_prefix", lambda *args, **kwargs: 0)
    monkeypatch.setattr(mlflow_client, "check_connection", lambda tracking_uri: None)

    base_run_config = RunConfig(
        experiment_id="baseline",
        models=ModelsConfig(
            manager=ModelRoute("sonnet", "medium"),
            proposer=ModelRoute("sonnet", "medium"),
            coder=ModelRoute("sonnet", "medium"),
        ),
        per_trial_seconds=60,
        skip_snowflake_live_check=skip_snowflake_probe,
    )
    session = _session(tmp_path)
    object.__setattr__(session.config, "run_config", base_run_config)
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


def test_config_flag_skips_probe_with_warning(monkeypatch, tmp_path):
    """skip_snowflake_live_check=True + probe=None → no connection attempt, one warning."""
    for name in ("SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER", "SNOWFLAKE_PASSWORD"):
        monkeypatch.setenv(name, "x")
    probe_calls = []
    monkeypatch.setattr(
        "automl.utils.io.snowflake.check_connection",
        lambda: probe_calls.append(1),
    )
    session = _snowflake_session(monkeypatch, tmp_path, skip_snowflake_probe=True)

    report = validate_project(session=session, live=True)

    sf_issues = [i for i in report.issues if i.check == "project.connections.snowflake"]
    assert probe_calls == []
    assert any(
        i.level == "warning"
        and "skipped" in i.message
        and "RUN_CONFIG.skip_snowflake_live_check" in i.message
        for i in sf_issues
    )


def test_probe_true_overrides_config_flag(monkeypatch, tmp_path):
    """probe=True forces the live probe even when skip_snowflake_live_check=True."""
    for name in ("SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER", "SNOWFLAKE_PASSWORD"):
        monkeypatch.setenv(name, "x")
    probe_calls = []
    monkeypatch.setattr(
        "automl.utils.io.snowflake.check_connection",
        lambda: probe_calls.append(1),
    )
    session = _snowflake_session(monkeypatch, tmp_path, skip_snowflake_probe=True)

    report = validate_project(session=session, live=True, probe_snowflake=True)

    assert probe_calls == [1]
    assert not any(
        "skipped" in i.message
        for i in report.issues
        if i.check == "project.connections.snowflake"
    )


def test_probe_false_skips_even_without_config_flag(monkeypatch, tmp_path):
    """probe=False skips the live probe even when skip_snowflake_live_check=False."""
    for name in ("SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER", "SNOWFLAKE_PASSWORD"):
        monkeypatch.setenv(name, "x")
    probe_calls = []
    monkeypatch.setattr(
        "automl.utils.io.snowflake.check_connection",
        lambda: probe_calls.append(1),
    )
    # skip_snowflake_probe=False → flag is False; probe=False should still skip
    session = _snowflake_session(monkeypatch, tmp_path, skip_snowflake_probe=False)

    report = validate_project(session=session, live=True, probe_snowflake=False)

    sf_issues = [i for i in report.issues if i.check == "project.connections.snowflake"]
    assert probe_calls == []
    assert any(
        i.level == "warning"
        and "skipped" in i.message
        and "probe override" in i.message
        for i in sf_issues
    )


def test_env_and_sql_checks_still_run_when_skipping(monkeypatch, tmp_path):
    """Missing SQL files produce error-level issues even when the live probe is skipped."""
    for name in ("SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER", "SNOWFLAKE_PASSWORD"):
        monkeypatch.setenv(name, "x")
    monkeypatch.setattr("automl.utils.io.snowflake.check_connection", lambda: None)
    session = _snowflake_session(
        monkeypatch, tmp_path, skip_snowflake_probe=True, with_sql_files=False
    )

    report = validate_project(session=session, live=True)

    sf_issues = [i for i in report.issues if i.check == "project.connections.snowflake"]
    assert any(i.level == "error" and "file not found" in i.message for i in sf_issues)


def test_validate_project_wraps_crashed_domain_checks(monkeypatch, tmp_path):
    from automl.project import checks as project_checks

    def boom(**kwargs):
        raise RuntimeError("project check exploded")

    monkeypatch.setattr(project_checks, "config_required_fields", boom)

    report = validate_project(session=_session(tmp_path))

    assert report.passed is False
    assert report.issues[0].check == "project.config_required_fields.crashed"
    assert "project check exploded" in report.issues[0].message
