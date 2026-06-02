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
        else DataSpec(source=LocalCSVSource(csv_path=Path("data.csv"), hash_key="id")),
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


def test_validate_project_passes_complete_config(tmp_path):
    from automl.validate import project

    report = project(session=_session(tmp_path))

    assert report.passed is True
    assert report.issues == []


def test_validate_project_reports_missing_recipe_fields(tmp_path):
    from automl.validate import project

    report = project(session=_session(tmp_path, partial=True))

    checks = {issue.check for issue in report.issues}
    assert report.passed is False
    assert "project.config.task" in checks
    assert "project.config.data_spec" in checks
    assert "project.config.eval_spec" in checks
    assert "project.config.run_config" in checks


def test_validate_project_reports_missing_storage_env(tmp_path):
    from automl.validate import project

    report = project(session=_session(tmp_path, missing_env=True))

    checks = {issue.check for issue in report.issues}
    assert report.passed is False
    assert "project.env.gcs_bucket" in checks
    assert "project.env.gcs_prefix" in checks
    assert "project.env.mlflow_tracking_uri" in checks


def test_validate_project_wraps_crashed_domain_checks(monkeypatch, tmp_path):
    from automl import validate
    from automl.project import checks as project_checks

    def boom(**kwargs):
        raise RuntimeError("project check exploded")

    monkeypatch.setattr(project_checks, "config_required_fields", boom)

    report = validate.project(session=_session(tmp_path))

    assert report.passed is False
    assert report.issues[0].check == "project.config_required_fields.crashed"
    assert "project check exploded" in report.issues[0].message
