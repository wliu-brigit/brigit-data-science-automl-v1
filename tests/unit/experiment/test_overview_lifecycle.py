import pytest

from automl.experiment import Experiment, ExperimentOverview, create
from automl.mlflow import client, project
from automl.project import ProjectConfig, Session

pytestmark = pytest.mark.unit


def test_experiment_overview_from_dict_strips_unknown_fields():
    overview = ExperimentOverview.from_dict(
        {
            "experiment_id": "baseline",
            "project_name": "home_credit",
            "created_at": "2026-05-28T00:00:00Z",
            "dry_run": True,
            "future": "ignored",
        }
    )

    assert overview.experiment_id == "baseline"
    assert overview.dry_run is True
    assert Experiment is ExperimentOverview


def test_create_returns_experiment_overview(tmp_path):
    client.clear()
    config = ProjectConfig(
        project_name="home_credit",
        repo_root=tmp_path,
        project_dir=tmp_path / "projects" / "home_credit",
        gcs_prefix="automl-root",
        mlflow_tracking_uri=(tmp_path / "mlruns").as_uri(),
    )
    active = Session(config=config, experiment_id="baseline", dry_run=True)
    client.bind(
        tracking_uri=config.mlflow_tracking_uri,
        bucket="",
        gcs_prefix=config.gcs_prefix,
        project_name=config.project_name,
        experiment_id="baseline",
        dry_run=True,
    )

    overview = create(session=active)

    assert overview.experiment_id == "baseline"
    assert overview.project_name == "home_credit"
    assert overview.dry_run is True
    assert overview.created_at


def test_project_list_experiments_returns_logical_active_ids(tmp_path):
    client.clear()
    client.bind(
        tracking_uri=(tmp_path / "mlruns").as_uri(),
        bucket="",
        gcs_prefix="automl-root",
        project_name="home_credit",
        experiment_id="baseline",
    )
    create(experiment_id="baseline")
    create(experiment_id="second")
    mlflow_client = client.raw()
    mlflow_client.create_experiment("home_credit/000_overview")
    mlflow_client.create_experiment("home_credit/baseline/nested")
    mlflow_client.create_experiment("other_project/baseline")

    assert project.list_experiments() == ["baseline", "second"]
