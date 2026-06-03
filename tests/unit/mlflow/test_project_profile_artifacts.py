import json

import pytest

from automl.mlflow import client
from automl.mlflow import project as mlflow_project
from automl.mlflow.project import artifacts

pytestmark = pytest.mark.unit


@pytest.fixture
def bound_file_mlflow(tmp_path):
    client.clear()
    client.bind(
        tracking_uri=(tmp_path / "mlruns").as_uri(),
        bucket="",
        gcs_prefix="automl-root",
        project_name="demo",
        experiment_id="baseline",
    )
    yield
    client.clear()


def test_ensure_overview_creates_project_overview_run(bound_file_mlflow):
    overview = mlflow_project.ensure_overview()

    assert overview.project_name == "demo"
    assert overview.created_at
    experiment = client.raw().get_experiment_by_name("demo/000_overview")
    assert experiment is not None
    runs = client.raw().search_runs([experiment.experiment_id])
    assert len(runs) == 1
    assert runs[0].data.tags["run.kind"] == "project_overview"
    assert runs[0].data.tags["mlflow.runName"] == "000_overview"


def test_project_log_json_writes_loose_json_to_project_overview(bound_file_mlflow):
    artifacts.log_json("debug/sample", {"rows": 4})

    experiment = client.raw().get_experiment_by_name("demo/000_overview")
    assert experiment is not None
    runs = client.raw().search_runs([experiment.experiment_id])
    assert len(runs) == 1
    local_path = client.raw().download_artifacts(runs[0].info.run_id, "debug/sample.json")

    with open(local_path, encoding="utf-8") as handle:
        assert json.load(handle) == {"rows": 4}
