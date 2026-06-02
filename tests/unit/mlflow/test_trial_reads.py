import pytest

from automl.mlflow import client, experiment, trial
from automl.trial.types import ParentExperimentRef, TrialDetails

pytestmark = pytest.mark.unit


@pytest.fixture
def bound_file_mlflow(tmp_path):
    client.clear()
    client.bind(
        tracking_uri=(tmp_path / "mlruns").as_uri(),
        bucket="",
        gcs_prefix="automl-root",
        project_name="home_credit",
        experiment_id="baseline",
        namespace="qa",
    )
    yield
    client.clear()


def test_get_details_and_parent_experiment_return_typed_values(bound_file_mlflow):
    experiment.ensure()
    with trial.active(slug="baseline", strategy="logistic") as run_id:
        trial.log_metric(run_id, "test.auc", 0.72)
        trial.log_param(run_id, "solver", "liblinear")

    details = trial.get_details(run_id)
    parent = trial.get_parent_experiment(run_id)

    assert isinstance(details, TrialDetails)
    assert details.run_id == run_id
    assert details.metrics["test.auc"] == 0.72
    assert details.params["solver"] == "liblinear"
    assert details.params["trial.strategy"] == "logistic"
    assert details.evaluations is None
    assert isinstance(parent, ParentExperimentRef)
    assert parent.mlflow_experiment_name == "qa/home_credit/baseline"
    assert parent.project_name == "home_credit"
    assert parent.experiment_id == "baseline"
