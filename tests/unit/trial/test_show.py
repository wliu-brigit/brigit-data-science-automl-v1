import pytest

from automl.eval import EvalResult
from automl.mlflow import client, experiment, trial
from automl.mlflow.trial import artifacts
from automl.trial import load_model, show_trial

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
    )
    yield
    client.clear()


def test_show_trial_enriches_get_details_with_eval_results(bound_file_mlflow):
    experiment.ensure()
    with trial.active(slug="baseline", strategy="logistic") as run_id:
        result = EvalResult(
            label="test",
            eval_dataset_id="eval_123",
            eval_dataset_kind="split_view",
            predictions_uri="",
            predictions_manifest_uri="",
            augmentations_used=(),
            primary="auc",
            metrics=({"name": "auc", "value": 0.8},),
            computed_at="2026-05-28T00:00:00Z",
        )
        artifacts.write_eval(run_id, "test", result)

    details = show_trial(run_id)

    assert details.run_id == run_id
    assert details.evaluations == (result,)


def test_show_trial_returns_loaded_empty_evaluations_when_none_exist(bound_file_mlflow):
    experiment.ensure()
    with trial.active(slug="baseline", strategy="logistic") as run_id:
        pass

    assert show_trial(run_id).evaluations == ()


def test_load_model_delegates_to_packaged_model_artifact(bound_file_mlflow):
    experiment.ensure()
    payload = {"model": "round-trip"}
    with trial.active(slug="baseline", strategy="logistic") as run_id:
        artifacts.write_pickle_model(run_id, payload)

    assert load_model(run_id) == payload
