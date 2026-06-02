import pytest

from automl.mlflow import client, experiment, tags, trial
from automl.trial.types import TrialStatus, TrialSummary

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


def test_list_trials_returns_typed_summaries_newest_first(bound_file_mlflow):
    experiment.ensure()
    with trial.active(slug="unscored", strategy="baseline") as unscored_id:
        trial.set_tags(unscored_id, {tags.TRIAL_NUMBER: "1", tags.TRIAL_ID: "1_unscored"})
    with trial.active(slug="scored", strategy="tree") as scored_id:
        trial.set_tags(scored_id, {tags.TRIAL_NUMBER: "2", tags.TRIAL_ID: "2_scored"})
        trial.log_metric(scored_id, "eval.test.auc", 0.81)
        trial.log_metric(scored_id, "auc", 0.81)
        trial.set_tag(scored_id, tags.EVAL_PRIMARY_LABEL, "test")

    rows = experiment.list_trials()

    assert [row.run_id for row in rows] == [scored_id, unscored_id]
    assert all(isinstance(row, TrialSummary) for row in rows)
    assert rows[0].status is TrialStatus.FINISHED
    assert rows[0].primary_metric_name == ""
    assert rows[0].primary_metric_value is None


def test_top_n_by_metric_filters_training_origin_and_returns_typed_rows(bound_file_mlflow):
    experiment.ensure()
    with trial.active(slug="automl", strategy="baseline") as run_a:
        trial.log_metric(run_a, "eval.test.auc", 0.7)
        trial.set_tag(run_a, "trial.origin", "automl")
    with trial.active(slug="human", strategy="baseline") as run_b:
        trial.log_metric(run_b, "eval.test.auc", 0.9)
        trial.set_tag(run_b, "trial.origin", "human")

    rows = experiment.top_n_by_metric("eval.test.auc", n=5, training_origin="automl")

    assert [row.run_id for row in rows] == [run_a]
    assert rows[0].primary_metric_name == "eval.test.auc"
    assert rows[0].primary_metric_value == pytest.approx(0.7)


def test_training_origin_all_is_unfiltered(bound_file_mlflow):
    experiment.ensure()
    with trial.active(slug="automl", strategy="baseline") as run_a:
        trial.log_metric(run_a, "eval.test.auc", 0.7)
        trial.set_tag(run_a, "trial.origin", "automl")
    with trial.active(slug="human", strategy="baseline") as run_b:
        trial.log_metric(run_b, "eval.test.auc", 0.9)
        trial.set_tag(run_b, "trial.origin", "human")

    listed = experiment.list_trials(training_origin="all")
    ranked = experiment.top_n_by_metric("eval.test.auc", n=5, training_origin="all")

    assert {row.run_id for row in listed} == {run_a, run_b}
    assert [row.run_id for row in ranked] == [run_b, run_a]
