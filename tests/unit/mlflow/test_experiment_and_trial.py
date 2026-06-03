from __future__ import annotations

import json

import pytest

from automl.errors import StorageError
from automl.mlflow import client, experiment, tags, trial

pytestmark = pytest.mark.unit


@pytest.fixture
def bound_file_mlflow(tmp_path):
    client.clear()
    tracking_dir = tmp_path / "mlruns"
    client.bind(
        tracking_uri=tracking_dir.as_uri(),
        bucket="",
        gcs_prefix="automl-root",
        project_name="home_credit",
        experiment_id="baseline",
    )
    yield tracking_dir.as_uri()
    client.clear()


def test_ensure_requires_an_experiment_id_when_unbound():
    client.clear()

    with pytest.raises(StorageError, match="MLflow not bound"):
        experiment.ensure()


def test_ensure_creates_routed_file_backed_experiment(bound_file_mlflow):
    experiment.ensure()

    mlflow_client = client.raw()
    created = mlflow_client.get_experiment_by_name("home_credit/baseline")

    assert created is not None
    assert created.tags["created_by"] == "brigit-automl"


def test_next_trial_number_returns_one_when_experiment_is_absent(bound_file_mlflow):
    assert experiment.next_trial_number() == 1


def test_active_dataset_round_trips_via_experiment_tag(bound_file_mlflow):
    assert experiment.get_active_dataset() is None

    experiment.set_active_dataset("v1_abc12345")

    assert experiment.get_active_dataset() == "v1_abc12345"
    mlflow_experiment = client.raw().get_experiment_by_name("home_credit/baseline")
    assert mlflow_experiment.tags[tags.ACTIVE_DATASET_ID] == "v1_abc12345"


def test_active_trial_logs_lifecycle_metric_json_and_next_number(bound_file_mlflow):
    experiment.ensure()

    with trial.active(slug="baseline_lr", strategy="baseline") as run_id:
        trial.log_metric(run_id, "holdout.auc", 0.71)
        trial.log_metrics(run_id, {"train.auc": 0.75})
        trial.log_param(run_id, "C", "1.0")
        trial.log_params(run_id, {"solver": "liblinear"})
        trial.set_tag(run_id, tags.TRIAL_NUMBER, "7")
        trial.log_json(run_id, "debug/sample", {"rows": 200})

    mlflow_client = client.raw()
    run = mlflow_client.get_run(run_id)
    artifact_path = mlflow_client.download_artifacts(run_id, "debug/sample.json")

    assert run.data.tags["run.kind"] == "trial"
    assert run.data.tags["project.name"] == "home_credit"
    assert run.data.tags["experiment.id"] == "baseline"
    assert run.data.tags["trial.slug"] == "baseline_lr"
    assert run.data.params["trial.strategy"] == "baseline"
    assert run.data.tags["trial.status"] == "FINISHED"
    assert run.data.metrics["holdout.auc"] == 0.71
    assert run.data.metrics["train.auc"] == 0.75
    assert run.data.params["C"] == "1.0"
    assert run.data.params["solver"] == "liblinear"
    assert json.loads(open(artifact_path, encoding="utf-8").read()) == {"rows": 200}
    assert experiment.next_trial_number() == 8


def test_active_trial_marks_failed_on_exception(bound_file_mlflow):
    experiment.ensure()

    with pytest.raises(RuntimeError, match="boom"):
        with trial.active(slug="failing_trial", strategy="baseline") as run_id:
            raise RuntimeError("boom")

    run = client.raw().get_run(run_id)
    assert run.data.tags["trial.status"] == "FAILED"


def test_http_run_and_artifact_urls_resolve_numeric_experiment_id(bound_file_mlflow):
    experiment.ensure()
    with trial.active(slug="baseline_lr", strategy="baseline") as run_id:
        pass

    client.bind(
        tracking_uri=bound_file_mlflow,
        bucket="",
        gcs_prefix="automl-root",
        project_name="home_credit",
        experiment_id="baseline",
    )
    mlflow_experiment_id = client.raw().get_run(run_id).info.experiment_id
    client.bind(
        tracking_uri="https://mlflow.example.com/",
        bucket="",
        gcs_prefix="automl-root",
        project_name="home_credit",
        experiment_id="baseline",
    )

    expected_run_url = (
        f"https://mlflow.example.com/#/experiments/{mlflow_experiment_id}/runs/{run_id}"
    )
    assert client.run_url(run_id) == expected_run_url
    assert client.artifact_url(run_id, "debug/sample.json") == (
        expected_run_url + "/artifacts/debug/sample.json"
    )


def test_local_run_url_is_empty(bound_file_mlflow):
    experiment.ensure()
    with trial.active(slug="baseline_lr", strategy="baseline") as run_id:
        pass

    assert client.run_url(run_id) == ""
    assert client.artifact_url(run_id, "debug/sample.json") == ""


def test_http_experiment_and_project_urls_resolve_numeric_experiment_id(monkeypatch):
    client.clear()
    client.bind(
        tracking_uri="https://mlflow.example.com/",
        bucket="",
        gcs_prefix="automl-root",
        project_name="home_credit",
        experiment_id="baseline",
        dry_run=True,
    )
    # The HTTP MLflow server is unavailable in a unit test, so stub the single
    # lookup that maps an experiment route name to its numeric id. Route
    # construction and URL formatting still run for real.
    numeric_ids = {
        "dry_run/home_credit/baseline": "11",
        "dry_run/home_credit/000_overview": "22",
    }

    class _Experiment:
        def __init__(self, experiment_id):
            self.experiment_id = experiment_id

    monkeypatch.setattr(
        client,
        "get_experiment_by_name",
        lambda name: _Experiment(numeric_ids[name]) if name in numeric_ids else None,
    )

    assert client.experiment_url() == "https://mlflow.example.com/#/experiments/11"
    assert client.project_url() == "https://mlflow.example.com/#/experiments/22"
    client.clear()


def test_experiment_and_project_urls_are_empty_for_local_store(bound_file_mlflow):
    experiment.ensure()

    assert client.experiment_url() == ""
    assert client.project_url() == ""


def test_list_route_experiment_names_includes_overview_and_deleted(bound_file_mlflow):
    from automl.mlflow import project as mlflow_project

    experiment.ensure()  # home_credit/baseline
    mlflow_project.ensure_overview()  # home_credit/000_overview
    raw = client.raw()
    old_id = raw.create_experiment("home_credit/old-exp")
    raw.delete_experiment(old_id)  # soft-deleted -> must still be listed for cleanup
    raw.create_experiment("dry_run/home_credit/other")  # different container -> excluded

    names = mlflow_project.list_route_experiment_names()

    assert names == ["home_credit/000_overview", "home_credit/baseline", "home_credit/old-exp"]


def test_list_all_experiment_names_returns_full_names_including_deleted(bound_file_mlflow):
    from automl.mlflow import project as mlflow_project

    experiment.ensure()  # home_credit/baseline
    raw = client.raw()
    raw.create_experiment("qa-smoke/home_credit/x")
    gone = raw.create_experiment("qa/agent/home_credit/y")
    raw.delete_experiment(gone)  # soft-deleted must still be listed

    names = mlflow_project.list_all_experiment_names()

    assert "home_credit/baseline" in names
    assert "qa-smoke/home_credit/x" in names
    assert "qa/agent/home_credit/y" in names


def test_experiment_url_is_empty_without_a_bound_experiment():
    client.clear()
    client.bind(
        tracking_uri="https://mlflow.example.com/",
        bucket="",
        gcs_prefix="automl-root",
        project_name="home_credit",
        experiment_id=None,
    )

    assert client.experiment_url() == ""
    client.clear()
