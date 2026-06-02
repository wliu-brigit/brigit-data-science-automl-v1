from datetime import UTC, datetime

import pytest

from automl.experiment import delete as delete_experiment
from automl.mlflow import client, experiment, trial
from automl.project import ProjectConfig, Session
from automl.trial.cleanup import delete as delete_trial
from automl.utils.io import gcs

pytestmark = pytest.mark.integration


class FakeBlob:
    def __init__(self, store, bucket, name):
        self._store = store
        self._bucket = bucket
        self.name = name

    def delete(self):
        self._store.pop((self._bucket, self.name), None)

    def upload_from_string(self, data, **kwargs):
        self._store[(self._bucket, self.name)] = data if isinstance(data, bytes) else data.encode()


class FakeBucket:
    def __init__(self, store, name):
        self._store = store
        self.name = name

    def blob(self, name):
        return FakeBlob(self._store, self.name, name)

    def list_blobs(self, prefix):
        return [
            FakeBlob(self._store, bucket, name)
            for (bucket, name) in sorted(self._store)
            if bucket == self.name and name.startswith(prefix)
        ]


class FakeGCSClient:
    def __init__(self):
        self.store = {}

    def bucket(self, name):
        return FakeBucket(self.store, name)

    def list_blobs(self, bucket, prefix):
        return self.bucket(bucket).list_blobs(prefix)


def _active(tmp_path, *, dry_run=False, namespace="", experiment_id="baseline"):
    project_dir = tmp_path / "projects" / "home_credit"
    project_dir.mkdir(parents=True, exist_ok=True)
    active = Session(
        config=ProjectConfig(
            project_name="home_credit",
            repo_root=tmp_path,
            project_dir=project_dir,
            gcs_bucket="automl-test-bucket",
            gcs_prefix="automl-root",
            mlflow_tracking_uri=(tmp_path / "mlruns").as_uri(),
        ),
        dry_run=dry_run,
        namespace=namespace,
        experiment_id=experiment_id,
    )
    client.bind(
        tracking_uri=active.config.mlflow_tracking_uri,
        bucket=active.config.gcs_bucket,
        gcs_prefix=active.config.gcs_prefix,
        project_name=active.project_name,
        experiment_id=experiment_id,
        dry_run=dry_run,
        namespace=namespace,
    )
    return active


def test_experiment_delete_apply_removes_only_current_universe(tmp_path, monkeypatch):
    fake = FakeGCSClient()
    monkeypatch.setattr(gcs, "_gcs_client", lambda: fake)
    active = _active(tmp_path, namespace="qa", experiment_id="cleanup-exp")
    sibling = _active(tmp_path, namespace="prod", experiment_id="cleanup-exp")
    active = _active(tmp_path, namespace="qa", experiment_id="cleanup-exp")

    experiment.ensure()
    with trial.active(slug="scored", strategy="baseline") as run_id:
        trial.log_metric(run_id, "test.auc", 0.8)
    fake.store[("automl-test-bucket", "automl-root/qa/home_credit/cleanup-exp/runs/blob.json")] = b"{}"
    fake.store[("automl-test-bucket", "automl-root/prod/home_credit/cleanup-exp/runs/blob.json")] = b"{}"
    local_root = active.config.project_dir / "experiments" / "qa" / "home_credit" / "cleanup-exp"
    sibling_root = sibling.config.project_dir / "experiments" / "prod" / "home_credit" / "cleanup-exp"
    local_root.mkdir(parents=True)
    sibling_root.mkdir(parents=True)

    report = delete_experiment("cleanup-exp", apply=True, session=active)

    assert report.applied is True
    assert report.plan.mlflow_experiment_targets == [("qa/home_credit/cleanup-exp", "")]
    assert report.plan.gcs_prefix_patterns == [
        "gs://automl-test-bucket/automl-root/qa/home_credit/cleanup-exp/"
    ]
    assert report.plan.local_paths == [str(local_root)]
    assert report.result is not None
    assert report.result.mlflow_experiments == {"qa/home_credit/cleanup-exp": "deleted"}
    assert report.result.gcs == {"gs://automl-test-bucket/automl-root/qa/home_credit/cleanup-exp/": 1}
    assert report.result.local == {str(local_root): "deleted"}
    assert (
        "automl-test-bucket",
        "automl-root/qa/home_credit/cleanup-exp/runs/blob.json",
    ) not in fake.store
    assert (
        "automl-test-bucket",
        "automl-root/prod/home_credit/cleanup-exp/runs/blob.json",
    ) in fake.store
    assert not local_root.exists()
    assert sibling_root.exists()


def test_trial_delete_rejects_run_from_other_namespace(tmp_path):
    _active(tmp_path, namespace="qa", experiment_id="cleanup-exp")
    experiment.ensure()
    with trial.active(slug="qa-run", strategy="baseline") as run_id:
        pass
    prod = _active(tmp_path, namespace="prod", experiment_id="cleanup-exp")

    with pytest.raises(Exception, match="current session"):
        delete_trial(run_id, apply=False, session=prod)


def test_trial_delete_rebinds_explicit_session_before_parent_lookup(tmp_path):
    qa = _active(tmp_path, namespace="qa", experiment_id="cleanup-exp")
    experiment.ensure()
    with trial.active(slug="qa-run", strategy="baseline") as run_id:
        pass
    _active(tmp_path, namespace="prod", experiment_id="cleanup-exp")

    report = delete_trial(run_id, apply=False, session=qa)

    run = client.raw().get_run(run_id)
    started_at = datetime.fromtimestamp(run.info.start_time / 1000, UTC)
    partition = started_at.strftime("%Y-%m")

    assert report.plan.mlflow_experiment_targets == []
    assert report.plan.mlflow_run_targets == [run_id]
    assert report.plan.gcs_prefix_patterns == [
        f"gs://automl-test-bucket/automl-root/qa/home_credit/cleanup-exp/runs/{partition}/{run_id}/"
    ]
    assert report.plan.local_paths == [
        str(qa.config.project_dir / "experiments" / "qa" / "home_credit" / "cleanup-exp" / run_id)
    ]
