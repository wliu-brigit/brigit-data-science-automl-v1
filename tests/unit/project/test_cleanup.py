import pytest

from automl.mlflow import client, experiment
from automl.project import ProjectConfig, Session
from automl.project import cleanup as cleanup_module
from automl.project.cleanup import CleanupPlan, CleanupReport, CleanupResult, delete
from automl.trial.types import ParentExperimentRef
from automl.utils.io import gcs

pytestmark = pytest.mark.unit


def _session(tmp_path, *, dry_run=False, namespace="", experiment_id="baseline"):
    project_dir = tmp_path / "projects" / "home_credit"
    project_dir.mkdir(parents=True)
    return Session(
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


class DeleteBlob:
    def __init__(self, client, name):
        self._client = client
        self.name = name

    def delete(self):
        self._client.deleted.append(self.name)
        self._client._names.remove(self.name)


class DeleteClient:
    def __init__(self, names):
        self.deleted = []
        self.copied = []
        self._names = names

    def list_blobs(self, bucket, prefix):
        assert bucket == "automl-test-bucket"
        return [DeleteBlob(self, name) for name in self._names if name.startswith(prefix)]

    def bucket(self, name):
        assert name == "automl-test-bucket"
        return self

    def copy_blob(self, blob, destination_bucket, new_name):
        assert destination_bucket is self
        self.copied.append((blob.name, new_name))
        self._names.append(new_name)


class FailingDeleteBlob:
    name = "automl-root/home_credit/baseline/a.json"

    def delete(self):
        raise RuntimeError("boom")


class FailingDeleteClient:
    def bucket(self, name):
        assert name == "automl-test-bucket"
        return self

    def copy_blob(self, blob, destination_bucket, new_name):
        assert destination_bucket is self
        assert new_name == "automl-root/deleted/home_credit/baseline/a.json"

    def list_blobs(self, bucket, prefix):
        assert bucket == "automl-test-bucket"
        assert prefix == "automl-root/home_credit/baseline/"
        return [FailingDeleteBlob()]


def test_cleanup_report_from_dict_strips_unknown_fields():
    report = CleanupReport.from_dict(
        {
            "applied": False,
            "plan": {"scope": "experiment", "identifier": "baseline", "future": "ignored"},
            "future": "ignored",
        }
    )

    assert isinstance(report.plan, CleanupPlan)
    assert report.plan.scope == "experiment"
    assert report.result is None


def test_experiment_delete_preview_builds_one_universe_plan_without_mutation(tmp_path):
    active = _session(tmp_path, dry_run=True, namespace="qa")
    local_root = (
        active.config.project_dir / "experiments" / "qa" / "dry_run" / "home_credit" / "baseline"
    )
    local_root.mkdir(parents=True)

    report = delete("baseline", scope="experiment", apply=False, session=active)

    assert report.applied is False
    assert report.result is None
    assert report.plan.dry_run is True
    assert report.plan.mlflow_experiment_targets == [("qa/dry_run/home_credit/baseline", "")]
    assert report.plan.gcs_prefix_patterns == [
        "gs://automl-test-bucket/automl-root/qa/dry_run/home_credit/baseline/"
    ]
    assert str(local_root) in report.plan.local_paths
    assert local_root.exists()


def test_experiment_delete_preview_handles_empty_gcs_prefix(tmp_path):
    active = _session(tmp_path)
    active = Session(
        config=ProjectConfig(
            project_name=active.project_name,
            repo_root=active.config.repo_root,
            project_dir=active.config.project_dir,
            gcs_bucket=active.config.gcs_bucket,
            gcs_prefix="",
            mlflow_tracking_uri=active.config.mlflow_tracking_uri,
        ),
        experiment_id=active.active_experiment_id,
    )

    report = delete("baseline", scope="experiment", apply=False, session=active)

    assert report.plan.gcs_prefix_patterns == ["gs://automl-test-bucket/home_credit/baseline/"]


def test_project_delete_plan_never_targets_user_authored_project_dir(tmp_path):
    active = _session(tmp_path)
    report = delete("home_credit", scope="project", apply=False, session=active)

    assert str(active.config.project_dir) not in report.plan.local_paths
    assert all("/experiments/" in path or "/.cache/" in path for path in report.plan.local_paths)


def test_project_delete_preview_covers_whole_route_and_gates_cache_off_subroutes(
    tmp_path, monkeypatch
):
    active = _session(tmp_path, dry_run=True, namespace="qa")
    # comprehensive enumeration returns full names incl. overview + soft-deleted
    monkeypatch.setattr(
        cleanup_module,
        "_route_experiment_names",
        lambda: ["qa/dry_run/home_credit/baseline", "qa/dry_run/home_credit/000_overview"],
    )

    report = delete("home_credit", scope="project", apply=False, session=active)

    assert set(report.plan.mlflow_experiment_targets) == {
        ("qa/dry_run/home_credit/baseline", ""),
        ("qa/dry_run/home_credit/000_overview", ""),
    }
    # one wholesale project-route prefix (catches data/, overview/, every experiment)
    assert report.plan.gcs_prefix_patterns == [
        "gs://automl-test-bucket/automl-root/qa/dry_run/home_credit/"
    ]
    # a sub-container (dry_run/namespace) cleanup must NOT touch the shared .cache
    assert set(report.plan.local_paths) == {
        str(active.config.project_dir / "experiments" / "qa" / "dry_run" / "home_credit"),
    }


def test_project_delete_base_route_includes_cache_and_uses_project_prefix(tmp_path, monkeypatch):
    active = _session(tmp_path)  # base route: namespace="", dry_run=False
    monkeypatch.setattr(
        cleanup_module,
        "_route_experiment_names",
        lambda: ["home_credit/baseline", "home_credit/000_overview"],
    )

    report = delete("home_credit", scope="project", apply=False, session=active)

    assert set(report.plan.mlflow_experiment_targets) == {
        ("home_credit/baseline", ""),
        ("home_credit/000_overview", ""),
    }
    assert report.plan.gcs_prefix_patterns == ["gs://automl-test-bucket/automl-root/home_credit/"]
    assert set(report.plan.local_paths) == {
        str(active.config.project_dir / "experiments" / "home_credit"),
        str(active.config.project_dir / ".cache" / "automl"),
    }


def test_project_delete_scope_qa_plan_targets_only_qa_namespaces(tmp_path, monkeypatch):
    active = _session(tmp_path)
    monkeypatch.setattr(
        cleanup_module,
        "_all_experiment_names",
        lambda: [
            "qa-smoke-1/home_credit/run-1",
            "qa/agent-e2e-1/dry_run/home_credit/example",
            "home_credit/baseline",  # not QA -> excluded
            "dry_run/generic/000_overview",  # not QA -> excluded
        ],
    )

    report = delete(scope="qa", apply=False, session=active)

    assert set(report.plan.mlflow_experiment_targets) == {
        ("qa-smoke-1/home_credit/run-1", ""),
        ("qa/agent-e2e-1/dry_run/home_credit/example", ""),
    }
    # one recursive GCS prefix per distinct QA namespace (wipes its whole subtree)
    assert set(report.plan.gcs_prefix_patterns) == {
        "gs://automl-test-bucket/automl-root/qa-smoke-1/",
        "gs://automl-test-bucket/automl-root/qa/agent-e2e-1/",
    }
    assert set(report.plan.local_paths) == {
        str(active.config.project_dir / "experiments" / "qa-smoke-1"),
        str(active.config.project_dir / "experiments" / "qa" / "agent-e2e-1"),
    }


def test_project_delete_scope_qa_plan_ignores_archived_qa_namespaces(tmp_path, monkeypatch):
    active = _session(tmp_path)
    monkeypatch.setattr(
        cleanup_module,
        "_all_experiment_names",
        lambda: [
            "deleted/qa/agent/dry_run/home_credit/example",
            "deleted/prod/home_credit/example",
            "qa/current/home_credit/example",
        ],
    )

    report = delete(scope="qa", apply=False, session=active)

    assert report.plan.mlflow_experiment_targets == [("qa/current/home_credit/example", "")]
    assert report.plan.gcs_prefix_patterns == [
        "gs://automl-test-bucket/automl-root/qa/current/"
    ]
    assert report.plan.local_paths == [
        str(active.config.project_dir / "experiments" / "qa" / "current")
    ]


def test_trial_delete_preview_pins_current_route_gcs_and_local_plan(tmp_path, monkeypatch):
    active = _session(tmp_path, dry_run=True, namespace="qa", experiment_id="cleanup-exp")
    monkeypatch.setattr(
        cleanup_module.mlflow_client,
        "run_start_time",
        lambda _: 1_766_016_000_000,
    )

    report = delete(
        "run-1",
        scope="trial",
        apply=False,
        session=active,
        parent_experiment=ParentExperimentRef(
            mlflow_experiment_name="qa/dry_run/home_credit/cleanup-exp",
            project_name="home_credit",
            experiment_id="cleanup-exp",
        ),
    )

    assert report.plan.mlflow_experiment_targets == []
    assert report.plan.mlflow_run_targets == ["run-1"]
    assert report.plan.gcs_prefix_patterns == [
        "gs://automl-test-bucket/automl-root/qa/dry_run/home_credit/cleanup-exp/runs/2025-12/run-1/"
    ]
    assert report.plan.local_paths == [
        str(
            active.config.project_dir
            / "experiments"
            / "qa"
            / "dry_run"
            / "home_credit"
            / "cleanup-exp"
            / "run-1"
        )
    ]


def test_project_delete_requires_name_to_match_active_project(tmp_path):
    active = _session(tmp_path)

    try:
        delete("other_project", scope="project", apply=False, session=active)
    except ValueError as exc:
        assert "active project" in str(exc)
    else:
        raise AssertionError("expected project name mismatch to fail")


def test_project_delete_scope_qa_rejects_name(tmp_path):
    active = _session(tmp_path)

    with pytest.raises(ValueError, match="does not accept name"):
        delete("qa", scope="qa", apply=False, session=active)


def test_project_delete_project_scope_requires_name(tmp_path):
    active = _session(tmp_path)

    with pytest.raises(ValueError, match="requires name"):
        delete(scope="project", apply=False, session=active)


def test_gcs_delete_prefix_collects_deleted_count():
    fake = DeleteClient(["automl-root/home_credit/baseline/a.json", "other/x.json"])

    result = gcs.delete_prefix(
        "gs://automl-test-bucket/automl-root/home_credit/baseline/",
        client=fake,
    )

    assert result == 1
    assert fake.deleted == ["automl-root/home_credit/baseline/a.json"]


def test_apply_soft_deletes_mlflow_then_gcs_then_local(tmp_path, monkeypatch):
    active = _session(tmp_path)
    client.bind(
        tracking_uri=active.config.mlflow_tracking_uri,
        bucket=active.config.gcs_bucket,
        gcs_prefix=active.config.gcs_prefix,
        project_name=active.project_name,
        experiment_id=active.active_experiment_id,
    )
    experiment.ensure()
    local_root = active.config.project_dir / "experiments" / "home_credit" / "baseline"
    local_root.mkdir(parents=True)
    fake = DeleteClient(["automl-root/home_credit/baseline/runs/run-1/model.pkl"])
    monkeypatch.setattr(gcs, "_gcs_client", lambda: fake)

    report = delete("baseline", scope="experiment", apply=True, session=active)
    archived_name = "deleted/home_credit/baseline"
    archived_local = (
        active.config.project_dir
        / "experiments"
        / "deleted"
        / "home_credit"
        / "baseline"
    )

    assert report.applied is True
    assert isinstance(report.result, CleanupResult)
    assert report.result.mlflow_experiments["home_credit/baseline"] == (
        f"archived: {archived_name}; deleted"
    )
    assert report.result.gcs["gs://automl-test-bucket/automl-root/home_credit/baseline/"] == (
        "archived 1 to "
        "gs://automl-test-bucket/automl-root/deleted/home_credit/baseline/"
    )
    assert fake.copied == [
        (
            "automl-root/home_credit/baseline/runs/run-1/model.pkl",
            "automl-root/deleted/home_credit/baseline/runs/run-1/model.pkl",
        )
    ]
    assert report.result.local[str(local_root)] == f"archived: {archived_local}"
    assert not local_root.exists()
    assert archived_local.exists()
    assert client.raw().get_experiment_by_name("home_credit/baseline") is None
    archived = client.raw().get_experiment_by_name(archived_name)
    assert archived is not None
    assert archived.lifecycle_stage == "deleted"

    rerun = delete("baseline", scope="experiment", apply=True, session=active)
    assert rerun.applied is True
    assert (
        rerun.result.mlflow_experiments["home_credit/baseline"] == "skipped: not found"
    )


def test_apply_records_gcs_delete_failure_and_continues(tmp_path, monkeypatch):
    active = _session(tmp_path)
    client.bind(
        tracking_uri=active.config.mlflow_tracking_uri,
        bucket=active.config.gcs_bucket,
        gcs_prefix=active.config.gcs_prefix,
        project_name=active.project_name,
        experiment_id=active.active_experiment_id,
    )
    experiment.ensure()
    local_root = active.config.project_dir / "experiments" / "home_credit" / "baseline"
    local_root.mkdir(parents=True)
    monkeypatch.setattr(gcs, "_gcs_client", lambda: FailingDeleteClient())

    report = delete("baseline", scope="experiment", apply=True, session=active)

    assert report.result.gcs[
        "gs://automl-test-bucket/automl-root/home_credit/baseline/"
    ].startswith("failed: failed to move")
    assert report.result.local[str(local_root)].startswith("archived:")


def test_trial_delete_archives_gcs_and_local(tmp_path, monkeypatch):
    active = _session(tmp_path)
    fake = DeleteClient(
        ["automl-root/home_credit/baseline/runs/2025-12/run-1/model.pkl"]
    )
    monkeypatch.setattr(gcs, "_gcs_client", lambda: fake)
    monkeypatch.setattr(
        cleanup_module.mlflow_client,
        "run_start_time",
        lambda run_id: 1764547200000,
    )
    deleted_runs = []
    monkeypatch.setattr(cleanup_module.mlflow_client, "delete_run", deleted_runs.append)
    local_root = (
        active.config.project_dir
        / "experiments"
        / "home_credit"
        / "baseline"
        / "run-1"
    )
    local_root.mkdir(parents=True)

    report = delete(
        "run-1",
        scope="trial",
        apply=True,
        session=active,
        parent_experiment=ParentExperimentRef(
            mlflow_experiment_name="home_credit/baseline",
            project_name="home_credit",
            experiment_id="baseline",
        ),
    )

    archived_local = (
        active.config.project_dir
        / "experiments"
        / "deleted"
        / "home_credit"
        / "baseline"
        / "run-1"
    )
    assert report.result.mlflow_runs == {"run-1": "deleted"}
    assert deleted_runs == ["run-1"]
    assert report.result.gcs[
        "gs://automl-test-bucket/automl-root/home_credit/baseline/runs/2025-12/run-1/"
    ] == (
        "archived 1 to "
        "gs://automl-test-bucket/automl-root/deleted/home_credit/baseline/runs/2025-12/run-1/"
    )
    assert fake.copied == [
        (
            "automl-root/home_credit/baseline/runs/2025-12/run-1/model.pkl",
            "automl-root/deleted/home_credit/baseline/runs/2025-12/run-1/model.pkl",
        )
    ]
    assert report.result.local[str(local_root)] == f"archived: {archived_local}"
    assert archived_local.exists()
    assert not local_root.exists()
