import sqlite3
import types

import pytest

from automl.mlflow import cleanup as cleanup_module
from automl.mlflow.cleanup import PurgePlan, PurgeReport, purge
from automl.project import ProjectConfig, Session

pytestmark = pytest.mark.unit


def _session(tmp_path):
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
    )


def test_purge_report_from_dict_strips_unknown_fields():
    report = PurgeReport.from_dict(
        {
            "applied": False,
            "plan": {"scope": "qa", "identifier": "qa", "future": "ignored"},
            "future": "ignored",
        }
    )

    assert isinstance(report.plan, PurgePlan)
    assert report.plan.scope == "qa"
    assert report.result is None


def test_purge_requires_exactly_one_selector(tmp_path):
    active = _session(tmp_path)

    with pytest.raises(ValueError, match="requires name or scope"):
        purge(apply=False, session=active)
    with pytest.raises(ValueError, match="mutually exclusive"):
        purge(name="deleted/home_credit", scope="qa", apply=False, session=active)


def test_purge_name_rejects_active_route(tmp_path):
    active = _session(tmp_path)

    with pytest.raises(ValueError, match="must start with 'deleted/'"):
        purge(name="home_credit/baseline", apply=False, session=active)


def test_purge_name_preview_targets_one_deleted_route_prefix(tmp_path, monkeypatch):
    active = _session(tmp_path)
    monkeypatch.setattr(
        cleanup_module,
        "_all_experiment_names",
        lambda: [
            "deleted/qa/dev/dry_run/home_credit/example",
            "deleted/qa/other/dry_run/home_credit/example",
            "qa/dev/dry_run/home_credit/example",
        ],
    )

    report = purge(name="deleted/qa/dev", apply=False, session=active)

    assert report.applied is False
    assert report.result is None
    assert report.plan.mlflow_experiment_targets == [
        ("deleted/qa/dev/dry_run/home_credit/example", "")
    ]
    assert report.plan.gcs_prefix_patterns == [
        "gs://automl-test-bucket/automl-root/deleted/qa/dev/"
    ]
    assert report.plan.local_paths == [
        str(active.config.project_dir / "experiments" / "deleted" / "qa" / "dev")
    ]


def test_purge_scope_qa_preview_targets_archived_qa_only(tmp_path, monkeypatch):
    active = _session(tmp_path)
    monkeypatch.setattr(
        cleanup_module,
        "_all_experiment_names",
        lambda: [
            "deleted/qa/agent/dry_run/home_credit/example",
            "deleted/qa-smoke/home_credit/example",
            "deleted/prod/home_credit/example",
            "qa/agent/dry_run/home_credit/example",
        ],
    )

    report = purge(scope="qa", apply=False, session=active)

    assert set(report.plan.mlflow_experiment_targets) == {
        ("deleted/qa/agent/dry_run/home_credit/example", ""),
        ("deleted/qa-smoke/home_credit/example", ""),
    }
    assert set(report.plan.gcs_prefix_patterns) == {
        "gs://automl-test-bucket/automl-root/deleted/qa/agent/",
        "gs://automl-test-bucket/automl-root/deleted/qa-smoke/",
    }
    assert set(report.plan.local_paths) == {
        str(active.config.project_dir / "experiments" / "deleted" / "qa" / "agent"),
        str(active.config.project_dir / "experiments" / "deleted" / "qa-smoke"),
    }


def test_purge_apply_deletes_gcs_local_and_runs_mlflow_gc(tmp_path, monkeypatch):
    active = _session(tmp_path)
    local_root = active.config.project_dir / "experiments" / "deleted" / "home_credit" / "baseline"
    local_root.mkdir(parents=True)
    (local_root / "manifest.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        cleanup_module,
        "_all_experiment_names",
        lambda: ["deleted/home_credit/baseline"],
    )
    monkeypatch.setattr(
        cleanup_module.mlflow_client,
        "get_experiment_by_name",
        lambda name: types.SimpleNamespace(experiment_id="7", lifecycle_stage="deleted"),
    )
    deleted_gcs = []
    monkeypatch.setattr(cleanup_module.gcs, "delete_prefix", lambda prefix: deleted_gcs.append(prefix) or 3)
    gc_calls = []

    def fake_gc(*, experiment_ids, run_ids, backend_store_uri, artifacts_destination, session):
        gc_calls.append(
            {
                "experiment_ids": experiment_ids,
                "run_ids": run_ids,
                "backend_store_uri": backend_store_uri,
                "artifacts_destination": artifacts_destination,
                "session": session,
            }
        )
        return "success", "gc complete"

    monkeypatch.setattr(cleanup_module, "_run_mlflow_gc", fake_gc)

    report = purge(
        name="deleted/home_credit/baseline",
        apply=True,
        backend_store_uri="sqlite:////tmp/mlflow.db",
        session=active,
    )

    assert report.applied is True
    assert report.result is not None
    assert report.result.mlflow_experiments == {"deleted/home_credit/baseline": "purged"}
    assert report.result.gcs == {
        "gs://automl-test-bucket/automl-root/deleted/home_credit/baseline/": 3
    }
    assert report.result.local == {str(local_root): "deleted"}
    assert report.result.mlflow_gc_status == "success"
    assert gc_calls == [
        {
            "experiment_ids": ["7"],
            "run_ids": [],
            "backend_store_uri": "sqlite:////tmp/mlflow.db",
            "artifacts_destination": "",
            "session": active,
        }
    ]
    assert deleted_gcs == ["gs://automl-test-bucket/automl-root/deleted/home_credit/baseline/"]
    assert not local_root.exists()


def test_purge_prunes_orphaned_mlflow_auth_permissions(tmp_path):
    tracking_db = tmp_path / "mlflow.db"
    auth_db = tmp_path / "basic_auth.db"
    with sqlite3.connect(tracking_db) as db:
        db.execute("create table experiments (experiment_id integer primary key, name text)")
        db.execute("create table registered_models (name text primary key)")
        db.execute("insert into experiments (experiment_id, name) values (1, 'kept')")
        db.execute("insert into registered_models (name) values ('kept-model')")
    with sqlite3.connect(auth_db) as db:
        db.execute(
            "create table experiment_permissions "
            "(experiment_id text, user_id integer, permission text)"
        )
        db.execute(
            "create table registered_model_permissions "
            "(name text, user_id integer, permission text)"
        )
        db.execute("insert into experiment_permissions values ('1', 1, 'MANAGE')")
        db.execute("insert into experiment_permissions values ('2', 1, 'MANAGE')")
        db.execute("insert into registered_model_permissions values ('kept-model', 1, 'MANAGE')")
        db.execute("insert into registered_model_permissions values ('gone-model', 1, 'MANAGE')")

    summary = cleanup_module._prune_orphaned_auth_permissions(f"sqlite:///{tracking_db}")

    assert summary == "auth permissions pruned: experiments=1, registered_models=1"
    with sqlite3.connect(auth_db) as db:
        assert db.execute("select experiment_id from experiment_permissions").fetchall() == [("1",)]
        assert db.execute("select name from registered_model_permissions").fetchall() == [
            ("kept-model",)
        ]
