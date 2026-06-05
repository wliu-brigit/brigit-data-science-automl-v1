import json

import pytest

from automl.data import Profile
from automl.errors import StorageError
from automl.mlflow import client
from automl.mlflow.experiment import artifacts

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


def _profile_dir(tmp_path):
    local_dir = tmp_path / "profile"
    (local_dir / "charts").mkdir(parents=True)
    (local_dir / "data_card.json").write_text('{"rows": 4}', encoding="utf-8")
    (local_dir / "data_observations.json").write_text(
        '{"observations": []}',
        encoding="utf-8",
    )
    (local_dir / "charts" / "label_distribution.png").write_bytes(b"png")
    (local_dir / "profile_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_id": "v1_profile",
                "target_column": "target",
                "created_at": "2026-05-27T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    return local_dir


def _experiment_overview_run():
    experiment = client.raw().get_experiment_by_name("demo/baseline")
    assert experiment is not None
    runs = client.raw().search_runs(
        [experiment.experiment_id],
        filter_string="tags.`run.kind` = 'experiment_overview'",
    )
    assert len(runs) == 1
    return runs[0]


def test_write_then_read_dataset_record_round_trips(bound_file_mlflow):
    payload = {"id": "v1_ab12cd34", "identity_hash": "sha256:x", "recipe": {"target": "y"}}
    uri = artifacts.write_dataset_record(payload, dataset_id="v1_ab12cd34")
    assert uri.startswith("runs:/") and uri.endswith("datasets/v1_ab12cd34/dataset.json")
    record = artifacts.read_dataset_record("v1_ab12cd34")
    assert record == {**payload, "record_uri": uri}


def test_read_dataset_record_returns_none_when_absent(bound_file_mlflow):
    assert artifacts.read_dataset_record("v9_missing") is None


def test_list_dataset_records_returns_every_version_folder(bound_file_mlflow):
    artifacts.write_dataset_record({"id": "v1_a"}, dataset_id="v1_a")
    artifacts.write_dataset_record({"id": "v2_b"}, dataset_id="v2_b")
    records = artifacts.list_dataset_records()
    assert [record["id"] for record in records] == ["v1_a", "v2_b"]


def test_list_dataset_records_is_empty_for_a_fresh_experiment(bound_file_mlflow):
    # Missing datasets/ folder lists cleanly as [] (verified against both
    # file-backed MLflow and the live prod proxy, 2026-06-04).
    assert artifacts.list_dataset_records() == []


def test_write_and_read_active_dataset_pointer_round_trips(bound_file_mlflow):
    uri = artifacts.write_active_dataset_pointer("v2_good")

    assert uri.startswith("runs:/")
    assert uri.endswith("datasets/active_pointer.json")
    assert artifacts.read_active_dataset_pointer() == {
        "schema_version": 1,
        "active_dataset_id": "v2_good",
    }


def test_read_active_dataset_pointer_returns_none_when_absent(bound_file_mlflow):
    assert artifacts.read_active_dataset_pointer() is None


def test_list_dataset_records_propagates_transport_failures(bound_file_mlflow, monkeypatch):
    # An exception from list_artifacts is a genuine transport/auth failure,
    # never "no datasets yet" — it must surface, not read as an empty index.
    artifacts.write_dataset_record({"id": "v1_a"}, dataset_id="v1_a")

    class ExplodingClient:
        def list_artifacts(self, run_id, path=None):
            raise RuntimeError("proxy unreachable")

    monkeypatch.setattr(artifacts, "_overview_run_id_or_none", lambda experiment_id=None: "run-123")
    monkeypatch.setattr(artifacts.client, "raw", lambda: ExplodingClient())
    with pytest.raises(StorageError, match="list dataset records"):
        artifacts.list_dataset_records()


def test_write_and_read_profile_round_trips_via_experiment_overview(bound_file_mlflow, tmp_path):
    uris = artifacts.write_profile("v1_profile", local_dir=_profile_dir(tmp_path))

    assert uris["data_card_uri"].endswith("/datasets/v1_profile/profile/data_card.json")
    assert uris["data_observations_uri"].endswith(
        "/datasets/v1_profile/profile/data_observations.json"
    )
    assert uris["profile_manifest_uri"].endswith(
        "/datasets/v1_profile/profile/profile_manifest.json"
    )
    assert uris["chart_uris"]["label_distribution"].endswith(
        "/datasets/v1_profile/profile/charts/label_distribution.png"
    )
    overview_run = _experiment_overview_run()
    assert uris["data_card_uri"].startswith(f"runs:/{overview_run.info.run_id}/")

    restored = artifacts.read_profile("v1_profile")

    assert isinstance(restored, Profile)
    assert restored.dataset_id == "v1_profile"
    assert restored.target_column == "target"
    assert restored.data_card_uri == uris["data_card_uri"]
    assert restored.chart_uris == uris["chart_uris"]


def test_read_profile_returns_none_when_missing(bound_file_mlflow):
    from automl.mlflow.experiment import lifecycle

    lifecycle.ensure_overview()

    assert artifacts.read_profile("missing") is None


def test_read_profile_returns_none_before_overview_exists(bound_file_mlflow):
    assert artifacts.read_profile("missing") is None


def test_log_source_trace_uploads_named_files_to_experiment_overview(bound_file_mlflow, tmp_path):
    trace_file = tmp_path / "executed.sql"
    trace_file.write_text("select 1 as value", encoding="utf-8")

    uris = artifacts.log_source_trace("v1_trace", {"base_table.executed.sql": trace_file})

    assert uris["base_table.executed.sql"].endswith(
        "/datasets/v1_trace/source_trace/base_table.executed.sql"
    )
    overview_run = _experiment_overview_run()
    local_path = client.raw().download_artifacts(
        overview_run.info.run_id,
        "datasets/v1_trace/source_trace/base_table.executed.sql",
    )
    assert open(local_path, encoding="utf-8").read() == "select 1 as value"
