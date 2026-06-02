import json

import pytest

from automl.data import Profile
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


def test_ensure_overview_creates_project_overview_run(bound_file_mlflow):
    overview = mlflow_project.ensure_overview()

    assert overview.project_name == "demo"
    assert overview.created_at
    experiment = client.raw().get_experiment_by_name("demo/overview")
    assert experiment is not None
    runs = client.raw().search_runs([experiment.experiment_id])
    assert len(runs) == 1
    assert runs[0].data.tags["run.kind"] == "project_overview"


def test_write_and_read_profile_round_trips_via_project_overview(bound_file_mlflow, tmp_path):
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

    restored = artifacts.read_profile("v1_profile")

    assert isinstance(restored, Profile)
    assert restored.dataset_id == "v1_profile"
    assert restored.target_column == "target"
    assert restored.data_card_uri == uris["data_card_uri"]
    assert restored.chart_uris == uris["chart_uris"]


def test_read_profile_returns_none_when_missing(bound_file_mlflow):
    mlflow_project.ensure_overview()

    assert artifacts.read_profile("missing") is None


def test_project_log_json_writes_loose_json_to_project_overview(bound_file_mlflow):
    artifacts.log_json("debug/sample", {"rows": 4})

    experiment = client.raw().get_experiment_by_name("demo/overview")
    assert experiment is not None
    runs = client.raw().search_runs([experiment.experiment_id])
    assert len(runs) == 1
    local_path = client.raw().download_artifacts(runs[0].info.run_id, "debug/sample.json")

    with open(local_path, encoding="utf-8") as handle:
        assert json.load(handle) == {"rows": 4}


def test_log_source_trace_uploads_named_files_to_project_overview(bound_file_mlflow, tmp_path):
    trace_file = tmp_path / "executed.sql"
    trace_file.write_text("select 1 as value", encoding="utf-8")

    uris = artifacts.log_source_trace("v1_trace", {"base_data.executed.sql": trace_file})

    assert uris["base_data.executed.sql"].endswith(
        "/datasets/v1_trace/source_trace/base_data.executed.sql"
    )
    experiment = client.raw().get_experiment_by_name("demo/overview")
    assert experiment is not None
    runs = client.raw().search_runs([experiment.experiment_id])
    local_path = client.raw().download_artifacts(
        runs[0].info.run_id,
        "datasets/v1_trace/source_trace/base_data.executed.sql",
    )
    assert open(local_path, encoding="utf-8").read() == "select 1 as value"
