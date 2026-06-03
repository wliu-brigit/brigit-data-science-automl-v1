import json

import pytest

from automl.data import Profile
from automl.errors import StorageError
from automl.mlflow import client
from automl.mlflow.experiment import artifacts
from automl.utils.io import gcs

pytestmark = pytest.mark.unit


class FakeBlob:
    def __init__(self, store: dict[tuple[str, str], bytes], bucket: str, name: str) -> None:
        self._store = store
        self._bucket = bucket
        self.name = name

    def download_as_bytes(self) -> bytes:
        return self._store[(self._bucket, self.name)]

    def exists(self) -> bool:
        return (self._bucket, self.name) in self._store


class FakeBucket:
    def __init__(self, store: dict[tuple[str, str], bytes], name: str) -> None:
        self._store = store
        self.name = name

    def blob(self, name: str) -> FakeBlob:
        return FakeBlob(self._store, self.name, name)


class FakeGCSClient:
    def __init__(self) -> None:
        self.store: dict[tuple[str, str], bytes] = {}

    def bucket(self, name: str) -> FakeBucket:
        return FakeBucket(self.store, name)


@pytest.fixture
def bound_artifacts(monkeypatch) -> FakeGCSClient:
    fake = FakeGCSClient()
    monkeypatch.setattr(gcs, "_gcs_client", lambda: fake)
    client.bind(
        tracking_uri="file:///tmp/mlruns",
        bucket="automl-test-bucket",
        gcs_prefix="root",
        project_name="demo",
        experiment_id="baseline",
    )
    yield fake
    client.clear()


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


def test_dataset_index_uri_is_experiment_scoped(bound_artifacts):
    assert (
        artifacts.dataset_index_uri()
        == "gs://automl-test-bucket/root/demo/baseline/data/dataset_index.json"
    )


def test_read_dataset_index_returns_empty_only_when_object_is_missing(bound_artifacts):
    assert artifacts.read_dataset_index() == {"schema_version": 1, "datasets": []}


def test_read_dataset_index_raises_storage_error_for_corrupt_json(bound_artifacts):
    bucket, blob = gcs.parse_gcs_uri(artifacts.dataset_index_uri())
    bound_artifacts.store[(bucket, blob)] = b"{not json"

    with pytest.raises(StorageError, match="Failed to read dataset index"):
        artifacts.read_dataset_index()


def test_read_dataset_index_raises_storage_error_for_unreadable_existing_object(monkeypatch):
    client.bind(
        tracking_uri="file:///tmp/mlruns",
        bucket="automl-test-bucket",
        gcs_prefix="root",
        project_name="demo",
        experiment_id="baseline",
    )
    monkeypatch.setattr(artifacts.gcs, "blob_exists", lambda uri: True)

    def raise_permission(uri):
        raise PermissionError("denied")

    monkeypatch.setattr(artifacts.gcs, "read_json", raise_permission)

    try:
        with pytest.raises(StorageError, match="Failed to read dataset index"):
            artifacts.read_dataset_index()
    finally:
        client.clear()


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


def test_log_dataset_catalog_writes_index_and_latest_to_experiment_overview(bound_file_mlflow):
    payload = {
        "schema_version": 1,
        "datasets": [{"id": "v1_abc", "n_rows": 4}],
    }

    artifacts.log_dataset_catalog(payload, active_dataset_id="v1_abc")

    overview_run = _experiment_overview_run()
    index_path = client.raw().download_artifacts(overview_run.info.run_id, "datasets/index.json")
    latest_path = client.raw().download_artifacts(overview_run.info.run_id, "datasets/latest.json")
    with open(index_path, encoding="utf-8") as handle:
        index = json.load(handle)
    with open(latest_path, encoding="utf-8") as handle:
        latest = json.load(handle)
    assert index["active_dataset_id"] == "v1_abc"
    assert latest == {
        "schema_version": 1,
        "dataset_id": "v1_abc",
        "dataset": {"id": "v1_abc", "n_rows": 4},
    }


def test_log_source_trace_uploads_named_files_to_experiment_overview(bound_file_mlflow, tmp_path):
    trace_file = tmp_path / "executed.sql"
    trace_file.write_text("select 1 as value", encoding="utf-8")

    uris = artifacts.log_source_trace("v1_trace", {"base_data.executed.sql": trace_file})

    assert uris["base_data.executed.sql"].endswith(
        "/datasets/v1_trace/source_trace/base_data.executed.sql"
    )
    overview_run = _experiment_overview_run()
    local_path = client.raw().download_artifacts(
        overview_run.info.run_id,
        "datasets/v1_trace/source_trace/base_data.executed.sql",
    )
    assert open(local_path, encoding="utf-8").read() == "select 1 as value"
