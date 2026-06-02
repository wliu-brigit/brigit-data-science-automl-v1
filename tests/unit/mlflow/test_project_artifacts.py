import pytest

from automl.errors import StorageError
from automl.mlflow import client
from automl.mlflow.project import artifacts
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
