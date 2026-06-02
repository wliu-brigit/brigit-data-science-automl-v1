import json

import pandas as pd
import pytest

from automl.utils.io import gcs

pytestmark = pytest.mark.unit


class FakeBlob:
    def __init__(self, store: dict[tuple[str, str], bytes], bucket: str, name: str) -> None:
        self._store = store
        self._bucket = bucket
        self.name = name
        self.uploads: list[dict[str, object]] = []

    def upload_from_string(
        self,
        data: str | bytes,
        *,
        content_type: str | None = None,
        if_generation_match: int | None = None,
    ) -> None:
        if if_generation_match == 0 and (self._bucket, self.name) in self._store:
            raise FileExistsError(f"object already exists: {self._bucket}/{self.name}")
        raw = data if isinstance(data, bytes) else data.encode("utf-8")
        self._store[(self._bucket, self.name)] = raw
        self.uploads.append(
            {"content_type": content_type, "if_generation_match": if_generation_match}
        )

    def upload_from_file(
        self,
        file_obj,
        *,
        content_type: str | None = None,
        if_generation_match: int | None = None,
    ) -> None:
        if if_generation_match == 0 and (self._bucket, self.name) in self._store:
            raise FileExistsError(f"object already exists: {self._bucket}/{self.name}")
        self._store[(self._bucket, self.name)] = file_obj.read()
        self.uploads.append(
            {"content_type": content_type, "if_generation_match": if_generation_match}
        )

    def download_as_bytes(self) -> bytes:
        return self._store[(self._bucket, self.name)]

    def exists(self) -> bool:
        return (self._bucket, self.name) in self._store


class FakeBucket:
    def __init__(self, store: dict[tuple[str, str], bytes], name: str) -> None:
        self._store = store
        self.name = name
        self.blobs: dict[str, FakeBlob] = {}

    def blob(self, name: str) -> FakeBlob:
        self.blobs.setdefault(name, FakeBlob(self._store, self.name, name))
        return self.blobs[name]


class FakeClient:
    def __init__(self) -> None:
        self.store: dict[tuple[str, str], bytes] = {}
        self.buckets: dict[str, FakeBucket] = {}

    def bucket(self, name: str) -> FakeBucket:
        self.buckets.setdefault(name, FakeBucket(self.store, name))
        return self.buckets[name]


class FailingDeleteBlob:
    name = "automl-root/home_credit/baseline/a.json"

    def delete(self):
        raise RuntimeError("boom")


class FailingDeleteClient:
    def list_blobs(self, bucket, prefix):
        assert bucket == "automl-test-bucket"
        assert prefix == "automl-root/home_credit/baseline/"
        return [FailingDeleteBlob()]


def test_gcs_uri_helpers_parse_validate_and_join():
    assert gcs.is_gcs_uri("gs://bucket/path/to/object.json")
    assert not gcs.is_gcs_uri("https://example.com/object.json")
    assert gcs.parse_gcs_uri("gs://bucket/path/to/object.json") == (
        "bucket",
        "path/to/object.json",
    )
    assert gcs.join_uri("gs://bucket/root/", "/child", "file.json") == (
        "gs://bucket/root/child/file.json"
    )
    assert gcs.join_uri("gs://bucket", "child") == "gs://bucket/child"


def test_gcs_uri_helpers_reject_invalid_values():
    for value in ["bucket/path", "gs://", "gs://bucket", "gs://bucket/"]:
        try:
            gcs.parse_gcs_uri(value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected parse_gcs_uri to reject {value!r}")


def test_json_helpers_round_trip_through_injected_client():
    client = FakeClient()
    uri = "gs://bucket/path/manifest.json"
    payload = {"b": 2, "a": [1, 3]}

    gcs.write_json(uri, payload, client=client)

    assert json.loads(client.store[("bucket", "path/manifest.json")].decode("utf-8")) == payload
    assert gcs.read_json(uri, client=client) == payload
    assert gcs.blob_exists(uri, client=client)
    assert not gcs.blob_exists("gs://bucket/missing.json", client=client)
    assert (
        client.bucket("bucket").blob("path/manifest.json").uploads[-1]["if_generation_match"] == 0
    )


def test_json_write_requires_explicit_overwrite_for_existing_object():
    client = FakeClient()
    uri = "gs://bucket/path/manifest.json"

    gcs.write_json(uri, {"version": 1}, client=client)
    try:
        gcs.write_json(uri, {"version": 2}, client=client)
    except FileExistsError:
        pass
    else:
        raise AssertionError("expected write_json to protect existing objects by default")

    gcs.write_json(uri, {"version": 2}, client=client, overwrite=True)

    blob = client.bucket("bucket").blob("path/manifest.json")
    assert blob.uploads[-1]["if_generation_match"] is None
    assert gcs.read_json(uri, client=client) == {"version": 2}


def test_bytes_helpers_write_with_generation_protection():
    client = FakeClient()
    uri = "gs://bucket/path/model.pkl"

    gcs.write_bytes(uri, b"model-bytes", content_type="application/octet-stream", client=client)

    assert client.store[("bucket", "path/model.pkl")] == b"model-bytes"
    blob = client.bucket("bucket").blob("path/model.pkl")
    assert blob.uploads[-1] == {
        "content_type": "application/octet-stream",
        "if_generation_match": 0,
    }


def test_read_json_rejects_non_object_payload():
    client = FakeClient()
    client.store[("bucket", "array.json")] = b"[1, 2, 3]"

    try:
        gcs.read_json("gs://bucket/array.json", client=client)
    except ValueError as exc:
        assert "Expected JSON object" in str(exc)
    else:
        raise AssertionError("expected read_json to reject a non-object payload")


def test_delete_prefix_raises_when_a_blob_delete_fails():
    with pytest.raises(
        RuntimeError,
        match=r"failed to delete 'automl-root/home_credit/baseline/a\.json' under",
    ):
        gcs.delete_prefix(
            "gs://automl-test-bucket/automl-root/home_credit/baseline/",
            client=FailingDeleteClient(),
        )


def test_parquet_helpers_round_trip_through_injected_client():
    client = FakeClient()
    uri = "gs://bucket/path/data.parquet"
    df = pd.DataFrame({"id": pd.Series([1, 2], dtype="int64"), "value": ["a", "b"]})

    gcs.write_parquet(uri, df, client=client)
    loaded = gcs.read_parquet(uri, client=client)

    pd.testing.assert_frame_equal(loaded, df)
    assert client.bucket("bucket").blob("path/data.parquet").uploads[-1]["if_generation_match"] == 0


def test_parquet_write_requires_explicit_overwrite_for_existing_object():
    client = FakeClient()
    uri = "gs://bucket/path/data.parquet"
    first = pd.DataFrame({"id": [1], "value": ["a"]})
    second = pd.DataFrame({"id": [2], "value": ["b"]})

    gcs.write_parquet(uri, first, client=client)
    try:
        gcs.write_parquet(uri, second, client=client)
    except FileExistsError:
        pass
    else:
        raise AssertionError("expected write_parquet to protect existing objects by default")

    gcs.write_parquet(uri, second, client=client, overwrite=True)

    blob = client.bucket("bucket").blob("path/data.parquet")
    assert blob.uploads[-1]["if_generation_match"] is None
    pd.testing.assert_frame_equal(gcs.read_parquet(uri, client=client), second)
