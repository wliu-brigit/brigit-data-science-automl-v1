import json
from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd
import pytest

from automl.data import (
    DataSpec,
    GCSParquetSource,
    LocalCSVSource,
    build_dataset,
    list_datasets,
    load_dataset,
    load_dataset_by_id,
    materialize,
)
from automl.errors import DataError
from automl.mlflow import client as mlflow_client
from automl.project import (
    BinaryClassification,
    ModelRoute,
    ModelsConfig,
    ProjectConfig,
    RunConfig,
    Session,
    Splits,
)
from automl.utils.io import gcs

pytestmark = pytest.mark.integration


class FakeBlob:
    def __init__(self, store: dict[tuple[str, str], bytes], bucket: str, name: str) -> None:
        self._store = store
        self._bucket = bucket
        self.name = name

    def upload_from_string(
        self,
        data: str | bytes,
        *,
        content_type: str | None = None,
        if_generation_match: int | None = None,
    ) -> None:
        if if_generation_match == 0 and (self._bucket, self.name) in self._store:
            raise FileExistsError(f"object already exists: {self._bucket}/{self.name}")
        self._store[(self._bucket, self.name)] = (
            data if isinstance(data, bytes) else data.encode("utf-8")
        )
        self._store[("__writes__", f"{self._bucket}/{self.name}")] = str(
            _write_count(self._store, self._bucket, self.name) + 1
        ).encode("utf-8")

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
        self._store[("__writes__", f"{self._bucket}/{self.name}")] = str(
            _write_count(self._store, self._bucket, self.name) + 1
        ).encode("utf-8")

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

    def write_count(self, bucket: str, name: str) -> int:
        return _write_count(self.store, bucket, name)


def _write_count(store: dict[tuple[str, str], bytes], bucket: str, name: str) -> int:
    return int(store.get(("__writes__", f"{bucket}/{name}"), b"0").decode("utf-8"))


@pytest.fixture
def fake_gcs(monkeypatch) -> FakeGCSClient:
    fake = FakeGCSClient()
    monkeypatch.setattr(gcs, "_gcs_client", lambda: fake)
    yield fake
    mlflow_client.clear()


def _models() -> ModelsConfig:
    route = ModelRoute("sonnet", "medium")
    return ModelsConfig(manager=route, proposer=route, coder=route)


def _session(tmp_path: Path, csv_path: Path) -> Session:
    spec = DataSpec(source=LocalCSVSource(csv_path=csv_path, hash_key="row_id"))
    return _session_for_spec(tmp_path, spec)


def _session_for_spec(tmp_path: Path, spec: DataSpec) -> Session:
    return Session(
        config=ProjectConfig(
            project_name="demo",
            repo_root=tmp_path,
            project_dir=tmp_path / "projects" / "demo",
            config_path=tmp_path / "projects" / "demo" / "config.py",
            task=BinaryClassification(target="target"),
            data_spec=spec,
            run_config=RunConfig(
                experiment_id="baseline",
                splits=Splits({"train": ((0, 50),), "test": ((50, 100),)}),
                models=_models(),
                per_trial_seconds=120,
            ),
            gcs_bucket="automl-test-bucket",
            gcs_prefix="",
            mlflow_tracking_uri=(tmp_path / "mlruns").as_uri(),
        )
    )


def _write_tiny_csv(tmp_path: Path) -> Path:
    csv_path = tmp_path / "tiny.csv"
    pd.DataFrame(
        {
            "row_id": list(range(1, 11)),
            "target": [0, 1] * 5,
            "amount": [100, 80, 95, 130, 50, 120, 99, 101, 75, 60],
        }
    ).to_csv(csv_path, index=False)
    return csv_path


def test_materialize_writes_dataset_index_and_loads_train_test_slices(tmp_path, fake_gcs):
    active = _session(tmp_path, _write_tiny_csv(tmp_path))

    loaded = materialize(session=active)
    index = list_datasets(session=active)
    train = load_dataset(split_name="train", session=active)
    test = load_dataset_by_id(loaded.id, split_name="test", session=active)

    assert loaded.id.startswith("v1_")
    assert loaded.dataset in index.datasets
    assert index.active.id == loaded.id
    assert loaded.dataset.manifest_gcs_uri.startswith(
        "gs://automl-test-bucket/demo/baseline/data/datasets/"
    )
    assert loaded.dataset.registry_gcs_uri.endswith("/feature_registry.csv")
    registry_bucket, registry_blob = gcs.parse_gcs_uri(loaded.dataset.registry_gcs_uri)
    assert (registry_bucket, registry_blob) in fake_gcs.store
    assert not registry_blob.endswith("registry.json")
    assert ("automl-test-bucket", "demo/baseline/data/dataset_index.json") in fake_gcs.store
    raw_index = json.loads(
        fake_gcs.store[("automl-test-bucket", "demo/baseline/data/dataset_index.json")].decode(
            "utf-8"
        )
    )
    assert "active_dataset_id" not in raw_index
    with mlflow_client.bound_for(active, experiment_id=active.active_experiment_id):
        experiment = mlflow_client.raw().get_experiment_by_name("demo/baseline")
        assert experiment is not None
        runs = mlflow_client.raw().search_runs([experiment.experiment_id])
        local_path = mlflow_client.raw().download_artifacts(
            runs[0].info.run_id,
            f"datasets/{loaded.id}/source_trace/source_identity.json",
        )
        latest_path = mlflow_client.raw().download_artifacts(
            runs[0].info.run_id,
            "datasets/latest.json",
        )
        index_path = mlflow_client.raw().download_artifacts(
            runs[0].info.run_id,
            "datasets/index.json",
        )
    source_identity = json.loads(Path(local_path).read_text(encoding="utf-8"))
    assert source_identity["kind"] == "local_csv"
    assert source_identity["csv_path"] == str(active.config.data_spec.source.csv_path)
    latest = json.loads(Path(latest_path).read_text(encoding="utf-8"))
    assert latest["dataset_id"] == loaded.id
    overview_index = json.loads(Path(index_path).read_text(encoding="utf-8"))
    assert overview_index["active_dataset_id"] == loaded.id
    assert [item["id"] for item in overview_index["datasets"]] == [loaded.id]
    assert set(train.df["SPLIT_PCT"]).isdisjoint(set(test.df["SPLIT_PCT"]))
    assert train.n_rows + test.n_rows == loaded.n_rows
    assert train.split_name == "train"
    assert test.split_name == "test"
    assert loaded.registry.to_dataframe().shape[0] == loaded.dataset.n_columns


def test_materialize_routes_dataset_objects_by_dry_run_and_namespace(tmp_path, fake_gcs):
    active = _session(tmp_path, _write_tiny_csv(tmp_path))

    real = materialize(session=active)
    dry = materialize(session=replace(active, dry_run=True))
    qa = materialize(session=replace(active, namespace="qa"))
    qa_dry = materialize(session=replace(active, namespace="qa", dry_run=True))

    assert real.dataset.manifest_gcs_uri.startswith(
        "gs://automl-test-bucket/demo/baseline/data/datasets/"
    )
    assert dry.dataset.manifest_gcs_uri.startswith(
        "gs://automl-test-bucket/dry_run/demo/baseline/data/datasets/"
    )
    assert qa.dataset.manifest_gcs_uri.startswith(
        "gs://automl-test-bucket/qa/demo/baseline/data/datasets/"
    )
    assert qa_dry.dataset.manifest_gcs_uri.startswith(
        "gs://automl-test-bucket/qa/dry_run/demo/baseline/data/datasets/"
    )
    assert ("automl-test-bucket", "demo/baseline/data/dataset_index.json") in fake_gcs.store
    assert ("automl-test-bucket", "dry_run/demo/baseline/data/dataset_index.json") in fake_gcs.store
    assert ("automl-test-bucket", "qa/demo/baseline/data/dataset_index.json") in fake_gcs.store
    assert (
        "automl-test-bucket",
        "qa/dry_run/demo/baseline/data/dataset_index.json",
    ) in fake_gcs.store


@dataclass(frozen=True)
class TraceCSVSource(LocalCSVSource):
    trace_file: Path | None = None

    def artifact_files(self, pipeline) -> dict[str, Path]:
        assert pipeline.session.project_name == "demo"
        if self.trace_file is None:
            return {}
        return {"source.sql": self.trace_file}


def test_materialize_logs_source_trace_artifacts_to_experiment_overview(tmp_path, fake_gcs):
    trace_file = tmp_path / "source.sql"
    trace_file.write_text("select 1", encoding="utf-8")
    spec = DataSpec(
        source=TraceCSVSource(
            csv_path=_write_tiny_csv(tmp_path),
            hash_key="row_id",
            trace_file=trace_file,
        )
    )
    active = _session_for_spec(tmp_path, spec)

    loaded = materialize(session=active)

    with mlflow_client.bound_for(active, experiment_id=active.active_experiment_id):
        experiment = mlflow_client.raw().get_experiment_by_name("demo/baseline")
        assert experiment is not None
        runs = mlflow_client.raw().search_runs(
            [experiment.experiment_id],
            filter_string="tags.`run.kind` = 'experiment_overview'",
        )
        assert len(runs) == 1
        local_path = mlflow_client.raw().download_artifacts(
            runs[0].info.run_id,
            f"datasets/{loaded.id}/source_trace/source.sql",
        )
    assert Path(local_path).read_text(encoding="utf-8") == "select 1"


def test_materialize_reuses_existing_complete_dataset_without_rewriting_objects(
    tmp_path,
    fake_gcs,
):
    active = _session(tmp_path, _write_tiny_csv(tmp_path))

    first = materialize(session=active)
    tracked_uris = (
        first.dataset.data_gcs_uri,
        first.dataset.registry_gcs_uri,
        first.dataset.manifest_gcs_uri,
    )
    write_counts = {uri: fake_gcs.write_count(*gcs.parse_gcs_uri(uri)) for uri in tracked_uris}

    second = materialize(session=active)

    assert second.id == first.id
    assert {
        uri: fake_gcs.write_count(*gcs.parse_gcs_uri(uri)) for uri in tracked_uris
    } == write_counts


def test_materialize_refuses_partial_existing_dataset_objects(tmp_path, fake_gcs):
    active = _session(tmp_path, _write_tiny_csv(tmp_path))
    loaded = materialize(session=active)
    registry_bucket, registry_blob = gcs.parse_gcs_uri(loaded.dataset.registry_gcs_uri)
    del fake_gcs.store[(registry_bucket, registry_blob)]

    with pytest.raises(DataError, match="partial dataset objects"):
        materialize(session=active)


def test_load_dataset_runs_l2_validation_and_rejects_corrupt_manifest(tmp_path, fake_gcs):
    active = _session(tmp_path, _write_tiny_csv(tmp_path))
    loaded = materialize(session=active)
    manifest_uri = loaded.dataset.manifest_gcs_uri
    bucket, blob = gcs.parse_gcs_uri(manifest_uri)
    manifest = json.loads(fake_gcs.store[(bucket, blob)].decode("utf-8"))
    manifest["component_hashes"]["data_content"] = "sha256:corrupt"
    fake_gcs.store[(bucket, blob)] = json.dumps(manifest).encode("utf-8")

    with pytest.raises(DataError, match="data_content"):
        load_dataset_by_id(loaded.id, session=active)


def test_build_dataset_reads_gcs_parquet_source_without_materializing_objects(tmp_path, fake_gcs):
    raw_uri = "gs://automl-test-bucket/raw/tiny.parquet"
    raw_df = pd.DataFrame(
        {
            "row_id": list(range(1, 7)),
            "target": [0, 1, 0, 1, 0, 1],
            "amount": [100, 80, 95, 130, 50, 120],
        }
    )
    gcs.write_parquet(raw_uri, raw_df)
    spec = DataSpec(source=GCSParquetSource(gcs_uri=raw_uri, hash_key="row_id"))

    loaded = build_dataset(session=_session_for_spec(tmp_path, spec))

    assert loaded.dataset.source_identity["kind"] == "gcs_parquet"
    assert loaded.dataset.hash_key == ("row_id",)
    assert "SPLIT_PCT" in loaded.df.columns
    assert not any(
        blob.startswith("demo/baseline/data/datasets/") for bucket, blob in fake_gcs.store
    )
    assert ("automl-test-bucket", "demo/baseline/data/dataset_index.json") not in fake_gcs.store
