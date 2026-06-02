from pathlib import Path

import pandas as pd
import pytest

from automl.data import DataSpec, LocalCSVSource, get_profile, materialize, profile
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
            gcs_prefix="automl-root",
            mlflow_tracking_uri=(tmp_path / "mlruns").as_uri(),
        )
    )


def _write_csv(tmp_path: Path) -> Path:
    csv_path = tmp_path / "tiny.csv"
    pd.DataFrame(
        {
            "row_id": list(range(1, 13)),
            "target": [0, 1] * 6,
            "amount": [100, 80, 95, 130, 50, 120, 99, 101, 75, 60, 42, 88],
            "segment": ["a", "b", "a", "c"] * 3,
        }
    ).to_csv(csv_path, index=False)
    return csv_path


def test_profile_writes_project_overview_artifacts_for_materialized_dataset(tmp_path, fake_gcs):
    active = _session(tmp_path, _write_csv(tmp_path))
    loaded = materialize(session=active)

    result = profile(session=active)
    restored = get_profile(loaded.id, session=active)

    assert result.dataset_id == loaded.id
    assert result.data_card_uri.endswith(f"/{loaded.id}/profile/data_card.json")
    assert result.data_observations_uri.endswith(f"/{loaded.id}/profile/data_observations.json")
    assert result.profile_manifest_uri.endswith(f"/{loaded.id}/profile/profile_manifest.json")
    assert result.chart_uris
    assert restored is not None
    assert restored.dataset_id == loaded.id
