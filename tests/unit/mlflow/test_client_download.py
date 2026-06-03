"""Pins for the seam download helper and the seam-wide HTTP retry cap.

Regression context (2026-06-02): tracking servers below MLflow 3.12 answer a
download of a *missing* artifact with a retryable HTTP 500, and MLflow's
default budget (7 retries, backoff factor 2) sleeps ~254s before surfacing the
error — an apparent hang. The seam therefore (a) caps the retry budget at
import, and (b) lists before downloading so absence never reaches the doomed
download call at all.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass

import pytest

from automl.errors import StorageError
from automl.mlflow import client

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def clear_mlflow_binding():
    client.clear()
    yield
    client.clear()


@dataclass
class _Item:
    path: str
    is_dir: bool = False


class _FakeMlflowClient:
    def __init__(self, existing: list[str]):
        self.existing = list(existing)
        self.list_calls: list[tuple[str, str | None]] = []
        self.download_calls: list[tuple[str, str]] = []

    def list_artifacts(self, run_id: str, path: str | None = None):
        self.list_calls.append((run_id, path))
        prefix = f"{path}/" if path else ""
        return [_Item(item) for item in self.existing if item.startswith(prefix)]

    def download_artifacts(self, run_id: str, artifact_path: str) -> str:
        self.download_calls.append((run_id, artifact_path))
        return f"/tmp/downloads/{artifact_path}"


def test_seam_import_caps_mlflow_http_retry_budget(monkeypatch):
    monkeypatch.delenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", raising=False)

    importlib.reload(client)

    assert os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] == client.HTTP_MAX_RETRIES == "1"


def test_seam_import_respects_operator_retry_override(monkeypatch):
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "4")

    importlib.reload(client)

    assert os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] == "4"


def test_download_artifact_absent_returns_none_without_downloading(monkeypatch):
    fake = _FakeMlflowClient(existing=["datasets/v2/source_trace/source_identity.json"])
    monkeypatch.setattr(client, "raw", lambda: fake)

    result = client.download_artifact("run-1", "datasets/v2/profile/profile_manifest.json")

    assert result is None
    # Absence must never reach download_artifacts — that call is what 500s and
    # retries on pre-3.12 servers.
    assert fake.download_calls == []
    assert fake.list_calls == [("run-1", "datasets/v2/profile")]


def test_download_artifact_absent_required_raises_storage_error(monkeypatch):
    fake = _FakeMlflowClient(existing=[])
    monkeypatch.setattr(client, "raw", lambda: fake)

    with pytest.raises(StorageError, match="not found"):
        client.download_artifact("run-1", "logs/errors/report.json", required=True)

    assert fake.download_calls == []


def test_download_artifact_present_downloads(monkeypatch):
    fake = _FakeMlflowClient(existing=["logs/errors/report.json"])
    monkeypatch.setattr(client, "raw", lambda: fake)

    result = client.download_artifact("run-1", "logs/errors/report.json")

    assert result == "/tmp/downloads/logs/errors/report.json"
    assert fake.download_calls == [("run-1", "logs/errors/report.json")]


def test_download_artifact_root_level_path_lists_root(monkeypatch):
    fake = _FakeMlflowClient(existing=["model.pkl"])
    monkeypatch.setattr(client, "raw", lambda: fake)

    assert client.download_artifact("run-1", "model.pkl") == "/tmp/downloads/model.pkl"
    assert fake.list_calls == [("run-1", None)]
