from __future__ import annotations

import pytest

from automl.mlflow import client
from automl.project import Session
from automl.project.config import ProjectConfig

pytestmark = pytest.mark.unit


def _http_session(project_name: str, experiment_id: str | None) -> Session:
    config = ProjectConfig(
        project_name=project_name,
        mlflow_tracking_uri="https://mlflow.example.com/",
        gcs_prefix="automl-root",
    )
    return Session(config=config, experiment_id=experiment_id)


def test_session_mlflow_urls_reflect_self_even_when_another_session_is_bound(monkeypatch):
    client.clear()
    numeric_ids = {"alpha/expA": "1", "beta/expB": "2", "beta/000_overview": "20"}

    class _Experiment:
        def __init__(self, experiment_id: str) -> None:
            self.experiment_id = experiment_id

    monkeypatch.setattr(
        client,
        "get_experiment_by_name",
        lambda name: _Experiment(numeric_ids[name]) if name in numeric_ids else None,
    )

    beta = _http_session("beta", "expB")

    # Bind alpha as the active/global session.
    client.bind(
        tracking_uri="https://mlflow.example.com/",
        bucket="",
        gcs_prefix="automl-root",
        project_name="alpha",
        experiment_id="expA",
    )

    # beta's methods must describe beta, not the globally-bound alpha.
    assert beta.mlflow_experiment_url() == "https://mlflow.example.com/#/experiments/2"
    assert beta.mlflow_project_url() == "https://mlflow.example.com/#/experiments/20"

    # ...and the global alpha binding is restored afterward.
    assert client.experiment_url() == "https://mlflow.example.com/#/experiments/1"
    client.clear()


def test_session_mlflow_experiment_url_is_empty_without_an_experiment(monkeypatch):
    client.clear()
    monkeypatch.setattr(client, "get_experiment_by_name", lambda *_: None)

    explore = _http_session("solo", None)  # no experiment_id, no run_config

    assert explore.mlflow_experiment_url() == ""
    client.clear()
