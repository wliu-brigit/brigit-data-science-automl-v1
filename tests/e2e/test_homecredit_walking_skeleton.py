from __future__ import annotations

from pathlib import Path

import pytest

from tests.e2e._gates import requires_live_e2e

from automl.data import materialize
from automl.mlflow import client as mlflow_client
from automl.mlflow import tags
from automl.project import clear_session, use_project
from automl.runner import run_trial
from automl.trial import TrialStatus

pytestmark = [pytest.mark.e2e, pytest.mark.qa]


@requires_live_e2e("Home Credit walking skeleton")
def test_homecredit_walking_skeleton_external_gate():
    repo_root = Path(__file__).resolve().parents[2]
    active = use_project("example_homecredit", repo_root=repo_root)
    try:
        materialize(session=active)

        result = run_trial("example_homecredit", session=active)

        assert result.status == TrialStatus.FINISHED.value
        assert result.run_id
        assert "auc" in result.metrics
        run = mlflow_client.raw().get_run(result.run_id)
        _assert_trial_artifact_exists(result.run_id, run.data.tags[tags.DATA_CONTRACT_URI])
        _assert_runs_uri_exists(run.data.tags[tags.MODEL_URI])
        _assert_trial_artifact_exists(result.run_id, run.data.tags[tags.eval_uri("test")])
        _assert_trial_artifact_exists(result.run_id, run.data.tags[tags.eval_uri("train")])
        assert run.data.tags[tags.TRIAL_SLUG] == "homecredit_logistic"
        assert run.data.tags[tags.TRIAL_ID] == result.trial_id
        assert not any(key.startswith("automl.trial") for key in run.data.tags)
    finally:
        clear_session()


def _assert_trial_artifact_exists(run_id: str, artifact_path: str) -> None:
    local_path = mlflow_client.raw().download_artifacts(run_id, artifact_path)
    assert Path(local_path).exists()


def _assert_runs_uri_exists(uri: str) -> None:
    assert uri.startswith("runs:/")
    _, remainder = uri.split("runs:/", 1)
    run_id, artifact_path = remainder.strip("/").split("/", 1)
    _assert_trial_artifact_exists(run_id, artifact_path)
