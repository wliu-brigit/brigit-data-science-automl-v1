from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tests.e2e._gates import requires_live_e2e

from automl.agent import build_launch, gather_proposer_context
from automl.agent.timeline import handle_event, publish
from automl.data import materialize
from automl.experiment import delete as delete_experiment
from automl.mlflow import trial as mlflow_trial
from automl.project import clear_session, use_project
from automl.runner import run_trial
from automl.trial import TrialStatus
from automl.utils.io import gcs
from automl.agent import validate_proposal

pytestmark = [pytest.mark.e2e, pytest.mark.qa]


@requires_live_e2e("agent timeline hooks")
def test_agent_timeline_hooks_gate():
    repo_root = Path(__file__).resolve().parents[2]
    stamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    namespace = f"qa/agent-timeline-hooks-{stamp}"
    experiment_id = f"agent-timeline-hooks-{stamp}"
    active = use_project(
        "example_homecredit",
        repo_root=repo_root,
        namespace=namespace,
        experiment_id=experiment_id,
    )
    try:
        materialize(session=active)
        context = gather_proposer_context(session=active, n_top=3)
        assert context["project_name"] == "example_homecredit"
        assert context["data_context"]["active_dataset"]

        proposal_payload = {
            "schema_version": 2,
            "slug": "baseline",
            "strategy": "baseline",
            "hypothesis": "Exercise the agent timeline proposal handoff contract.",
            "implementation_plan": ["Run the committed project baseline model."],
            "constraints": ["Do not read test data directly."],
            "required_dependencies": ["pandas", "numpy", "scikit-learn"],
        }
        report = validate_proposal(proposal=proposal_payload, session=active)
        assert report.passed, report.to_json()

        launch = build_launch(
            session=active,
            automl_args=[],
            max_budget_usd="1",
            output_format="json",
            claude_bin="claude",
        )
        assert launch.env["AUTOML_PROJECT"] == "example_homecredit"
        assert launch.env["AUTOML_EXPERIMENT_ID"] == experiment_id
        assert launch.env["AUTOML_NAMESPACE"] == namespace

        result = run_trial("example_homecredit", session=active)
        assert result.status == TrialStatus.FINISHED.value

        session_id = f"agent-timeline-hooks-{stamp}"
        handle_event(
            {
                "hook_event_name": "SubagentStart",
                "session_id": session_id,
                "agent_id": "agent-p",
                "agent_type": "automl-proposer",
                "time_s": 1.0,
            },
            session=active,
        )
        handle_event(
            {
                "hook_event_name": "SubagentStop",
                "session_id": session_id,
                "agent_id": "agent-p",
                "agent_type": "automl-proposer",
                "time_s": 2.0,
            },
            session=active,
        )
        handle_event(
            {
                "hook_event_name": "SubagentStart",
                "session_id": session_id,
                "agent_id": "agent-c",
                "agent_type": "automl-coder",
                "time_s": 3.0,
            },
            session=active,
        )
        handle_event(
            {
                "hook_event_name": "SubagentStop",
                "session_id": session_id,
                "agent_id": "agent-c",
                "agent_type": "automl-coder",
                "trial_id": result.trial_id,
                "run_id": result.run_id,
                "runner_execution_s": 1.0,
                "time_s": 5.0,
            },
            session=active,
        )
        published = publish(session_id=session_id, session=active)

        artifact_paths = {artifact.path for artifact in mlflow_trial.list_artifacts(result.run_id)}
        assert "agent/manifest.json" in artifact_paths
        assert "agent/coder/report.json" in artifact_paths
        metrics = mlflow_trial.get_metrics(result.run_id)
        assert metrics["agent.runner_execution_seconds"] == 1.0

        local_manifest = json.loads(
            Path(published["trial_artifacts"][0]["agent_manifest_path"]).read_text(encoding="utf-8")
        )
        gcs_uri = local_manifest["gcs"]["raw_events_uri"]
        assert gcs_uri.startswith("gs://")
        assert gcs.blob_exists(gcs_uri)
        trial_manifest = mlflow_trial.get_details(result.run_id)
        assert any(artifact.path == "agent/manifest.json" for artifact in trial_manifest.artifacts)
    finally:
        try:
            delete_experiment(experiment_id, apply=True, session=active)
        finally:
            clear_session()
