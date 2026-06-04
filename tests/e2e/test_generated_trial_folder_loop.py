from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tests.e2e._gates import requires_live_e2e

from automl.agent import gather_proposer_context
from automl.agent.timeline import handle_event, publish
from automl.data import materialize
from automl.experiment import delete as delete_experiment
from automl.experiment import leaderboard
from automl.mlflow import tags
from automl.mlflow import trial as mlflow_trial
from automl.project import clear_session, use_project
from automl.runner import run_trial
from automl.trial import TrialStatus, create
from automl.agent import validate_proposal

pytestmark = [pytest.mark.e2e, pytest.mark.qa]


@requires_live_e2e("generated trial folder")
def test_generated_trial_folder_full_loop_gate():
    repo_root = Path(__file__).resolve().parents[2]
    stamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    namespace = f"qa-generated-trial-folder-{stamp}"
    experiment_id = f"generated-trial-folder-{stamp}"
    project = "example_homecredit"
    active = use_project(
        project,
        repo_root=repo_root,
        dry_run=True,
        namespace=namespace,
        experiment_id=experiment_id,
    )
    try:
        materialize(session=active)
        context = gather_proposer_context(session=active, n_top=3)
        assert context["project_name"] == project
        assert context["data_context"]["active_dataset"]

        proposal_payload = {
            "schema_version": 2,
            "slug": "generated_trial_folder",
            "strategy": "copied_project_baseline",
            "hypothesis": "A generated trial folder can run the committed baseline model.",
            "implementation_plan": [
                "Create a routed trial folder.",
                "Copy the committed baseline model into model.py.",
                "Run the folder through the runner.",
            ],
            "constraints": ["Use only approved project dependencies."],
            "required_dependencies": ["pandas", "numpy", "scikit-learn"],
        }
        report = validate_proposal(proposal=proposal_payload, session=active)
        assert report.passed, report.to_json()

        trial_dir = create(
            slug=proposal_payload["slug"],
            strategy=proposal_payload["strategy"],
            hypothesis=proposal_payload["hypothesis"],
            model_source=repo_root / "projects" / project / "model" / "__init__.py",
            proposal=proposal_payload,
            session=active,
        )
        metadata = json.loads((trial_dir / "metadata.json").read_text(encoding="utf-8"))
        assert metadata["slug"] == proposal_payload["slug"]
        assert "dry_run" not in metadata
        assert "run_mode" not in metadata

        result = run_trial(trial_dir, session=active)
        assert result.status == TrialStatus.FINISHED.value
        assert result.run_id

        source = mlflow_trial.artifacts.load_model_source(result.run_id)
        assert "HomeCreditLogisticModel" in source
        run_tags = mlflow_trial.get_tags(result.run_id)
        assert tags.MODEL_SOURCE_URI not in run_tags
        artifact_paths = {
            artifact.path for artifact in mlflow_trial.get_details(result.run_id).artifacts
        }
        assert "source/model.py" not in artifact_paths
        assert "agent/proposer/proposal.json" in artifact_paths
        assert any(
            path.startswith("model/code/trial_model_") and path.endswith(".py")
            for path in artifact_paths
        )
        assert not any("/__pycache__/" in path or path.endswith(".pyc") for path in artifact_paths)
        assert not any("/experiments/" in path for path in artifact_paths)
        assert not any("/notebooks/" in path for path in artifact_paths)

        board = leaderboard(session=active, n=5)
        assert any(row.run_id == result.run_id for row in board.rows)

        session_id = f"generated-trial-folder-{stamp}"
        handle_event(
            {
                "hook_event_name": "SubagentStart",
                "session_id": session_id,
                "agent_id": "agent-c",
                "agent_type": "automl-coder",
                "time_s": 1.0,
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
                "time_s": 2.0,
            },
            session=active,
        )
        published = publish(session_id=session_id, session=active)
        assert published["status"] == "published"
        assert published["trial_artifacts"]
        assert any(
            artifact.path == "agent/manifest.json"
            for artifact in mlflow_trial.get_details(result.run_id).artifacts
        )
    finally:
        try:
            delete_experiment(experiment_id, apply=True, session=active)
        finally:
            clear_session()
