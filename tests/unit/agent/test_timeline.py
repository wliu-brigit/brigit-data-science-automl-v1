import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from automl.eval import Auc, EvalSpec
from automl.mlflow import client, experiment, tags, trial
from automl.project import (
    BinaryClassification,
    ModelRoute,
    ModelsConfig,
    ProjectConfig,
    RunConfig,
    Session,
    Splits,
)

pytestmark = pytest.mark.unit


def _session(tmp_path: Path, *, gcs_bucket: str = "bucket") -> Session:
    route = ModelRoute("sonnet", "medium")
    return Session(
        config=ProjectConfig(
            project_name="demo",
            repo_root=tmp_path,
            project_dir=tmp_path / "projects" / "demo",
            config_path=tmp_path / "projects" / "demo" / "config.py",
            task=BinaryClassification(target="target"),
            eval_spec=EvalSpec(primary=Auc()),
            run_config=RunConfig(
                experiment_id="exp",
                splits=Splits({"train": ((0, 80),), "test": ((80, 100),)}),
                models=ModelsConfig(manager=route, proposer=route, coder=route),
                per_trial_seconds=120,
            ),
            gcs_bucket=gcs_bucket,
            gcs_prefix="root",
            mlflow_tracking_uri=(tmp_path / "mlruns").as_uri(),
        ),
        dry_run=True,
        namespace="qa",
    )


def _write_tool_transcript(path: Path, *, tool_name: str, target: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "name": tool_name,
                            "input": {"file_path": target},
                        }
                    ]
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_handle_event_appends_route_scoped_hook_event(tmp_path):
    from automl.agent.timeline import handle_event

    result = handle_event(
        {
            "hook_event_name": "SubagentStart",
            "session_id": "session-1",
            "agent_id": "agent-p",
            "agent_type": "automl-proposer",
            "cwd": str(tmp_path),
        },
        session=_session(tmp_path),
    )

    path = Path(result["timeline_path"])
    assert path == (
        tmp_path
        / ".cache"
        / "automl"
        / "tmp"
        / "timelines"
        / "qa"
        / "dry_run"
        / "demo"
        / "exp"
        / "agent_timeline.jsonl"
    )
    assert path.exists()
    assert "qa/dry_run/demo/exp" == result["event"]["route"]
    event = json.loads(path.read_text(encoding="utf-8").strip())
    assert event["event"] == "start"
    assert event["phase"] == "proposer"
    assert event["agent_type"] == "automl-proposer"
    assert event["session_id"] == "session-1"


def test_publish_stages_and_logs_agent_artifacts(monkeypatch, tmp_path):
    from automl.agent.timeline import handle_event, publish

    publish_module = importlib.import_module("automl.agent.timeline._publish")
    experiment_json: list[tuple[str, dict]] = []
    trial_json: list[tuple[str, str, dict]] = []
    trial_metrics: list[tuple[str, str, float]] = []
    gcs_writes: list[str] = []

    monkeypatch.setattr(
        publish_module.mlflow_experiment,
        "log_json",
        lambda name, payload: experiment_json.append((name, payload)),
    )
    monkeypatch.setattr(
        publish_module.mlflow_trial,
        "log_json",
        lambda run_id, name, payload: trial_json.append((run_id, name, payload)),
    )
    monkeypatch.setattr(
        publish_module.mlflow_trial,
        "log_metric",
        lambda run_id, key, value: trial_metrics.append((run_id, key, value)),
    )
    monkeypatch.setattr(
        publish_module.gcs,
        "write_bytes",
        lambda uri, payload, **kwargs: gcs_writes.append(uri),
    )

    active = _session(tmp_path)
    handle_event(
        {
            "hook_event_name": "SubagentStart",
            "session_id": "session-1",
            "agent_id": "agent-p",
            "agent_type": "automl-proposer",
        },
        session=active,
    )
    handle_event(
        {
            "hook_event_name": "SubagentStop",
            "session_id": "session-1",
            "agent_id": "agent-p",
            "agent_type": "automl-proposer",
        },
        session=active,
    )
    handle_event(
        {
            "hook_event_name": "SubagentStart",
            "session_id": "session-1",
            "agent_id": "agent-c",
            "agent_type": "automl-coder",
        },
        session=active,
    )
    handle_event(
        {
            "hook_event_name": "SubagentStop",
            "session_id": "session-1",
            "agent_id": "agent-c",
            "agent_type": "automl-coder",
            "trial_id": "1_baseline",
            "run_id": "run-123",
            "runner_execution_s": 4.5,
        },
        session=active,
    )

    result = publish(session_id="session-1", session=active)

    assert Path(result["session_summary_path"]).exists()
    assert experiment_json[0][0] == "agent/sessions/session-1/report.json"
    assert ("run-123", "agent/manifest.json") in {
        (run_id, name) for run_id, name, _payload in trial_json
    }
    assert ("run-123", "agent/proposer/report.json") in {
        (run_id, name) for run_id, name, _payload in trial_json
    }
    assert ("run-123", "agent/coder/report.json") in {
        (run_id, name) for run_id, name, _payload in trial_json
    }
    assert ("run-123", "agent.runner_execution_seconds", 4.5) in trial_metrics
    assert gcs_writes
    assert gcs_writes[0].startswith("gs://bucket/root/qa/dry_run/demo/exp/runs/")


def test_publish_backfills_trial_artifacts_when_real_hook_lacks_run_fields(tmp_path):
    from automl.agent.timeline import handle_event, publish
    from automl.mlflow import trial as mlflow_trial

    active = _session(tmp_path, gcs_bucket="")
    client.clear()
    try:
        with client.bound_for(active, experiment_id=active.active_experiment_id):
            experiment.ensure()
            with trial.active(slug="baseline", strategy="baseline") as run_id:
                trial.set_tags(run_id, {tags.TRIAL_NUMBER: "1", tags.TRIAL_ID: "1_baseline"})
                trial.log_metric(run_id, "eval.test.auc", 0.71)
            run = client.raw().get_run(run_id)

        start_s = float(run.info.start_time or 0) / 1000
        end_s = float(run.info.end_time or run.info.start_time or 0) / 1000
        proposer_transcript = tmp_path / "transcripts" / "proposer.jsonl"
        coder_transcript = tmp_path / "transcripts" / "coder.jsonl"
        _write_tool_transcript(proposer_transcript, tool_name="Read", target="config.py")
        _write_tool_transcript(coder_transcript, tool_name="Edit", target="model.py")

        handle_event(
            {
                "hook_event_name": "SubagentStart",
                "session_id": "session-2",
                "agent_id": "agent-p",
                "agent_type": "automl-proposer",
                "agent_transcript_path": str(proposer_transcript),
                "time_s": start_s - 30,
            },
            session=active,
        )
        handle_event(
            {
                "hook_event_name": "SubagentStop",
                "session_id": "session-2",
                "agent_id": "agent-p",
                "agent_type": "automl-proposer",
                "agent_transcript_path": str(proposer_transcript),
                "time_s": start_s - 20,
            },
            session=active,
        )
        handle_event(
            {
                "hook_event_name": "SubagentStart",
                "session_id": "session-2",
                "agent_id": "agent-c",
                "agent_type": "automl-coder",
                "agent_transcript_path": str(coder_transcript),
                "time_s": start_s - 10,
            },
            session=active,
        )
        handle_event(
            {
                "hook_event_name": "SubagentStop",
                "session_id": "session-2",
                "agent_id": "agent-c",
                "agent_type": "automl-coder",
                "agent_transcript_path": str(coder_transcript),
                "time_s": end_s + 10,
            },
            session=active,
        )

        published = publish(session_id="session-2", session=active)

        timeline_events = [
            json.loads(line)
            for line in Path(published["timeline_path"]).read_text(encoding="utf-8").splitlines()
        ]
        assert [
            (event["event"], event["phase"], event["agent_id"]) for event in timeline_events
        ] == [
            ("start", "proposer", "agent-p"),
            ("end", "proposer", "agent-p"),
            ("start", "coder", "agent-c"),
            ("end", "coder", "agent-c"),
        ]

        trial_artifact = published["trial_artifacts"][0]
        assert trial_artifact["run_id"] == run_id
        assert trial_artifact["proposer_report_path"].endswith("/proposer/report.json")
        assert trial_artifact["coder_report_path"].endswith("/coder/report.json")
        proposer_tool_events = json.loads(
            Path(trial_artifact["proposer_tool_events_path"]).read_text(encoding="utf-8")
        )
        coder_tool_events = json.loads(
            Path(trial_artifact["coder_tool_events_path"]).read_text(encoding="utf-8")
        )
        assert proposer_tool_events["events"] == [
            {
                "agent_id": "agent-p",
                "phase": "proposer",
                "sequence": 1,
                "target": "config.py",
                "tool_name": "Read",
            }
        ]
        assert coder_tool_events["events"] == [
            {
                "agent_id": "agent-c",
                "phase": "coder",
                "sequence": 2,
                "target": "model.py",
                "tool_name": "Edit",
            }
        ]
        with client.bound_for(active, experiment_id=active.active_experiment_id):
            artifact_paths = {artifact.path for artifact in mlflow_trial.list_artifacts(run_id)}
            metrics = mlflow_trial.get_metrics(run_id)
        assert "agent/proposer/report.json" in artifact_paths
        assert "agent/coder/report.json" in artifact_paths
        assert metrics["agent.proposer_seconds"] == 10.0
        assert metrics["agent.coder_seconds"] == pytest.approx(20.0, abs=0.01)
        assert metrics["agent.tool_calls"] == 2.0
    finally:
        client.clear()


def test_hook_script_imports_library_from_project_root_env(tmp_path):
    repo_root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        [sys.executable, str(repo_root / "agent-skills" / "hooks" / "agent_timeline.py"), "--help"],
        cwd=tmp_path,
        env={**os.environ, "AUTOML_PROJECT_ROOT": str(repo_root)},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
