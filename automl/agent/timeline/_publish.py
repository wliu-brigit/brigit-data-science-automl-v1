"""Publishing for reconciled agent timeline artifacts."""

from __future__ import annotations

import gzip
import json
import time
from pathlib import Path
from typing import Any

from automl.agent.timeline.paths import (
    _route,
    _route_segment,
    _session_dir,
    _timeline_path,
    _trial_dir,
)
from automl.agent.timeline.reconcile import (
    _agent_transcript_paths_by_id,
    _compact_iteration,
    _events_for_session,
    _final_assistant_message,
    _latest_session_id_from_events,
    _main_transcript_paths,
    _optional_float,
    _read_events,
    _summarize_events,
    _tool_counts_by_name,
    _tool_events_for_iteration,
)
from automl.mlflow import client as mlflow_client
from automl.mlflow import experiment as mlflow_experiment
from automl.mlflow import routing as mlflow_routing
from automl.mlflow import trial as mlflow_trial
from automl.project import Session, session as active_project_session
from automl.trial.timing_summary import enrich_agent_timing_summary
from automl.utils.io import gcs


def publish(
    *,
    session_id: str | None = None,
    session: Session | None = None,
) -> dict[str, Any]:
    """Stage and publish reconciled session/trial agent artifacts."""

    publish_started = time.monotonic()
    active = session if session is not None else active_project_session()
    route = _route(active)
    timeline_path = _timeline_path(active.config.repo_root, route)
    all_events = _read_events(timeline_path)
    resolved_session_id = session_id or _latest_session_id_from_events(all_events)
    events = _events_for_session(all_events, resolved_session_id)
    summary = _summarize_events(route, events, session=active)
    staged = _stage_publish_artifacts(
        project_root=active.config.repo_root,
        route=route,
        timeline_path=timeline_path,
        events=events,
        summary=summary,
        session_id=resolved_session_id,
    )
    gcs_refs = _publish_raw_artifacts_to_gcs(active, route=route, events=events, staged=staged)
    # Reconciliation + staging + GCS upload dominate publish; the trailing
    # MLflow JSON uploads below are excluded (the report is already written).
    summary["publish_s"] = max(0.0, time.monotonic() - publish_started)
    staged = _stage_publish_artifacts(
        project_root=active.config.repo_root,
        route=route,
        timeline_path=timeline_path,
        events=events,
        summary=summary,
        session_id=resolved_session_id,
        gcs_refs=gcs_refs or None,
    )
    _publish_to_mlflow(active, staged=staged)
    return {**staged, "status": "published"}


def _stage_publish_artifacts(
    *,
    project_root: Path,
    route: str,
    timeline_path: Path,
    events: list[dict[str, Any]],
    summary: dict[str, Any],
    session_id: str,
    gcs_refs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    session_dir = _session_dir(project_root, route, session_id)
    session_summary_path = session_dir / "report.json"
    session_summary = _session_summary_payload(
        route=route,
        session_id=session_id,
        summary=summary,
        gcs_refs=gcs_refs,
    )
    _write_json(session_summary_path, session_summary)
    trial_artifacts = []
    for iteration in summary.get("iterations", []):
        if not isinstance(iteration, dict):
            continue
        artifact = _stage_trial_artifact(
            project_root=project_root,
            route=route,
            session_id=session_id,
            events=events,
            iteration=iteration,
            gcs_refs=gcs_refs,
        )
        if artifact is not None:
            trial_artifacts.append(artifact)
    return {
        "route": route,
        "session_id": session_id,
        "timeline_path": str(timeline_path),
        "session_summary_path": str(session_summary_path),
        "session_summary": session_summary,
        "trial_artifacts": trial_artifacts,
    }


def _session_summary_payload(
    *,
    route: str,
    session_id: str,
    summary: dict[str, Any],
    gcs_refs: dict[str, Any] | None,
) -> dict[str, Any]:
    payload = {key: value for key, value in summary.items() if key not in {"events", "iterations"}}
    payload.update(
        {
            "artifact_kind": "session_summary",
            "route": route,
            "session_id": session_id,
            "iterations": [_compact_iteration(item) for item in summary.get("iterations", [])],
        }
    )
    if gcs_refs:
        payload["gcs"] = gcs_refs.get("session", {})
    return payload


def _stage_trial_artifact(
    *,
    project_root: Path,
    route: str,
    session_id: str,
    events: list[dict[str, Any]],
    iteration: dict[str, Any],
    gcs_refs: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    trial_id = str(iteration.get("trial_id") or "")
    run_id = str(iteration.get("run_id") or "")
    if not trial_id or not run_id:
        return None
    trial_dir = _trial_dir(project_root, route, session_id, trial_id, run_id)
    tool_events = _tool_events_for_iteration(events, iteration)
    proposer_tool_events = [
        event for event in tool_events if str(event.get("phase") or "") == "proposer"
    ]
    coder_tool_events = [event for event in tool_events if str(event.get("phase") or "") == "coder"]
    paths = {
        "agent_manifest_path": trial_dir / "manifest.json",
        "proposer_report_path": trial_dir / "proposer" / "report.json",
        "proposer_tool_events_path": trial_dir / "proposer" / "tool_events.json",
        "coder_report_path": trial_dir / "coder" / "report.json",
        "coder_tool_events_path": trial_dir / "coder" / "tool_events.json",
    }
    _write_json(
        paths["agent_manifest_path"],
        _agent_manifest_payload(
            route=route,
            session_id=session_id,
            iteration=iteration,
            gcs_refs=gcs_refs,
        ),
    )
    _write_json(
        paths["proposer_report_path"],
        _phase_report_payload(
            route=route,
            session_id=session_id,
            iteration=iteration,
            phase="proposer",
        ),
    )
    _write_json(
        paths["proposer_tool_events_path"],
        _tool_events_payload(
            route=route,
            session_id=session_id,
            iteration=iteration,
            phase="proposer",
            tool_events=proposer_tool_events,
        ),
    )
    _write_json(
        paths["coder_report_path"],
        _phase_report_payload(
            route=route,
            session_id=session_id,
            iteration=iteration,
            phase="coder",
        ),
    )
    _write_json(
        paths["coder_tool_events_path"],
        _tool_events_payload(
            route=route,
            session_id=session_id,
            iteration=iteration,
            phase="coder",
            tool_events=coder_tool_events,
        ),
    )
    message_paths = _stage_phase_messages(
        trial_dir=trial_dir,
        events=events,
        iteration=iteration,
    )
    return {
        "route": route,
        "trial_id": trial_id,
        "run_id": run_id,
        **{key: str(path) for key, path in paths.items()},
        **{key: str(path) for key, path in message_paths.items()},
    }


def _stage_phase_messages(
    *,
    trial_dir: Path,
    events: list[dict[str, Any]],
    iteration: dict[str, Any],
) -> dict[str, Path]:
    """Write each agent's full closing message as a per-phase markdown file.

    The free-flow report is the human-readable record of what the agent did
    and why; the JSON reports beside it carry only structured fields. Absent
    transcripts (or empty messages) stage nothing rather than an empty file.
    """
    transcript_paths = _agent_transcript_paths_by_id(events)
    agent_ids = iteration.get("agent_ids") if isinstance(iteration.get("agent_ids"), dict) else {}
    staged: dict[str, Path] = {}
    for phase in ("proposer", "coder"):
        agent_id = str(agent_ids.get(phase) or "")
        message = _final_assistant_message(transcript_paths.get(agent_id, ""))
        if not message:
            continue
        message_path = trial_dir / phase / "message.md"
        message_path.parent.mkdir(parents=True, exist_ok=True)
        message_path.write_text(message + "\n", encoding="utf-8")
        staged[f"{phase}_message_path"] = message_path
    return staged


def _agent_manifest_payload(
    *,
    route: str,
    session_id: str,
    iteration: dict[str, Any],
    gcs_refs: dict[str, Any] | None,
) -> dict[str, Any]:
    trial_id = str(iteration.get("trial_id") or "")
    run_id = str(iteration.get("run_id") or "")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "agent_manifest",
        "route": route,
        "session_id": session_id,
        "trial_id": trial_id,
        "run_id": run_id,
        "artifacts": {
            "manifest": "agent/manifest.json",
            "proposer_report": "agent/proposer/report.json",
            "proposer_message": "agent/proposer/message.md",
            "proposer_tool_events": "agent/proposer/tool_events.json",
            "coder_report": "agent/coder/report.json",
            "coder_message": "agent/coder/message.md",
            "coder_tool_events": "agent/coder/tool_events.json",
        },
    }
    key = f"{trial_id}:{run_id}"
    if gcs_refs and key in gcs_refs.get("trials", {}):
        payload["gcs"] = gcs_refs["trials"][key]
    return payload


def _phase_report_payload(
    *,
    route: str,
    session_id: str,
    iteration: dict[str, Any],
    phase: str,
) -> dict[str, Any]:
    phases = iteration.get("phases") if isinstance(iteration.get("phases"), dict) else {}
    tool_uses = iteration.get("tool_uses") if isinstance(iteration.get("tool_uses"), dict) else {}
    agent_ids = iteration.get("agent_ids") if isinstance(iteration.get("agent_ids"), dict) else {}
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "agent_report",
        "phase": phase,
        "route": route,
        "session_id": session_id,
        "iteration": iteration.get("iteration"),
        "trial_id": str(iteration.get("trial_id") or ""),
        "run_id": str(iteration.get("run_id") or ""),
        "duration_unit": "seconds",
        "tool_count_unit": "tool_uses",
        "agent_id": str(agent_ids.get(phase) or ""),
        "duration_s": _optional_float(phases.get(f"{phase}_s")),
        "tool_uses": int(tool_uses.get(phase) or 0),
    }
    if phase == "coder":
        payload["runner_execution_s"] = _optional_float(iteration.get("runner_execution_s"))
        payload["runner_status"] = str(iteration.get("runner_status") or "")
        if isinstance(iteration.get("runner_phases_s"), dict):
            payload["runner_phases_s"] = iteration["runner_phases_s"]
        if payload["duration_s"] is not None and payload["runner_execution_s"] is not None:
            payload["coder_non_runner_s"] = max(
                0.0,
                float(payload["duration_s"]) - float(payload["runner_execution_s"]),
            )
    return payload


def _tool_events_payload(
    *,
    route: str,
    session_id: str,
    iteration: dict[str, Any],
    phase: str,
    tool_events: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_kind": "agent_tool_events",
        "phase": phase,
        "route": route,
        "session_id": session_id,
        "iteration": iteration.get("iteration"),
        "trial_id": str(iteration.get("trial_id") or ""),
        "run_id": str(iteration.get("run_id") or ""),
        "event_count": len(tool_events),
        "tool_uses_by_name": _tool_counts_by_name(tool_events),
        "events": tool_events,
    }


def _publish_raw_artifacts_to_gcs(
    active: Session,
    *,
    route: str,
    events: list[dict[str, Any]],
    staged: dict[str, Any],
) -> dict[str, Any]:
    if not active.config.gcs_bucket or not active.config.gcs_prefix:
        return {}
    refs: dict[str, Any] = {"session": {}, "trials": {}}
    session_transcripts = _upload_session_transcripts_to_gcs(active, route=route, events=events)
    if session_transcripts:
        refs["session"]["raw_transcript_uris"] = session_transcripts
    for artifact in staged.get("trial_artifacts", []):
        if not isinstance(artifact, dict):
            continue
        run_id = str(artifact.get("run_id") or "")
        trial_id = str(artifact.get("trial_id") or "")
        if not run_id or not trial_id:
            continue
        with mlflow_client.bound_for(active, experiment_id=active.active_experiment_id):
            base_uri = mlflow_routing.bucket_uri_for(kind="agent_events", run_id=run_id).rstrip("/")
        raw_events_uri = f"{base_uri}/agent_timeline.jsonl"
        gcs.write_bytes(
            raw_events_uri,
            _jsonl_bytes(events),
            content_type="application/jsonl",
            overwrite=True,
        )
        refs["trials"][f"{trial_id}:{run_id}"] = {
            "raw_events_uri": raw_events_uri,
            "raw_transcript_uris": _transcript_refs_for_iteration(
                session_transcripts,
                artifact,
            ),
        }
    return refs if refs["trials"] else {}


def _upload_session_transcripts_to_gcs(
    active: Session,
    *,
    route: str,
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    del route
    if not events:
        return {}
    session_id = _latest_session_id_from_events(events)
    with mlflow_client.bound_for(active, experiment_id=active.active_experiment_id):
        base_uri = mlflow_routing.bucket_uri_for(kind="agent_events", run_id=session_id).rstrip("/")
    refs: dict[str, Any] = {"main": [], "subagents": {}}
    for index, transcript in enumerate(_main_transcript_paths(events), start=1):
        uri = f"{base_uri}/transcripts/main/main_{index}.jsonl.gz"
        if _write_compressed_transcript(uri, transcript):
            refs["main"].append(uri)
    for agent_id, transcript in sorted(_agent_transcript_paths_by_id(events).items()):
        uri = f"{base_uri}/transcripts/subagents/{_route_segment(agent_id)}.jsonl.gz"
        if _write_compressed_transcript(uri, transcript):
            refs["subagents"][agent_id] = uri
    return refs


def _write_compressed_transcript(uri: str, transcript_path: str) -> bool:
    path = Path(transcript_path)
    if not path.exists():
        return False
    gcs.write_bytes(
        uri,
        gzip.compress(path.read_bytes()),
        content_type="application/gzip",
        overwrite=True,
    )
    return True


def _transcript_refs_for_iteration(
    transcript_refs: dict[str, Any],
    artifact: dict[str, Any],
) -> dict[str, Any]:
    if not transcript_refs:
        return {}
    agent_ids = {}
    for report_key, phase in (
        ("proposer_report_path", "proposer"),
        ("coder_report_path", "coder"),
    ):
        try:
            payload = json.loads(Path(str(artifact[report_key])).read_text(encoding="utf-8"))
        except Exception:
            continue
        agent_id = str(payload.get("agent_id") or "")
        if agent_id:
            agent_ids[phase] = agent_id
    subagents = transcript_refs.get("subagents", {})
    return {
        "main": transcript_refs.get("main", []),
        "subagents": {
            phase: subagents[agent_id]
            for phase, agent_id in agent_ids.items()
            if isinstance(subagents, dict) and agent_id in subagents
        },
    }


def _publish_to_mlflow(active: Session, *, staged: dict[str, Any]) -> None:
    with mlflow_client.bound_for(active, experiment_id=active.active_experiment_id):
        session_id = str(staged.get("session_id") or "unknown_session")
        mlflow_experiment.log_json(
            f"agent/sessions/{_route_segment(session_id)}/report.json",
            staged["session_summary"],
        )
        for artifact in staged.get("trial_artifacts", []):
            if not isinstance(artifact, dict):
                continue
            run_id = str(artifact.get("run_id") or "")
            if not run_id:
                continue
            _log_trial_json_artifacts(run_id, artifact)
            _log_trial_message_artifacts(run_id, artifact)
            _log_trial_metrics(run_id, artifact)
            _log_trial_timing_summary(run_id, artifact, staged["session_summary"])


def _log_trial_json_artifacts(run_id: str, artifact: dict[str, Any]) -> None:
    mapping = {
        "agent/manifest.json": "agent_manifest_path",
        "agent/proposer/report.json": "proposer_report_path",
        "agent/proposer/tool_events.json": "proposer_tool_events_path",
        "agent/coder/report.json": "coder_report_path",
        "agent/coder/tool_events.json": "coder_tool_events_path",
    }
    for artifact_name, path_key in mapping.items():
        payload = json.loads(Path(str(artifact[path_key])).read_text(encoding="utf-8"))
        mlflow_trial.log_json(run_id, artifact_name, payload)


def _log_trial_message_artifacts(run_id: str, artifact: dict[str, Any]) -> None:
    """Log each staged agent closing message as a readable run artifact."""
    for phase in ("proposer", "coder"):
        path_value = artifact.get(f"{phase}_message_path")
        if not path_value:
            continue
        mlflow_client.log_artifact_file(
            run_id,
            f"agent/{phase}/message.md",
            Path(str(path_value)),
        )


def _log_trial_metrics(run_id: str, artifact: dict[str, Any]) -> None:
    proposer = json.loads(Path(str(artifact["proposer_report_path"])).read_text(encoding="utf-8"))
    coder = json.loads(Path(str(artifact["coder_report_path"])).read_text(encoding="utf-8"))
    metrics = {
        "agent.proposer_seconds": _optional_float(proposer.get("duration_s")),
        "agent.coder_seconds": _optional_float(coder.get("duration_s")),
        "agent.runner_execution_seconds": _optional_float(coder.get("runner_execution_s")),
        "agent.tool_calls": float(
            int(proposer.get("tool_uses") or 0) + int(coder.get("tool_uses") or 0)
        ),
    }
    for key, value in metrics.items():
        if value is not None:
            mlflow_trial.log_metric(run_id, key, value)


def _log_trial_timing_summary(
    run_id: str,
    artifact: dict[str, Any],
    session_summary: dict[str, Any],
) -> None:
    runner_summary = _read_trial_json_artifact(run_id, "timing/summary.json")
    if runner_summary is None:
        return
    iteration = _iteration_for_artifact(artifact, session_summary)
    if iteration is None:
        return
    spans = iteration.get("phase_spans") if isinstance(iteration.get("phase_spans"), dict) else {}
    proposer = spans.get("proposer") if isinstance(spans.get("proposer"), dict) else {}
    coder = spans.get("coder") if isinstance(spans.get("coder"), dict) else {}
    runner = (
        iteration.get("runner_span") if isinstance(iteration.get("runner_span"), dict) else None
    )
    steps = session_summary.get("steps") if isinstance(session_summary.get("steps"), list) else []
    timing = enrich_agent_timing_summary(
        runner_summary,
        setup_steps=_setup_steps_for_iteration(steps, proposer),
        proposer=proposer,
        proposal_handoff_steps=_handoff_steps_for_iteration(steps, proposer, coder),
        coder=coder,
        runner=runner,
        publish_s=float(session_summary.get("publish_s") or 0.0),
    )
    mlflow_trial.log_json(run_id, "timing/summary", timing)


def _read_trial_json_artifact(run_id: str, path: str) -> dict[str, Any] | None:
    try:
        local_path = mlflow_client.download_artifact(run_id, path)
    except Exception:
        return None
    if local_path is None:
        return None
    try:
        payload = json.loads(Path(local_path).read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _iteration_for_artifact(
    artifact: dict[str, Any],
    session_summary: dict[str, Any],
) -> dict[str, Any] | None:
    run_id = str(artifact.get("run_id") or "")
    trial_id = str(artifact.get("trial_id") or "")
    iterations = session_summary.get("iterations")
    if not isinstance(iterations, list):
        return None
    for iteration in iterations:
        if not isinstance(iteration, dict):
            continue
        if (
            str(iteration.get("run_id") or "") == run_id
            and str(iteration.get("trial_id") or "") == trial_id
        ):
            return iteration
    return None


def _setup_steps_for_iteration(
    steps: list[Any],
    proposer: dict[str, Any],
) -> list[dict[str, Any]]:
    proposer_start = float(proposer.get("start_s") or 0.0)
    if not proposer_start:
        return []
    return [
        _timing_step(step)
        for step in steps
        if isinstance(step, dict) and _step_end(step) <= proposer_start
    ]


def _handoff_steps_for_iteration(
    steps: list[Any],
    proposer: dict[str, Any],
    coder: dict[str, Any],
) -> list[dict[str, Any]]:
    proposer_end = float(proposer.get("end_s") or 0.0)
    coder_start = float(coder.get("start_s") or 0.0)
    if not proposer_end or not coder_start:
        return []
    return [
        _timing_step(step)
        for step in steps
        if isinstance(step, dict)
        and proposer_end <= _step_start(step)
        and _step_end(step) <= coder_start
    ]


def _timing_step(step: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": str(step.get("step") or ""),
        "duration_s": float(step.get("duration_s") or 0.0),
        "start_s": _step_start(step),
        "time_s": _step_end(step),
    }


def _step_start(step: dict[str, Any]) -> float:
    return float(
        step.get("start_s")
        or (float(step.get("time_s") or 0.0) - float(step.get("duration_s") or 0.0))
    )


def _step_end(step: dict[str, Any]) -> float:
    return float(step.get("time_s") or 0.0)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _jsonl_bytes(events: list[dict[str, Any]]) -> bytes:
    return b"".join(json.dumps(event, sort_keys=True).encode("utf-8") + b"\n" for event in events)
