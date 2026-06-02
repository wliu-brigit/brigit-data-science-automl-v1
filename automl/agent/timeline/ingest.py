"""Hook event ingestion for agent timelines."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from automl.agent.timeline.paths import _route, _timeline_path
from automl.project import Session, session as active_project_session


def handle_event(payload: dict[str, Any], *, session: Session | None = None) -> dict[str, Any]:
    """Append a Claude hook event to the active route timeline."""

    active = session if session is not None else active_project_session()
    route = _route(active)
    event = _event_from_hook_payload(route, payload)
    if event is None:
        return {"status": "ignored", "route": route}
    path = _append_event(active.config.repo_root, route, event)
    output: dict[str, Any] = {"status": "recorded", "timeline_path": str(path), "event": event}
    if _should_publish_trial_on_hook_event(event):
        output["trial_publish"] = _publish_trial_for_coder_stop(event, session=active)
    return output


def _event_from_hook_payload(route: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    hook_event_name = str(payload.get("hook_event_name") or "")
    if hook_event_name not in {"SubagentStart", "SubagentStop"}:
        return None
    agent_type = str(payload.get("agent_type") or "")
    event: dict[str, Any] = {
        **_now_event_fields(route, payload),
        "event": "start" if hook_event_name == "SubagentStart" else "end",
        "agent_id": str(payload.get("agent_id") or ""),
        "agent_type": agent_type,
        "phase": _phase_from_agent_type(agent_type),
        "transcript_path": str(payload.get("transcript_path") or ""),
    }
    for key in (
        "agent_transcript_path",
        "last_assistant_message",
        "trial_id",
        "run_id",
        "runner_execution_s",
        "runner_status",
    ):
        if payload.get(key) not in (None, ""):
            event[key] = payload[key]
    if "last_assistant_message" in event:
        event["last_assistant_message"] = str(event["last_assistant_message"])[:1000]
    return event


def _now_event_fields(route: str, hook_payload: dict[str, Any]) -> dict[str, Any]:
    now = float(hook_payload.get("time_s") or time.time())
    return {
        "schema_version": 1,
        "route": route,
        "timestamp": datetime.now(UTC).isoformat(timespec="milliseconds").replace(
            "+00:00",
            "Z",
        ),
        "time_s": now,
        "duration_unit": "seconds",
        "source": "claude_hook",
        "session_id": str(hook_payload.get("session_id") or ""),
        "cwd": str(hook_payload.get("cwd") or ""),
        "hook_event_name": str(hook_payload.get("hook_event_name") or ""),
    }


def _phase_from_agent_type(agent_type: object) -> str:
    value = str(agent_type or "")
    if value.endswith("automl-proposer") or value == "automl-proposer":
        return "proposer"
    if value.endswith("automl-coder") or value == "automl-coder":
        return "coder"
    return "unknown"


def _append_event(project_root: Path, route: str, event: dict[str, Any]) -> Path:
    path = _timeline_path(project_root, route)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")
    return path


def _should_publish_trial_on_hook_event(event: dict[str, Any]) -> bool:
    return (
        event.get("source") == "claude_hook"
        and event.get("event") == "end"
        and event.get("phase") == "coder"
        and bool(event.get("agent_id"))
        and bool(event.get("run_id"))
    )


def _publish_trial_for_coder_stop(
    event: dict[str, Any],
    *,
    session: Session,
) -> dict[str, Any]:
    session_id = str(event.get("session_id") or "")
    if not session_id:
        return {"status": "skipped_missing_session_id"}
    from automl.agent.timeline._publish import publish

    return publish(session_id=session_id, session=session)
