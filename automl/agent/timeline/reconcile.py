"""Timeline reconciliation and transcript parsing helpers."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from automl.mlflow import client as mlflow_client
from automl.mlflow import experiment as mlflow_experiment
from automl.project import Session


def _read_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            events.append(payload)
    return events


def _latest_session_id_from_events(events: list[dict[str, Any]]) -> str:
    for event in reversed(events):
        session_id = str(event.get("session_id") or "")
        if session_id:
            return session_id
    return "unknown_session"


def _events_for_session(events: list[dict[str, Any]], session_id: str) -> list[dict[str, Any]]:
    if not session_id or session_id == "unknown_session":
        return events
    return [event for event in events if str(event.get("session_id") or "") == session_id]


def _summarize_events(
    route: str,
    ordered: list[dict[str, Any]],
    *,
    session: Session | None = None,
) -> dict[str, Any]:
    events = sorted(ordered, key=lambda item: float(item.get("time_s") or 0.0))
    starts: dict[str, list[dict[str, Any]]] = {}
    spans: list[dict[str, Any]] = []
    unmatched_end_count = 0
    for event in events:
        agent_id = str(event.get("agent_id") or "")
        if not agent_id:
            continue
        event_kind = str(event.get("event") or "")
        if event_kind == "start":
            starts.setdefault(agent_id, []).append(event)
            continue
        if event_kind != "end":
            continue
        stack = starts.get(agent_id) or []
        if not stack:
            unmatched_end_count += 1
            continue
        started = stack.pop()
        start_s = float(started.get("time_s") or 0.0)
        end_s = float(event.get("time_s") or start_s)
        span = {
            "agent_id": agent_id,
            "agent_type": str(event.get("agent_type") or started.get("agent_type") or ""),
            "phase": str(event.get("phase") or started.get("phase") or ""),
            "start_s": start_s,
            "end_s": end_s,
            "duration_s": max(0.0, end_s - start_s),
            "agent_transcript_path": str(
                event.get("agent_transcript_path")
                or started.get("agent_transcript_path")
                or "",
            ),
            "tool_uses": int(event.get("tool_uses") or 0),
            "trial_id": str(event.get("trial_id") or ""),
            "run_id": str(event.get("run_id") or ""),
            "runner_execution_s": event.get("runner_execution_s"),
            "runner_status": str(event.get("runner_status") or ""),
        }
        spans.append(span)

    phase_durations: dict[str, float] = {}
    phase_tool_uses: dict[str, int] = {}
    iterations: list[dict[str, Any]] = []
    pending_proposers: list[dict[str, Any]] = []
    iteration_number = 0
    execution_matches = _match_coder_spans_to_executions(
        [span for span in spans if span.get("phase") == "coder"],
        _read_trial_executions(session),
    )
    tool_counts = _tool_counts_by_agent(spans)
    for span in spans:
        phase = str(span.get("phase") or "unknown")
        phase_durations[phase] = phase_durations.get(phase, 0.0) + float(span["duration_s"])
        agent_id = str(span["agent_id"])
        tool_uses = int(span.get("tool_uses") or tool_counts.get(agent_id, 0) or 0)
        if tool_uses:
            phase_tool_uses[phase] = phase_tool_uses.get(phase, 0) + tool_uses
        if phase == "proposer":
            span["tool_uses"] = tool_uses
            pending_proposers.append(span)
            continue
        if phase != "coder":
            continue
        iteration_number += 1
        matched = execution_matches.get(agent_id, {})
        item = {
            "iteration": iteration_number,
            "trial_id": span["trial_id"] or str(matched.get("trial_id") or ""),
            "run_id": span["run_id"] or str(matched.get("run_id") or ""),
            "phases": {"coder_s": span["duration_s"]},
            "agent_ids": {"coder": span["agent_id"]},
            "tool_uses": {"coder": tool_uses},
            "runner_execution_s": _optional_float(
                span["runner_execution_s"]
                if span["runner_execution_s"] not in (None, "")
                else matched.get("runner_execution_s")
            ),
            "runner_status": span["runner_status"] or str(matched.get("runner_status") or ""),
        }
        if pending_proposers:
            proposer = pending_proposers.pop(0)
            item["phases"]["proposer_s"] = proposer["duration_s"]
            item["agent_ids"]["proposer"] = proposer["agent_id"]
            item["tool_uses"]["proposer"] = proposer["tool_uses"]
        iterations.append(item)

    first = float(events[0].get("time_s") or 0.0) if events else 0.0
    last = float(events[-1].get("time_s") or first) if events else first
    return {
        "schema_version": 1,
        "route": route,
        "duration_unit": "seconds",
        "timing_source": "claude_hooks",
        "tool_count_unit": "tool_uses",
        "tool_count_source": "claude_subagent_transcripts",
        "event_count": len(events),
        "unmatched_start_count": sum(len(stack) for stack in starts.values()),
        "unmatched_end_count": unmatched_end_count,
        "total_s": max(0.0, last - first),
        "phase_durations_s": phase_durations,
        "phase_tool_uses": phase_tool_uses,
        "runner_execution_total_s": sum(
            float(item.get("runner_execution_s") or 0.0) for item in iterations
        ),
        "iterations": iterations,
        "events": events,
    }


def _read_trial_executions(session: Session | None) -> list[dict[str, Any]]:
    if session is None:
        return []
    try:
        with mlflow_client.bound_for(session, experiment_id=session.active_experiment_id):
            rows = mlflow_experiment.list_trials()
    except Exception:
        return []
    executions: list[dict[str, Any]] = []
    for row in rows:
        run_id = str(getattr(row, "run_id", "") or "")
        slug = str(getattr(row, "slug", "") or "")
        trial_number = getattr(row, "trial_number", None)
        trial_id = f"{trial_number}_{slug}" if trial_number is not None and slug else slug
        executions.append(
            {
                "trial_id": trial_id,
                "run_id": run_id,
                "trial_number": trial_number,
                "runner_execution_s": getattr(row, "training_time_s", None),
                "runner_status": str(
                    getattr(getattr(row, "status", ""), "value", "")
                    or getattr(row, "status", "")
                    or ""
                ),
                "start_s": _iso_to_epoch(getattr(row, "started_at", None)),
                "end_s": _iso_to_epoch(getattr(row, "ended_at", None)),
            }
        )
    return executions


def _match_coder_spans_to_executions(
    spans: list[dict[str, Any]],
    executions: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    matched: dict[str, dict[str, Any]] = {}
    unused = list(executions)
    for span in sorted(spans, key=lambda item: float(item.get("start_s") or 0.0)):
        agent_id = str(span.get("agent_id") or "")
        if not agent_id or not unused:
            continue
        start_s = float(span.get("start_s") or 0.0)
        end_s = float(span.get("end_s") or start_s)
        overlap = [
            execution
            for execution in unused
            if _execution_overlaps_span(execution, start_s=start_s, end_s=end_s)
        ]
        if overlap:
            selected = sorted(overlap, key=lambda item: float(item.get("start_s") or 0.0))[0]
        else:
            selected = sorted(
                unused,
                key=lambda item: (
                    float(item.get("start_s") or 0.0),
                    float(item.get("end_s") or 0.0),
                ),
            )[-1]
        unused.remove(selected)
        matched[agent_id] = selected
    return matched


def _execution_overlaps_span(execution: dict[str, Any], *, start_s: float, end_s: float) -> bool:
    run_start = float(execution.get("start_s") or 0.0)
    run_end = float(execution.get("end_s") or run_start)
    return start_s - 5.0 <= run_end and run_start <= end_s + 5.0


def _iso_to_epoch(value: object) -> float:
    if value in (None, ""):
        return 0.0
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def _agent_transcript_paths_by_id(events: list[dict[str, Any]]) -> dict[str, str]:
    paths: dict[str, str] = {}
    for event in events:
        agent_id = str(event.get("agent_id") or "")
        transcript_path = str(event.get("agent_transcript_path") or "")
        if agent_id and transcript_path:
            paths[agent_id] = transcript_path
    return paths


def _main_transcript_paths(events: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    paths: list[str] = []
    for event in events:
        path = str(event.get("transcript_path") or "")
        if path and path not in seen:
            seen.add(path)
            paths.append(path)
    return paths


def _tool_counts_by_agent(spans: list[dict[str, Any]]) -> dict[str, int]:
    return {
        str(span.get("agent_id") or ""): _count_tool_uses_in_transcript(
            str(span.get("agent_transcript_path") or "")
        )
        for span in spans
        if span.get("agent_id")
    }


def _count_tool_uses_in_transcript(transcript_path: str) -> int:
    return len(_read_tool_use_blocks(transcript_path))


def _tool_events_for_iteration(
    events: list[dict[str, Any]],
    iteration: dict[str, Any],
) -> list[dict[str, Any]]:
    transcript_paths = _agent_transcript_paths_by_id(events)
    agent_ids = iteration.get("agent_ids")
    if not isinstance(agent_ids, dict):
        return []
    output: list[dict[str, Any]] = []
    sequence = 1
    for phase in ("proposer", "coder"):
        agent_id = str(agent_ids.get(phase) or "")
        for block in _read_tool_use_blocks(transcript_paths.get(agent_id, "")):
            output.append(
                _sanitize_tool_use(
                    block,
                    sequence=sequence,
                    phase=phase,
                    agent_id=agent_id,
                )
            )
            sequence += 1
    return output


def _read_tool_use_blocks(transcript_path: str) -> list[dict[str, Any]]:
    if not transcript_path:
        return []
    path = Path(transcript_path)
    if not path.exists():
        return []
    blocks: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict) or event.get("type") != "assistant":
            continue
        message = event.get("message")
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                blocks.append(block)
    return blocks


def _sanitize_tool_use(
    block: dict[str, Any],
    *,
    sequence: int,
    phase: str,
    agent_id: str,
) -> dict[str, Any]:
    tool_input = block.get("input") if isinstance(block.get("input"), dict) else {}
    event: dict[str, Any] = {
        "sequence": sequence,
        "phase": phase,
        "agent_id": agent_id,
        "tool_name": str(block.get("name") or ""),
    }
    target = _tool_target(tool_input)
    if target:
        event["target"] = target
    description = str(tool_input.get("description") or "")
    if description:
        event["description"] = _safe_text(description)
    return event


def _tool_target(tool_input: dict[str, Any]) -> str:
    for key in ("file_path", "path", "notebook_path", "target"):
        value = tool_input.get(key)
        if isinstance(value, str) and value:
            return _safe_text(value)
    return ""


def _tool_counts_by_name(tool_events: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for event in tool_events:
        tool_name = str(event.get("tool_name") or "")
        if tool_name:
            counts[tool_name] = counts.get(tool_name, 0) + 1
    return counts


def _safe_text(value: object, *, limit: int = 300) -> str:
    text = str(value or "")
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _compact_iteration(item: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in item.items() if key not in {"events"}}


def _optional_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    return float(value)  # type: ignore[arg-type]
