"""Canonical trial timing summary helpers."""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Mapping


def round_seconds(value: object) -> float:
    """Round a seconds value to five decimal places."""
    numeric = Decimal(str(float(value or 0.0)))
    return float(numeric.quantize(Decimal("0.00001"), rounding=ROUND_HALF_UP))


def build_runner_timing_summary(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Build the canonical runner-only v2 timing summary."""
    if int(snapshot.get("schema_version", 1) or 1) == 2 and isinstance(
        snapshot.get("phase_details"),
        Mapping,
    ):
        return normalize_timing_summary(snapshot)
    total = round_seconds(snapshot.get("total_seconds"))
    raw_phases = snapshot.get("phases")
    phases = raw_phases if isinstance(raw_phases, Mapping) else {}
    runner_phases = {str(name): round_seconds(value) for name, value in phases.items()}
    return {
        "schema_version": 2,
        "unit": "seconds",
        "total_seconds": total,
        "phases": {"runner": total},
        "phase_details": {
            "runner": {
                "total_seconds": total,
                "phases": runner_phases,
            }
        },
    }


def normalize_timing_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize any v2-like timing payload to the canonical summary shape."""
    raw_phases = payload.get("phases")
    phases = raw_phases if isinstance(raw_phases, Mapping) else {}
    raw_details = payload.get("phase_details")
    details = raw_details if isinstance(raw_details, Mapping) else {}
    normalized_details: dict[str, Any] = {}
    for name, value in phases.items():
        phase_name = str(name)
        total = round_seconds(value)
        detail = details.get(phase_name)
        if isinstance(detail, Mapping):
            normalized_detail: dict[str, Any] = {
                "total_seconds": round_seconds(detail.get("total_seconds", total))
            }
            detail_phases = detail.get("phases")
            if isinstance(detail_phases, Mapping):
                normalized_detail["phases"] = {
                    str(key): round_seconds(duration) for key, duration in detail_phases.items()
                }
            normalized_details[phase_name] = normalized_detail
        else:
            normalized_details[phase_name] = {"total_seconds": total}
    return {
        "schema_version": 2,
        "unit": "seconds",
        "total_seconds": round_seconds(payload.get("total_seconds")),
        "phases": {str(name): round_seconds(value) for name, value in phases.items()},
        "phase_details": normalized_details,
    }


def enrich_agent_timing_summary(
    runner_summary: Mapping[str, Any],
    *,
    setup_steps: list[Mapping[str, Any]],
    proposer: Mapping[str, Any],
    proposal_handoff_steps: list[Mapping[str, Any]],
    coder: Mapping[str, Any],
    runner: Mapping[str, Any] | None,
    publish_s: float,
) -> dict[str, Any]:
    """Build a canonical trial timing summary enriched with agent-loop phases."""
    normalized_runner = build_runner_timing_summary(runner_summary)
    runner_total = _phase_total(normalized_runner, "runner")
    runner_span = runner or {}
    runner_start = _float(runner_span.get("start_s"))
    runner_end = _float(runner_span.get("end_s")) or runner_start
    proposer_start = _float(proposer.get("start_s"))
    proposer_end = _float(proposer.get("end_s"))
    coder_start = _float(coder.get("start_s"))
    coder_end = _float(coder.get("end_s")) or coder_start

    if runner_start:
        coder_implementation = max(0.0, runner_start - coder_start)
        coder_report = max(
            0.0,
            (coder_end - coder_start) - coder_implementation - runner_total,
        )
    else:
        coder_implementation = max(0.0, (coder_end - coder_start) - runner_total)
        coder_report = 0.0

    phases: dict[str, float] = {}
    details: dict[str, Any] = {}
    setup_start = min(
        (value for value in [_earliest_step_start(setup_steps), proposer_start] if value > 0.0),
        default=proposer_start,
    )
    _append_step_group(
        phases,
        details,
        "setup",
        setup_steps,
        total_seconds=max(0.0, proposer_start - setup_start) if proposer_start else 0.0,
    )
    _append_simple_phase(phases, details, "proposer", _duration(proposer))
    _append_step_group(
        phases,
        details,
        "proposal_handoff",
        proposal_handoff_steps,
        total_seconds=max(0.0, coder_start - proposer_end)
        if coder_start and proposer_end
        else 0.0,
    )
    _append_simple_phase(phases, details, "coder_implementation", coder_implementation)
    _append_runner(phases, details, normalized_runner)
    _append_simple_phase(phases, details, "coder_report", coder_report)
    _append_simple_phase(phases, details, "publish", publish_s)

    starts = [_step_start(step) for step in setup_steps + proposal_handoff_steps] + [
        _float(proposer.get("start_s")),
        coder_start,
        runner_start,
    ]
    ends = [
        _float(proposer.get("end_s")),
        coder_end,
        runner_end,
    ]
    positive_starts = [value for value in starts if value > 0.0]
    positive_ends = [value for value in ends if value > 0.0]
    first = min(positive_starts) if positive_starts else 0.0
    last = max(positive_ends) if positive_ends else first
    return {
        "schema_version": 2,
        "unit": "seconds",
        "total_seconds": round_seconds(max(0.0, last - first) + max(0.0, publish_s)),
        "phases": phases,
        "phase_details": details,
    }


def _append_simple_phase(
    phases: dict[str, float],
    details: dict[str, Any],
    name: str,
    value: float,
) -> None:
    rounded = round_seconds(value)
    phases[name] = rounded
    details[name] = {"total_seconds": rounded}


def _append_step_group(
    phases: dict[str, float],
    details: dict[str, Any],
    name: str,
    steps: list[Mapping[str, Any]],
    *,
    total_seconds: float | None = None,
) -> None:
    observed_total = sum(_float(step.get("duration_s")) for step in steps)
    total = round_seconds(observed_total if total_seconds is None else total_seconds)
    phases[name] = total
    payload: dict[str, Any] = {"total_seconds": total}
    step_phases = {
        _step_name(step): round_seconds(step.get("duration_s"))
        for step in steps
        if _step_name(step)
    }
    other = round_seconds(max(0.0, total - round_seconds(observed_total)))
    if other:
        step_phases["other"] = other
    if step_phases:
        payload["phases"] = step_phases
    details[name] = payload


def _append_runner(
    phases: dict[str, float],
    details: dict[str, Any],
    runner_summary: Mapping[str, Any],
) -> None:
    total = _phase_total(runner_summary, "runner")
    phases["runner"] = round_seconds(total)
    raw_details = runner_summary.get("phase_details")
    if isinstance(raw_details, Mapping) and isinstance(raw_details.get("runner"), Mapping):
        detail = dict(raw_details["runner"])
        normalized = {"total_seconds": round_seconds(detail.get("total_seconds", total))}
        runner_phases = detail.get("phases")
        if isinstance(runner_phases, Mapping):
            normalized["phases"] = {
                str(name): round_seconds(value) for name, value in runner_phases.items()
            }
        details["runner"] = normalized
        return
    details["runner"] = {"total_seconds": round_seconds(total)}


def _phase_total(summary: Mapping[str, Any], name: str) -> float:
    phases = summary.get("phases")
    if not isinstance(phases, Mapping):
        return 0.0
    return _float(phases.get(name))


def _duration(span: Mapping[str, Any]) -> float:
    return max(0.0, _float(span.get("end_s")) - _float(span.get("start_s")))


def _float(value: object) -> float:
    if value in (None, ""):
        return 0.0
    return float(value)  # type: ignore[arg-type]


def _step_name(step: Mapping[str, Any]) -> str:
    raw = str(step.get("name") or step.get("step") or "")
    return raw.strip().replace(" ", "_").replace("-", "_")


def _step_start(step: Mapping[str, Any]) -> float:
    if step.get("start_s") not in (None, ""):
        return _float(step.get("start_s"))
    return max(0.0, _float(step.get("time_s")) - _float(step.get("duration_s")))


def _earliest_step_start(steps: list[Mapping[str, Any]]) -> float:
    starts = [_step_start(step) for step in steps]
    positives = [value for value in starts if value > 0.0]
    return min(positives) if positives else 0.0


__all__ = [
    "build_runner_timing_summary",
    "enrich_agent_timing_summary",
    "normalize_timing_summary",
    "round_seconds",
]
