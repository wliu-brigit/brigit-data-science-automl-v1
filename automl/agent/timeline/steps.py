"""CLI step timing events for the agent timeline.

The agent loop drives the harness through ``automl`` CLI verbs running in
their own processes (``experiment proposer-context``, ``data materialize``,
lock verbs, ...). Those steps used to be invisible in the published timing —
a multi-minute stall in ``proposer-context`` showed up nowhere. Recording one
"step" event per CLI invocation into the same route timeline the Claude hooks
append to lets the session summary account for the whole loop, not just the
proposer/coder agent spans.

Recording is env-gated: outside an agent session (no ``AUTOML_SESSION_ID``)
this is a no-op, so interactive human CLI use stays untouched.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any


def record_cli_step(
    step: str,
    *,
    duration_s: float,
    exit_code: int,
) -> None:
    """Append one CLI step event to the active agent-session timeline.

    ``step`` is the resolved ``"<noun> <verb>"`` (e.g. ``"experiment
    proposer-context"``). Best-effort by design: timing must never fail (or
    slow) the verb it measures, so resolution problems simply skip the record.
    """
    try:
        _record_cli_step(step, duration_s=duration_s, exit_code=exit_code)
    except Exception:
        return


def _record_cli_step(step: str, *, duration_s: float, exit_code: int) -> None:
    session_id = os.environ.get("AUTOML_SESSION_ID") or os.environ.get("CLAUDE_SESSION_ID")
    project_root = os.environ.get("AUTOML_PROJECT_ROOT")
    project_name = os.environ.get("AUTOML_PROJECT")
    if not session_id or not project_root or not project_name or not step:
        return
    from automl.agent.timeline.ingest import _append_event
    from automl.mlflow import routing as mlflow_routing

    route = mlflow_routing.experiment_route_for(
        project_name=project_name,
        experiment_id=os.environ.get("AUTOML_EXPERIMENT_ID") or None,
        namespace=os.environ.get("AUTOML_NAMESPACE", ""),
        dry_run=os.environ.get("AUTOML_INHERIT_DRY_RUN", "") == "1",
    )
    now = time.time()
    event: dict[str, Any] = {
        "schema_version": 1,
        "source": "automl_cli",
        "event": "step",
        "step": step,
        "route": route,
        "session_id": str(session_id),
        "time_s": now,
        "start_s": now - max(0.0, duration_s),
        "duration_s": max(0.0, duration_s),
        "duration_unit": "seconds",
        "exit_code": int(exit_code),
    }
    _append_event(Path(project_root), route, event)


__all__ = ["record_cli_step"]
