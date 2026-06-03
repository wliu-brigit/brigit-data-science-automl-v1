# Handoff — continue here

The running note of where the last working session left off. **Read this first
when resuming**, then [`README.md`](README.md) for the docs lifecycle. Keep it to
*current state + next actions* (git history is the changelog, not this).

**Last updated:** 2026-06-02

## Where things stand

- Repo is `main` at the squashed initial commit; the pre-squash
  `refactor/four-layer` world this note used to describe is history.
- **Working tree carries an uncommitted change set** from the 2026-06-02
  hang/observability session (root cause + decisions in
  [`to-do/agent-observability-follow-ups.md`](to-do/agent-observability-follow-ups.md)):
  - seam-wide `MLFLOW_HTTP_REQUEST_MAX_RETRIES=1` (`automl/mlflow/client.py`),
    replacing the runner-only cap;
  - list-first `client.download_artifact()` helper; all 8 raw
    `download_artifacts` call sites routed through it (missing artifact →
    fast `None`, no 500-retry storm on pre-3.12 servers);
  - per-trial `agent/proposer/message.md` + `agent/coder/message.md` (full,
    untruncated closing messages, extracted from transcripts at publish);
  - unified timing: CLI verbs append `step` events to the session timeline
    (`automl/agent/timeline/steps.py`), session report gains
    `step_durations_s`/`steps`/`publish_s`, coder report gains
    `runner_phases_s` from the runner's `timing.*` metrics.
- Full local suite green:
  `uv run pytest tests/unit tests/integration tests/contracts` → **478 passed**.
- Verified against production: `automl experiment proposer-context` on the
  `example_homecredit` dry-run route went from hanging past the 120s guard to
  **~28s** end-to-end.
- Notebook run-state in the `example_homecredit` notebooks remains intentional,
  not yet committed.

## Next actions

1. **Commit the session change set** (single change: fix + tests + docs notes).

## On hold — waiting, not next fixes

- **End-to-end validation of the change set** — waiting on a healthy Claude-API
  day: re-run `2_run_agent_automl.ipynb` and walk the checklist in
  [`to-do/agent-observability-follow-ups.md`](to-do/agent-observability-follow-ups.md)
  §0. (The 2026-06-02 attempt was swamped by Claude API timeouts — manager
  requests hung ~16 min/attempt; not a harness issue. The new
  `unattributed_s` summary field surfaces exactly this.)
- **MLflow server upgrade** — waiting on the platform team:
  [`to-do/upgrade-mlflow-server.md`](to-do/upgrade-mlflow-server.md).
- **Parked:** untangle the MLflow-seam import cycles — extract shared
  value-types into a low contracts layer so `mlflow ↔ eval/project/trial`
  stops cycling.
- **Forward work:** `docs/to-do/agent-orchestration/` (also picks up the
  manager end-of-run summary placement).

## Gotchas (don't relitigate)

- Repo-local plugins don't flag-free auto-load in Claude Code (v2.1.159). Load via
  **`--plugin-dir agent-skills`** (the loop does this) or symlink `agent-skills` into
  `~/.claude/skills/`. The `agents` custom-path manifest field isn't honored — agents
  must live at the plugin's default `agents/`.
- **The Jupyter kernel must point at THIS clone's `.venv`.** VSCode can launch a
  sibling clone's venv even when you pick "Brigit AutoML (.venv)" — confirm with
  `import sys; print(sys.executable)` in cell 1 (`from automl import experiment`
  failing means the kernel is bound to the wrong clone).
