# Handoff — continue here

The running note of where the last working session left off. **Read this first
when resuming**, then [`README.md`](README.md) for the docs lifecycle. Keep it to
*current state + next actions* (git history is the changelog, not this).

**Last updated:** 2026-06-04

## Where things stand

- **Active effort — step 1 of 4 landed:**
  [`execution/snowflake-source-and-split-keys/`](execution/snowflake-source-and-split-keys/)
  — the full design for the real `SnowflakeSource` plus the data-layer
  contract work it surfaced. **`design.md` is the source of truth**;
  the status ledger + fresh-session protocol live in
  [`plans/`](execution/snowflake-source-and-split-keys/plans/README.md).
  **Step 1 (keys & naming) landed 2026-06-04**: `SPLITID` → `SPLIT_PCT`,
  `hash_key` → required `unique_key` + optional `split_group_key`, row
  fallback deleted, materialize-edge validation (unique_key duplicate-free;
  SPLIT_PCT present/integer/0–99; loud collision error on a source-provided
  split column). Steps 2–4 (dataset record & lifecycle, Snowflake, flexible
  splits) remain — one step per session, in order.
- **Known-stale, deferred to the tail-end notebook pass:**
  `example_homecredit` notebooks still reference `hash_key`/`SPLITID` in
  code cells and outputs (see plans README "Tail-end activities").

## Next actions

1. **Execute step 2** (dataset record & lifecycle) following the protocol in
   [`plans/README.md`](execution/snowflake-source-and-split-keys/plans/README.md).
2. Before/with step 2: **wendao manually wipes old MLflow/GCS state** for a
   clean slate — the implementation must never delete or migrate old state
   itself (design §14 ground rule).

## On hold — waiting, not next fixes

- **MLflow server upgrade** — waiting on the platform team:
  [`to-do/upgrade-mlflow-server.md`](to-do/upgrade-mlflow-server.md).
- **Agent observability follow-ups** — live-run checklist in
  [`to-do/agent-observability-follow-ups.md`](to-do/agent-observability-follow-ups.md) §0.
- **Forward work:** [`to-do/agent-orchestration/`](to-do/agent-orchestration/).
- **Parked:** untangle the MLflow-seam import cycles (extract shared
  value-types into a low contracts layer).

## Gotchas (don't relitigate)

- Repo-local plugins don't flag-free auto-load in Claude Code (v2.1.159). Load via
  **`--plugin-dir agent-skills`** (the loop does this) or symlink `agent-skills` into
  `~/.claude/skills/`. The `agents` custom-path manifest field isn't honored — agents
  must live at the plugin's default `agents/`.
- **The Jupyter kernel must point at THIS clone's `.venv`.** VSCode can launch a
  sibling clone's venv even when you pick "Brigit AutoML (.venv)" — confirm with
  `import sys; print(sys.executable)` in cell 1 (`from automl import experiment`
  failing means the kernel is bound to the wrong clone).
