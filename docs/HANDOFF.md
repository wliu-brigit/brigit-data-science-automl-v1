# Handoff — continue here

The running note of where the last working session left off. **Read this first
when resuming**, then [`README.md`](README.md) for the docs lifecycle. Keep it to
*current state + next actions* (git history is the changelog, not this).

**Last updated:** 2026-06-04

## Where things stand

- **Active effort — APPROVED, ready to implement:**
  [`execution/snowflake-source-and-split-keys/`](execution/snowflake-source-and-split-keys/)
  — the full design for the real `SnowflakeSource` plus the data-layer
  contract work it surfaced: `unique_key`/`split_group_key`, `SPLIT_PCT`
  rename, row-fallback removal, attach-as-pinned materialize with explicit
  `--refresh-data`/`--refresh-source`, dataset records relocating GCS →
  MLflow (`dataset.json`), order-insensitive identity, connector-python,
  and flexible `Where(...)` predicate splits. **`design.md` is the source
  of truth — read it before touching code.** 4-step plan in §14; open
  items in §15.
- **Working tree (uncommitted):** the design dossier
  (`docs/execution/snowflake-source-and-split-keys/`); the scaffolded
  `projects/fraud_anomaly_detection/` (all `<TBD` placeholders — the
  consumer waiting on this effort); small pre-existing edits to
  `example_homecredit` config + notebooks 1–2.

## Next actions

1. **Commit the design dossier** (docs-only change).
2. **Start step 1** of [`design.md`](execution/snowflake-source-and-split-keys/design.md)
   §14 (keys & naming cleanup — no Snowflake yet).
3. Before/with step 2: **wendao manually wipes old MLflow/GCS state** for a
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
