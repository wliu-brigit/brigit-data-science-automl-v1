# Handoff — continue here

The running note of where the last working session left off. **Read this first
when resuming**, then [`README.md`](README.md) for the docs lifecycle. Keep it to
*current state + next actions* (git history is the changelog, not this).

**Last updated:** 2026-06-04

## Where things stand

- **Active effort — steps 1–2 of 4 landed:**
  [`execution/snowflake-source-and-split-keys/`](execution/snowflake-source-and-split-keys/)
  — the full design for the real `SnowflakeSource` plus the data-layer
  contract work it surfaced. **`design.md` is the source of truth**;
  the status ledger + fresh-session protocol live in
  [`plans/`](execution/snowflake-source-and-split-keys/plans/README.md).
  **Step 1 (keys & naming) landed 2026-06-04**: `SPLITID` → `SPLIT_PCT`,
  `hash_key` → required `unique_key` + optional `split_group_key`, row
  fallback deleted, materialize-edge validation (unique_key duplicate-free;
  SPLIT_PCT present/integer/0–99; loud collision error on a source-provided
  split column). **Step 2 (dataset record & lifecycle) landed 2026-06-04**:
  dataset records relocated GCS → MLflow (`datasets/<id>/dataset.json` on
  the overview run; index/latest mirrors and GCS manifests deleted — the
  folder structure is the index, the experiment tag the pointer); recipe
  (config-derived identity) recorded on the record; `materialize()` attaches
  as pinned by default, warns with a field diff on recipe drift, re-derives
  only on `--refresh-data` (`--refresh-source` implies it); content hash now
  row-order-insensitive; eval records renamed for their nouns
  (`eval_dataset.json` / `augmentation.json`, `record_gcs_uri`); eval key
  normalization unified onto the data normalizer. Steps 3–4 (Snowflake,
  flexible splits) remain — one step per session, in order.
- **Known-stale, deferred to the tail-end notebook pass:**
  `example_homecredit` notebooks still reference `hash_key`/`SPLITID` in
  code cells and outputs (see plans README "Tail-end activities").

## Next actions

1. **Execute step 3** (Snowflake) following the protocol in
   [`plans/README.md`](execution/snowflake-source-and-split-keys/plans/README.md).
2. **wendao: two orphan GCS objects to wipe at your discretion** — the step-2
   session's first full-suite run (before the test was re-routed) minted
   `…/example_homecredit/example-homecredit/data/datasets/v1_da6dbdc5/`
   (`data.parquet` + `feature_registry.csv`) in the real bucket with no MLflow
   record (its tmp `mlruns` is gone). The code never deletes state (design §14
   ground rule); details in the step-2 deviations column.
3. For reference, the 2026-06-04 wipe that preceded step 2: **the entire
   `dry_run/example_homecredit` route** — both MLflow experiments (`overview`
   id 23 and `example-homecredit` id 24, each renamed to `…__trash-2026-06-04`
   then soft-deleted so the route names are free; hard delete waits on a
   platform-team `mlflow gc`), all GCS objects under
   `automl/dry_run/example_homecredit/` (52: dataset index + bytes, eval
   datasets, trial artifacts), and the local `experiments/dry_run` folder.
   Verified empty. Non-dry-run routes untouched.

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
