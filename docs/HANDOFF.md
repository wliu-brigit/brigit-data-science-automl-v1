# Handoff — continue here

The running note of where the last working session left off. **Read this first
when resuming**, then [`README.md`](README.md) for the docs lifecycle. Keep it to
*current state + next actions* (git history is the changelog, not this).

**Last updated:** 2026-06-04

## Where things stand

- **Active effort — steps 1–3 of 4 landed, step 4 is next:**
  [`execution/snowflake-source-and-split-keys/`](execution/snowflake-source-and-split-keys/)
  — the real `SnowflakeSource` plus the data-layer contract work it
  surfaced. **`design.md` is the source of truth**; the status ledger
  (per-step commits, deviations, review outcomes) + fresh-session protocol
  live in [`plans/`](execution/snowflake-source-and-split-keys/plans/README.md).
  In one line each: **step 1** renamed the key/split vocabulary
  (`unique_key`/`split_group_key`, `SPLIT_PCT`) and added materialize-edge
  validation; **step 2** moved dataset records GCS → MLflow and made
  `materialize()` attach-as-pinned with recipe-drift warnings and explicit
  `--refresh-data`/`--refresh-source`; **step 3** made `SnowflakeSource`
  real (SELECT-only `base_table_sql`, harness-owned DDL with `SPLIT_PCT`
  injection, split-invariant check, bucket-sample dry-run, live validate
  probe, Snowpark dropped) — suite 554-green after a five-agent post-landing
  review (outcome + fixes in the ledger row).
- **All live/VPN-dependent verification is deliberately batched** into one
  tail-end session after step 4 lands (wendao 2026-06-04: getting on the
  VPN is slow — finish everything first, then run the live items as one
  isolated batch). The list lives in the plans README "Tail-end activities".
- **Known-stale until that tail-end pass:** `example_homecredit` notebooks
  (old `hash_key`/`SPLITID`/`base_data_sql` in cells and cached outputs;
  notebook 2's `Splits(train=[(0, 80)]…)` breaks after step 4's hard cut).

## Next actions

1. **Execute step 4** (flexible splits — `Where` predicates replace bucket
   ranges, hard cut) following the protocol in
   [`plans/README.md`](execution/snowflake-source-and-split-keys/plans/README.md).
   Read the step-4 row's heads-up notes first — they list range-API
   consumers created after the plan was written.
2. **Then the batched tail-end session** (needs wendao + VPN): live
   Snowflake e2e, live notebook verification, first real
   `fraud_anomaly_detection` materialize, archive the effort. Details in
   the plans README.

(State wipes are always a human call, never the code's — design §14 ground
rule. The 2026-06-04 `example_homecredit` wipes are recorded in the step-2
ledger row and git history; both stores verified clean on that route.)

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
