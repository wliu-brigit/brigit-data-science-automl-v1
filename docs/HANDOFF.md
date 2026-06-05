# Handoff — continue here

The running note of where the last working session left off. **Read this first
when resuming**, then [`README.md`](README.md) for the docs lifecycle. Keep it to
*current state + next actions* (git history is the changelog, not this).

**Last updated:** 2026-06-04

## Where things stand

- **Active effort — all 4 code steps landed; tail-end cleanup is partly done:**
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
  probe, Snowpark dropped); **step 4** hard-cut `Splits` from bucket ranges
  to `Where(...)` predicates (serializable JSON AST; trial contracts and
  eval split-view identities carry the AST; `SPLIT_PCT` is an ordinary
  column; `to_pyarrow` push-down compiled but deliberately not wired —
  layout + reader ship together later). Current non-live suite:
  **587 passed** (`uv run pytest tests/unit tests/contracts tests/integration`,
  2026-06-04 local check).
- **Tail-end items already completed after the ledger row was written:**
  live Snowflake e2e passed; `example_homecredit` notebooks were updated to
  current vocabulary and stripped of stale cached outputs; the renamed
  notebook workflow (`3.1`/`3.2`/`3.3`) executed end to end; the retired
  split-vocabulary ratchet was added; `list_dataset_records` now propagates
  transport failures and treats missing `datasets/` as a clean empty list.
- **Still not done:** first real `fraud_anomaly_detection` materialize. The
  project config intentionally still has TBD placeholders for target, base
  table, source table, unique key, and experiment id, so it is not ready for
  validation/materialization until those are filled in during the VPN session.

## Next actions

1. **Finish the remaining tail-end work**: fill the
   `fraud_anomaly_detection` TBDs with the designated warehouse values, run
   the first real materialize on VPN, then move the effort
   `execution/ → archive/`. Details live in the plans README "Tail-end
   activities". (**Live Snowflake e2e: done 2026-06-04** — passed in 91s
   against a sampled `fct_loans` dev table; details in the tail-end list.
   Snowflake is confirmed reachable with `.env` configured — the one gotcha
   was the `SNOWFLAKE_ACCOUNT` identifier form, now documented in
   `.env.example`.)
2. Notebook QA is complete for this pass: `AUTOML_E2E_NOTEBOOKS=1
   uv run pytest tests/e2e/test_homecredit_notebooks.py -q` passed on
   2026-06-04 in 8:48.

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
