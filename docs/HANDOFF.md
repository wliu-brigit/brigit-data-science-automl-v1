# Handoff — continue here

The running note of where the last working session left off. **Read this first
when resuming**, then [`README.md`](README.md) for the docs lifecycle. Keep it to
*current state + next actions* (git history is the changelog, not this).

**Last updated:** 2026-06-05

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
  layout + reader ship together later). Suite 573-green.
- **All live/VPN-dependent verification is deliberately batched** into one
  tail-end session now that step 4 has landed (wendao 2026-06-04: getting on
  the VPN is slow — run the live items as one isolated batch). The list
  lives in the plans README "Tail-end activities".
- **Notebooks verified live 2026-06-04:** all 8 `example_homecredit`
  notebooks updated to the current vocabulary and executed end-to-end
  against live services (33m55s incl. one dry-run agent-loop iteration).

- **Fraud pilot ran end-to-end (2026-06-05)** — the `fraud_anomaly_detection`
  project is set up, materialized (1.88M-row 99/1 pull over a pre-built
  table), and has 6 dry-run trials logged (IF / PCA / GMM; first full metric
  table on trial 6). The pilot shook out seven library issues — all fixed
  same-day on branch `feature/fraud-pilot-library-feedback` (sampler,
  Decimal seam, predicate case, error chains, scaffold front door, AP
  metric); see `archive/fraud-pilot-library-feedback/` for symptom→fix
  records. Spin-offs: `to-do/leaderboard-dataset-pinning.md`,
  `to-do/loop-observability.md`, `to-do/split-pct-lowercase.md`. Fraud
  project files are deliberately uncommitted (wendao: revisit later; next
  steps accumulate in the session task list — fair 3-way comparison run,
  per_trial_seconds bump, full run, projected precision, fraud-dive).

## Next actions

1. **Archive the Snowflake effort** `execution/ → archive/`: its last
   tail-end item — the first real `fraud_anomaly_detection` materialize —
   completed 2026-06-05 (the expected duplicate-unique-key conversation
   resolved itself: the upstream table dedupes by advance_id). Everything
   else on the tail-end list was done 2026-06-04. Follow the effort's own
   plans-README protocol for the move.
2. **Merge `feature/fraud-pilot-library-feedback`** (7 commits, suite
   592-green) once wendao has reviewed.

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
