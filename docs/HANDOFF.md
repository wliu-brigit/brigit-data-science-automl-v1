# Handoff — continue here

Temporary hand-over note: where the last session left off and what's open, so a
new session can pick up. **Not a changelog** — keep only what's current and
relevant; detail lives in the project docs. Rewritten at each wrap, not appended
to.

**Last updated:** 2026-06-09 (neobank_ncm replication project built + QA-proven end-to-end → start here)

## How to pick up (per wendao)

Don't dive into code. (1) Read this plus the project docs below; (2) summarize
where things stand; (3) **recommend 2–3 options and let wendao pick.** The next
move is wendao's call, not a queue to drain.

## What this effort is

Branch `neobank_NCM_V3_replicate`, project `projects/neobank_ncm/`. Goal:
**faithfully replicate** the legacy Neobank NCM underwriting model v3 inside
this harness — same data, same techniques, same metrics — then let the AutoML
loop explore beyond it. Legacy home (read-only):
`brigit/data-science/models/underwriting/neobank/new_user/v3.0/` — its
`CLAUDE.md` is the legacy build plan; `notebooks/neobank_ncm_model_v3_final.ipynb`
is the canonical reference. Mimic legacy first; "what a new project would do
differently" is explicitly out of scope.

The whole project folder is **uncommitted** (user wants to commit at a finish
point). Everything below is in the working tree.

## Where things stand (all verified, not just written)

- **Recipe** (`config.py`): target `went_dpd45`; SnowflakeSource wrapping the
  three legacy sandbox snapshots (spine ⋈ risk_features ⋈
  synthetic_scores_final, read-only in `sandbox_hyong`; our copy materializes
  as `neobank_ncm_v3_replicate_base` under `SNOWFLAKE_SCHEMA=sandbox_wliu`).
  Splits: `train` (Jan–Oct 2025, known+unknown), `train_known` (eval-only
  view), `test` (Nov–Dec 2025 known-only, the loop's leaderboard),
  `oot` (Jan–Feb 2026 known-only, touched once post-AutoML). experiment_id
  `neobank_ncm_v3_replicate`.
- **Candidate set audited against legacy**: the locked 163 features all
  resolve; the experiment notebook's "Pass 0" exclusions are enforced via
  `exclude_cols`; all 11 derived features (incl. 3 that died in their
  selection) are materialized in `data/queries/base_table.sql`.
- **WoE** (`model/preprocessing.py`): legacy fit_woe/apply_woe ported
  (log(dist_good/dist_bad), 0.5 smoothing, min_obs 30, OTHER=CHIME), fit on
  labeled rows only; `PrefitBankInstitutionWOEEncoder` mounts the
  known-only-fit mapping into the model's ColumnTransformer (required by the
  harness contract). Legacy fitted mapping bundled at
  `model/bankinstitutionwoe.json` for VPN-day diffing.
- **Baseline trial** (`model/baseline.py`, the project MODEL_CLASS): full
  legacy Phase-4 recipe driven by `data/legacy/experiment_decisions.json` —
  reject-inference dual records (fuzzy augmentation) at 80/20,
  random_state=42, locked params/constraints/n_estimators.
- **Tests**: 14 passing (`uv run pytest projects/neobank_ncm/tests/`) —
  WoE semantics, synthetic-fixture offline e2e (pipeline → splits → runner
  pre-fit contract WITH session → fit → known-only AUC).
- **QA run through the real stack** (local MLflow + GCS, CSV stand-in for
  Snowflake): `scripts/qa_local_run.py` → materialize + run_trial FINISHED;
  `scripts/evaluate_oot.py` works for any named split. Namespace
  `qa/neobank-csv-dryrun-20260609` (sweepable). Fixture numbers:
  train_known .799 / test .746 / oot .719.

## Learnings that will bite a new session

- Required transformers must appear as a **named entry inside
  `model.preprocessor` (ColumnTransformer)**; ColumnTransformer refits
  entries on the training frame, hence the prefit-WoE pattern (fit on known
  rows must not see synthetic dual-record labels).
- Fitted models must set `feature_cols` == the registry's `model=True` set
  exactly (`automl/runner/contract.py`).
- The runner's automatic train-split eval **silently skips** on this project
  (train carries NULL targets by design) — that's why `train_known` exists,
  evaluated on demand via `scripts/evaluate_oot.py --split train_known`.
  Systemic fix parked at `docs/to-do/runner-best-effort-visibility.md`.
- `exclude_cols` strips feature/model flags but keeps the column in the frame.
- Synthetic labels train; they never enter a metric. Every reported metric is
  known-only — that's the legacy's own locked decision.

## Open items (options for wendao, roughly in order)

1. **Exact-sampling deviation**: legacy downsampled unknowns 600K→200K via
   Snowflake `ORDER BY HASH(entity_id)` before the ratio draw; baseline
   currently samples the ratio target from all unknowns (counts match, exact
   rows differ). Proposed fix (not yet approved): add a hash-rank column in
   base_table.sql + replay the draw in baseline. Small change, kills the only
   known training deviation.
2. **Commit the checkpoint** (project folder + docs entry are untracked).
3. **Kick off the actual AutoML loop** — never attempted yet; only the manual
   baseline trial has run. Could dry-run the agent loop against the CSV
   fixture under a qa/ namespace before VPN day.
4. **VPN day (last step, user will say when)**: materialize real snapshot →
   DESCRIBE checks (96 plaid/netflow columns exist; derived-name collisions;
   row counts vs legacy: 282,642 known train / 70,662 sampled unknowns) →
   baseline trial → test AUC vs legacy 0.7016 → `evaluate_oot.py` → vs legacy
   OOT AUC (`data/legacy/preprocessor_meta.json → performance`) → diff fitted
   WoE vs bundled mapping.
5. Parked small: rename `evaluate_oot.py` → `evaluate_split.py` (it already
   takes `--split`).

## Constraints to respect

- **No VPN/Snowflake until the user calls it** — keep everything runnable
  offline (CSV fixture path) or against local MLflow/GCS (.env works).
- Legacy folder is read-only. QA/dev MLflow runs go under `qa/...` namespaces.
