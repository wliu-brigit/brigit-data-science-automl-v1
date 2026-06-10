# Handoff — continue here

Temporary hand-over note: where the last session left off and what's open, so a
new session can pick up. **Not a changelog** — keep only what's current and
relevant; detail lives in the project docs. Rewritten at each wrap, not appended
to.

**Last updated:** 2026-06-09 (legacy replication complete offline — audited,
loop-proven, downstream analyses ported; commit + VPN day are what's left)

## How to pick up (per wendao)

Don't dive into code. (1) Read this plus the project docs below; (2) summarize
where things stand; (3) **recommend 2–3 options and let wendao pick.** The next
move is wendao's call, not a queue to drain.

## What this effort is

Branch `neobank_NCM_V3_replicate`, project `projects/neobank_ncm/`. Goal:
**faithfully replicate** the legacy Neobank NCM underwriting model v3 inside
this harness — same data, same techniques, same metrics — then let the AutoML
loop explore beyond it. Legacy home (read-only):
`brigit/data-science/models/underwriting/neobank/new_user/v3.0/`; its
`notebooks/neobank_ncm_model_v3_final.ipynb` is the canonical reference.
Everything except running against the real warehouse is **done and
offline-verified**; the working tree holds the uncommitted checkpoint
(touches `projects/neobank_ncm/` and `docs/` only — zero core changes).

## Where things stand (all verified, not just written)

- **Training parity audited line-by-line vs legacy** (notebooks + artifacts):
  locked params/constraints/features byte-identical (`data/legacy/*.json` ==
  legacy artifacts), test window confirmed Nov–Dec 2025, derived-feature SQL
  exact. Two real fixes landed: the preprocessor (medians/OHE) now fits on
  **known rows only** as legacy did, and the legacy 600K→200K unknown
  downsample is **replayed exactly** via a materialized
  `unknown_train_hash_rank` (Snowflake HASH is deterministic) + rank-sorted
  random_state=42 draw. Baseline checkpoint number = **0.7002**
  (`test_auc_constrained`); 0.7016 is the unconstrained stretch reference.
- **Source toggle**: `NEOBANK_NCM_CSV=/path/to.csv` swaps the recipe to a
  LocalCSVSource at config load; unset = real Snowflake. The whole harness —
  QA script, loop, tests — runs offline through it.
- **Agent loop dry-ran end-to-end** (first time ever) under
  `qa/neobank-loop-dryrun-20260609`: proposed/coded/ran an `lgbm_challenger`
  (fixture test AUC 0.7346 vs baseline 0.7457). One coder failure
  (`__file__` path math) became a Trial-code rule in PROJECT_INSTRUCTIONS.
- **Downstream analyses ported** (wendao pulled the phase forward):
  `analysis/` (data/scoring/policy/impact) replicates the legacy
  financial-impact + new-links-eval computations exactly;
  `scripts/evaluate_new_links_daily.py` logs the QA/eval run to MLflow;
  `notebooks/financial_impact_analysis.ipynb` mirrors the legacy cell story
  (parity mode = legacy artifacts; trial mode = any logged model). The RI
  model is never re-run — its outputs are frozen snapshots. Adversarial
  parity review + code review done; findings fixed.
- **Tests: 84 green** — 52 contracts + 32 project (WoE, replay determinism,
  split isolation incl. oot-unknowns-in-no-split, analysis formulas
  hand-computed, script e2e with stub model + file MLflow).

## Learnings that will bite a new session

- Required transformers must be named entries inside `model.preprocessor`;
  the prefit-WoE pattern exists because ColumnTransformer refits on the
  training frame (known-only fit must not see dual-record labels).
- The runner's automatic train-split eval silently skips here (NULL targets
  by design) — `train_known` + `scripts/evaluate_split.py` cover it.
- Trial code must resolve project assets via the package
  (`projects.neobank_ncm.__file__`), never `__file__` parents math — trials
  execute from deep trial dirs.
- The legacy financial notebook's D1 "null-out" was **dead code** (case bug);
  the port replicates the executed behavior (raw D1 score) — see the comment
  in `analysis/policy.py`.
- `import mlflow` is allowed only inside `automl/mlflow` (contract test);
  project code goes through the seam (`automl.mlflow.trial.artifacts
  .load_model`, `bound_for` + `raw()`).
- Synthetic labels train; they never enter a reported metric. The analysis
  layer's `effective_bad` is policy-analysis-only, same legacy rule.

## Open items (options for wendao, roughly in order)

1. **Commit the checkpoint** (working tree: project folder + this file +
   `docs/to-do/neobank-ncm-vpn-day.md`).
2. **VPN day** — the full runbook with expected numbers is
   [`docs/to-do/neobank-ncm-vpn-day.md`](to-do/neobank-ncm-vpn-day.md):
   flip the toggle, DESCRIBE checks, materialize, baseline vs 0.7002, WoE
   diff, real loop, then §7 downstream parity (D2 AUC ≈ 0.6935, RI corr
   0.9999) and the winner's financial analysis.
3. **Sweep QA namespaces** when done inspecting:
   `automl project delete --scope qa` (covers the csv-dryrun + loop-dryrun
   namespaces and the gitignored local state).
4. Parked design call: core preflight live-probes Snowflake even when the
   experiment has a pinned dataset (the toggle sidesteps it; a fix would
   live in `automl/project/checks.py` + the validate recipe).
5. Next phase after the parity run: the decision-memo write-up the analysis
   feeds (explicitly out of scope so far).

## Constraints to respect

- **No VPN/Snowflake until the user calls it** — everything must stay
  runnable offline (the toggle + parquet escape hatches; .env works for
  MLflow/GCS).
- Legacy folder is read-only. QA/dev MLflow runs go under `qa/...`
  namespaces. The `oot` split is touched once, post-AutoML, via
  `scripts/evaluate_split.py`.
