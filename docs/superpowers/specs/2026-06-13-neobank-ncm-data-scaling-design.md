# Data-scaling study — neobank_ncm (legacy XGBoost baseline)

**Date:** 2026-06-13 · **Branch:** `neobank_ncm_v3_replicate`

## Goal

Characterize how the **legacy replication XGBoost** (`model/baseline.py`,
`NeobankNCMReplicationModel`) scales with training-data size along two
independent axes — synthetic/unknown reject-inference data, and known
(ground-truth) data — measured on the **business decision metrics**, not just
AUC. Output: a scaling curve per axis to justify pulling in more/earlier data
vs. shrinking the set for faster training, and to see how boosting scales
(vs. the data-hungry reputation of nets).

This is **not** the decision-focused / boundary / stack thread — that is a
separate effort. Here the model is the faithful legacy XGBoost, run exactly as
run #1, with **only the data-size knob changed**.

## Fixed baseline (run #1)

`model/baseline.py` as-is: locked 162-feature set, WoE(bank), monotone
constraints, dual-record soft labels at the locked 80/20 known/unknown ratio,
locked XGBoost params, `random_state=42`. Run #1 is the 20%-synthetic /
full-known point and already exists — we do **not** re-run it.

## The two knobs (both already in `baseline.py`)

- **Synthetic share** — the dual-record expansion (lines ~150–184). The
  unknown's share of the training mix; `ratio` (known-fraction) knob computes
  `n_syn_target = len(known) * (1-ratio)/ratio`. The `else` branch at ~182–184
  already handles **0% synthetic** (known-only).
- **Known size** — `known = df_train[df_train[target].notna()]` (line ~138).
  Random downsample with `random_state=42`, tiny-sample-safe:
  `known.sample(min(n, len(known)), random_state=42)`.

## Run list (8 new runs)

Slug convention: `data_scaling_known{SIZE}_synth{PCT}` —
`SIZE ∈ {050k,100k,150k,200k,250k,full}`, `PCT ∈ {00,10,20,30}` = unknown's
share of the training mix. Every slug encodes both dimensions; the shared
corner runs once.

**Study A — synthetic axis** (known = full ≈ 282,642):

| slug | known | synth share | synth rows ≈ |
|---|---|---|---|
| `data_scaling_knownfull_synth00` | full | 0% | 0 |
| `data_scaling_knownfull_synth10` | full | 10% | ~31.4K |
| `data_scaling_knownfull_synth30` | full | 30% | ~121K |

(20% = run #1, anchors the curve; not re-run.)

**Study B — known axis** (synthetic = 0%):

| slug | known | synth share |
|---|---|---|
| `data_scaling_known050k_synth00` | 50K | 0% |
| `data_scaling_known100k_synth00` | 100K | 0% |
| `data_scaling_known150k_synth00` | 150K | 0% |
| `data_scaling_known200k_synth00` | 200K | 0% |
| `data_scaling_known250k_synth00` | 250K | 0% |
| `data_scaling_knownfull_synth00` | full | 0% | ← shared with Study A |

## Execution flow (standard, per run)

1. Trial dir under `experiments/neobank_ncm/neobank_ncm_v3_replicate/<slug>/`
   with `metadata.json` (slug + hypothesis), `model.py` (baseline recipe with
   the two knobs baked in — logged code reflects exactly what ran), `run.py`.
2. `uv run python <dir>/run.py` → `runner.run_trial` (fit + test AUC + serving
   validation), logs to experiment `neobank_ncm_v3_replicate` under the slug.
3. Post-eval: `scripts/score_trial_financials.py --eval-dataset-id <id>
   --model-run-id <run_id>` records the OOT new-links decision report.
4. Aggregate: add the runs to `scripts/consolidated_results.py` and render the
   table (test AUC, day2 AUC, scenario-2 UW ΔAR, swap-in BR, latency).

## Metrics reported per point

Scenario-2 UW **ΔAR**, **swap-in BR**, **day2 AUC**, **test AUC** — same
columns as the consolidated table — one row per data-size point, plus a short
writeup of the scaling shape per axis.

## Operational

- **Off-VPN** (GCS flaps on VPN); data already materialized, no Snowflake.
- Single seed per config (no repeats).
- Smoke-test the parametrized `model.py` in-process via
  `scripts/_smoke_model.py` before the batch; run one trial end-to-end to
  validate the full flow before batch-running the rest.
