# Handoff — continue here

Temporary hand-over note: where the last session left off and what's open, so a
new session can pick up cold. **Not a changelog** — keep only what's current and
relevant. Rewritten at each wrap, not appended to.

> **These docs are best-effort documentation. The code is the source of truth
> for current behavior.**

**Last updated:** 2026-06-13 — branch `neobank_ncm_v3_replicate`. This session ran
a **breadth-first model search** read on the business metric (scenario-2 UW ΔAR +
swap-in BR), not AUC. Arc: post-hoc XGB+MLP blend → trained stack → **decision-focused
training (the winner)** → controls + breadth (ensemble diversity, representation
transfer). Net finding: **two real levers — better preprocessing and decision-focused
boundary weighting — lifted ΔAR from +4.65% (prod-replica XGB) to +6.45%, while
ensembling and neural-representation transfer did NOT help.**

## TL;DR — where we are

1. **New best model: `boundary XGB`** (decision-focused, single XGBoost) — ΔAR
   **+6.45%**, swap-in BR **0.2018**, test AUC 0.696, single-row latency **446 ms**,
   no torch → clean serving validation + cheap deploy. `boundary LightGBM` ties it
   (+6.38%, lowest swap-in 0.2012) — the lever reproduces across two GBM libraries.
2. **AUC ↔ business divergence is now extreme and ordered:** the top-3 ΔAR models
   have the **lowest** day2 AUC (0.660–0.664). Selecting on AUC picks the *worst*
   business model. Decision metrics must drive selection here, never AUC.
3. **Two levers, both real, ~equal size (decomposed via a matched control):**
   - **Preprocessing / pool** (clean 168 + per-feature missingness flags + scaling,
     vs the prod-replica recipe): +4.65% → **+5.60%** (+0.95pp).
   - **Decision-focused boundary weighting** (upweight the v3a-boundary cohort):
     +5.60% → **+6.45%** (+0.85pp), and lowers swap-in BR. **Positive across all 4
     match-bad-rate scenarios × both UW/CLE tracks** (not a scenario-2 artifact).
4. **Negative results (don't re-chase):** ensembling/stacking does NOT add once
   preprocessing is fixed (single XGB ≈ stack; boundary stack +6.20% < boundary XGB
   +6.45%); MLP-embedding→XGB representation transfer is marginal (+5.72%, ~noise
   over the +5.60% baseline). Consistent with the ~0.70 AUC **signal ceiling**:
   little *new* information is extractable from the 168 features.
5. **Latency is preprocessing-bound (~440–475 ms/row), model-agnostic** — confirmed
   across every measured model. The real latency lever is the transform, not the model.

## Consolidated results (scenario 2, UW track; sorted by ΔAR)

`uv run python projects/neobank_ncm/scripts/consolidated_results.py`
(→ `.cache/automl/fin/consolidated_results.parquet`)

| model | family | test AUC | day2 AUC | ΔAR % | swap-in BR | latency ms |
|---|---|---|---|---|---|---|
| **boundary XGB** | tree+decision | 0.696 | 0.664 | **+6.45** | 0.2018 | 446 |
| boundary LightGBM | tree+decision | 0.693 | 0.660 | +6.38 | **0.2012** | 449 |
| boundary stack | ensemble+decision | 0.695 | 0.662 | +6.20 | 0.2034 | n/a (torch SIGSEGV) |
| mlp_embed_xgb (repr-transfer) | repr-transfer | n/a* | 0.656 | +5.72 | 0.2045 | n/a |
| baseline XGB (good prep, no weighting) | tree | 0.699 | 0.670 | +5.60 | 0.2044 | 435 |
| t15 XGB+MLP stack | ensemble | 0.698 | 0.669 | +5.53 | 0.2037 | n/a (torch SIGSEGV) |
| t3 XGBoost (prod replica) | tree | 0.701 | 0.673 | +4.65 | 0.2108 | 474 |
| t1 XGBoost | tree | 0.700 | 0.673 | +4.29 | 0.2125 | n/a |
| t13 compact MLP | neural | 0.684 | 0.648 | +4.23 | 0.2090 | 466 |
| t12 spline GAM | additive | 0.663 | 0.643 | +1.46 | 0.2177 | 437 |
| t11 WoE scorecard | linear | 0.656 | 0.632 | +0.37 | 0.2214 | 476 |

*mlp_embed_xgb trial was marked FAILED on a GCS data-read flap during serving
validation (hash mismatch — known transient, not a model bug); the model was logged
and decision-eval'd off local frames. test AUC not captured.

Run IDs: boundary XGB `a14bca0a287d49f2b8934a400e2d547c` (trial 16) · boundary
LightGBM `da2065072b104c15893347d4976feb61` · boundary stack
`41eb1d6fc7a04305ad505ab06fba1f19` · baseline XGB `c9ba2f1368884974a0beca26e5601aed`
· mlp_embed `528699abdcfb4a12b11573f478f12053` · stack `e7973486fc8e4dfda1f2a4eb3207920f`.

## How to read the decision metrics (unchanged from before)
Each model run carries `eval/oot_new_links/report.json` (full per-scenario × per-track
table). Cross-trial: `consolidated_results.py` (table above) or
`rebuild_decision_comparison.py` (headline parquet). One report:
`artifacts.load_eval("<run_id>","oot_new_links")` → the `decision_report` metric.
Vocabulary contract: [`to-do/decision-metric-vocabulary.md`]. Headline = scenario 2
(income>500, BR-match), UW track, `approval_rate_delta`. Selection stays on `test`;
day2/decision metrics are a study, never `set_as_primary_label`.

## What the decision-focus model does
`boundary_weighted_xgb` (trial dir `experiments/neobank_ncm/neobank_ncm_v3_replicate/
boundary_weighted_xgb/model.py`): two-stage. Stage-1 XGB scores the soft-label
dual-record training rows; a Gaussian weight bump centered at the ~30%-approval
percentile (`BOUNDARY_Q=0.30, SIGMA=0.12, AMP=3.0`) upweights the contested cohort;
stage-2 XGB refits on those weights. Preprocessing = WoE(bank) + OHE(payfreq) +
numeric inf→nan / median-impute + missingness-flag / scale. 168 deployable features.
`boundary_weighted_lgbm` is the same recipe with LightGBM.

## Next steps (recommended)

### A. Promote / harden the decision-focus winner — *do first*
The +6.45% is one OOT time-snapshot. It already holds across all scenarios × both
tracks (checked). Before production: confirm on a refit/second time window, then
promote a learning to `000_overview`. Recommend **deploying `boundary XGB` or
`boundary LightGBM`** (single tree, ~446 ms, clean validation).

### B. Tune the decision-focus lever — *now appropriate (was deferred)*
The boundary knobs were fixed heuristics. Sweep `Q / SIGMA / AMP`; try centering the
band on the **actual matched-bad-rate threshold** rather than a fixed 30% percentile;
try a hard band (train stage-2 only on the cohort) vs the soft Gaussian. Cheap, and
the lever clearly has signal.

### C. Principled decision objective (higher-ceiling, needs a small data join)
Replace the self-referential boundary heuristic with a target that *is* the decision:
**incumbent-disagreement weighting** (upweight rows where the candidate would swap
the v3a decision) or a **rank-at-threshold / pairwise loss** on the marginal cohort.
Blocker: `v2_score` (incumbent) is **not in the training frame** (only the eval frame)
— join it in at materialize time first. This is the most promising untested direction.

### D. Latency (separate ops track, if real-time matters)
~446 ms/row is **preprocessing-bound** (WoE + ColumnTransformer + per-row pandas),
not model-bound — and it's the same for every family. Optimize the transform
(vectorize WoE, drop per-row pandas) to cut latency for *all* models at once.

### E. Do NOT re-chase (evidence this session): deeper nets, ensembling/stacking,
and embedding-as-features — all failed to beat a well-preprocessed single tree here.
The one untested representation idea is a **self-supervised DAE** (unsupervised
structure, unlike the supervised embedding that was marginal) — low priority given
the embedding-transfer result, but the only representation stone unturned.

### F. Torch serving SIGSEGV (if ever deploying a torch model)
The stack / boundary-stack / mlp_embed (all torch) SIGSEGV in serving validation
(torch + xgboost OpenMP thread clash in the fresh subprocess). Fix: `nthread=1` on the
XGB and/or `OMP_NUM_THREADS=1` in the serving image. Not needed for the recommended
single-tree models. (We did not re-run the stack just to measure its latency, since
it isn't the recommendation.)

## Where things live (this session's additions)
- **Winning models:** `experiments/neobank_ncm/neobank_ncm_v3_replicate/{boundary_weighted_xgb,
  boundary_weighted_lgbm,boundary_baseline_xgb,boundary_weighted_stack,mlp_embed_xgb,xgb_mlp_stack}/model.py`
- **Scripts:** `scripts/consolidated_results.py` (the table above + latency),
  `scripts/blend_study.py` (post-hoc rank-blend study),
  `scripts/score_trial_decision_chunked.py` (decision eval, **now persists predictions** —
  the fix so reruns/blends reuse them), `scripts/rebuild_decision_comparison.py`.
- **Staging copies** of the new models live at `projects/neobank_ncm/staging_*.py` (source
  of the trial `model.py`s) and `scripts/_smoke_*.py` (in-process smoke harness; the runner's
  formal `--dry-run` is a different container and the prefix sampler empties the temporal
  test split, so smoke-test in-process instead). Prunable.
- **Batch log:** `.cache/automl/fin/AUTONOMOUS_BATCH.md` (overnight run tracker).

## Gotchas
- **OFF-VPN for everything.** GCS flaps on VPN. Full-data reads also hit occasional
  corrupted-bytes/oauth transients even off-VPN (the registry retries; mlp_embed's
  serving validation still caught one → trial FAILED but model was logged).
- **Decision eval for non-tree / heavy models: use `score_trial_decision_chunked.py`**
  (250k-row chunks over local frames). Native `evaluate()` thrashes them.
- **Runner pre-fit guard** fits on `head(200)`/`head(10)`; any new model must be
  tiny-sample-safe (a meta-learner needing both classes will fail there — guard it).
- **Smoke-test new model.py in-process** (`scripts/_smoke_model.py <path>`) before the
  full run; it mirrors the pre-fit guard and the contract check.
- Selection on `test` AUC only; decision metrics never drive selection.
