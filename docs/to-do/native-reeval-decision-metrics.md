# Native re-evaluation for decision / financial metrics

**Status: to-do (design).** Make post-training **decision-metric re-evaluation**
(approval-rate gain at matched bad rate, LTV-per-link, swap-in BR on the OOT
population) a **first-class, natively-integrated** capability of the harness —
instead of the ad-hoc project script `projects/neobank_ncm/scripts/score_trial_financials.py`.
Driving case: neobank_ncm. Parent architecture question:
[out-of-sample eval & dataset management](out-of-sample-eval-and-dataset-management.md).

**Settled + planned:** the naming/recording contract is
[`decision-metric-vocabulary.md`](decision-metric-vocabulary.md); the task-by-task
build is [`native-decision-reeval-plan.md`](native-decision-reeval-plan.md). One
known core gap surfaced — `evaluate()` predicts the whole frame at once — parked as
[`eval-chunked-prediction.md`](eval-chunked-prediction.md).

## What we want

Decision metrics should flow through the **same eval/re-eval machinery** as every
other metric, so they are queryable, comparable across trials, and provenance-clean:

- A managed **`EvalDataset`** for the OOT decision frame — `oot` (the
  `oot_new_links_daily` frame), and `oot_with_ltv` when the per-user LTV is joined
  in — instead of hand-cached `.cache/automl/fin/*.parquet`.
- Decision metrics as project **`Metric` subclasses** (`projects/neobank_ncm/eval/metrics.py`):
  approval-rate-at-matched-BR, ΔAR-vs-incumbent, swap-in BR, LTV-per-link.
- Run via the **re-evaluation flow** → standard `eval.oot.<metric>` metrics +
  `eval/oot/report.json` artifact + eval-index entry, exactly as
  `scripts/evaluate_split.py` adds `oot`/`train_known`. `evaluate(..., eval_spec=...)`
  already accepts a custom EvalSpec (`automl/eval/evaluate.py:27,56`), so this needs
  **no change to the training runner**.

## How far to push "native"

Two levels, smallest first:
1. **Re-eval-flow only (near-term):** the above — EvalDataset + custom Metrics +
   `evaluate(eval_spec=...)`. No core change.
2. **Eval-machinery awareness (larger):** teach the eval/re-eval path to natively
   validate context-rich metrics — `required_columns`, `required_augmentations`,
   and the daily→user **grain/collapse** — so a decision metric declares its needs
   and the machinery checks them (the "catch missing required columns" goal). This
   is the bigger architecture change. The **universal training runner stays
   generic** — decision re-eval is a post-training step, not part of training.

## Design requirements (must hold by construction)

- **One consistent policy per scenario drives every metric.** A scenario = KO gate
  × match objective; ΔAR, swap-in, and LTV must all use that scenario's gate. (The
  ad-hoc script previously mixed a no-KO ΔAR with a scenario-gated LTV — the native
  design must make this impossible.)
- **Report the full per-scenario set, not a cherry-picked row.** The legacy emits
  the complete table for every KO variant × objective, UW and CLE rows. Log that
  set (per scenario, both tracks, swap-in *and* swap-out), not one hand-picked
  number.
- **Be explicit about track scope** per metric — UW-only vs UW∪CLE (loan amounts
  $50 / $25). Don't conflate a UW-track ΔAR with a UW∪CLE LTV in one flat record.
- **Confidence intervals.** Emit CIs / a power-sizing view, not point estimates
  only — a go/no-go decision needs them.
- **Provenance** with the numbers: scenario, data snapshot id, incumbent benchmark,
  population N, and the LTV-pull date (LTV is a live, non-snapshotted pull).
- **Vocabulary:** the full settled naming — populations `oot_new_links` /
  `oot_new_links_with_ltv`, `candidate` vs `v3a`, spelled-out metric names, and the
  `report.json` shape — is the contract in
  [`decision-metric-vocabulary.md`](decision-metric-vocabulary.md). Metrics are
  `eval.oot_new_links.*`; no foreign `newlinks.*` namespace.

## Wrinkles to solve

- **Grain + LTV join.** Metrics need a daily→user collapse (min model score over
  policy-eligible days) *between* scoring and the metric; `evaluate()` hands
  `(df, y_pred)` at one grain. A custom metric can collapse internally (it sees the
  whole `df`), but LTV is **user-grain** and the augmentation join is `one_to_one`,
  so LTV can't augment the daily frame at runtime — **pre-join LTV into the
  `oot_with_ltv` EvalDataset** (broadcast per user) so the metric reads it as
  columns. That is the one non-standard move.
- **Incumbent benchmark is intrinsic** to the metric (ΔAR is relative to the fixed
  V3A policy computed from the same frame) — the metric needs a reference policy,
  not just `(y_true, y_score)`.
- **Scenario parameterization** (1–7) as a metric/eval-dataset config, recorded and
  namespaced.
