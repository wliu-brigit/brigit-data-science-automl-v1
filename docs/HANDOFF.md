# Handoff — continue here

Temporary hand-over note: where the last session left off and what's open, so a
new session can pick up cold. **Not a changelog** — keep only what's current and
relevant. Rewritten at each wrap, not appended to.

> **These docs are best-effort documentation. The code is the source of truth
> for current behavior.**

**Last updated:** 2026-06-12 — branch `neobank_ncm_v3_replicate`. This session
(1) built the **native decision/financial re-eval** (settled vocabulary, all
scenarios, recorded through the harness eval flow), (2) **re-validated the AUC ↔
business-metric divergence** with the bug-fixed numbers — **it holds**, and (3)
hit + worked around a **memory wall** in `evaluate()` for non-tree models.

## TL;DR — where we are

1. **Native decision re-eval is built and run for all 5 trials.** Each trial now
   has `eval/oot_new_links/report.json` on its MLflow run: the full per-scenario ×
   per-track decision table, settled metric names, recorded the same way every
   other eval is. No more ad-hoc `newlinks.*`.
2. **The divergence finding is VALIDATED** (was tentative — rested on a gate bug).
   Corrected ΔAR roughly halved vs the buggy values, but the headline holds: the
   **MLP ties the trees on approval-gain at ~0.025 lower AUC**, and beats the GAM
   **3×** at near-identical AUC. **Don't select on AUC alone.**
3. **`evaluate()` thrashes swap for non-tree models** on the 5.3M-row frame (it
   predicts the whole frame at once). Tree models are fine; the GAM ran 5 hours
   paging before we killed it. Workaround in place; core fix parked.

## How to read the decision metrics (for the next session)

The numbers live on each trial run as a structured artifact — three ways in:

- **Cross-trial table (easiest):**
  `uv run python projects/neobank_ncm/scripts/rebuild_decision_comparison.py`
  prints + writes `.cache/automl/fin/decision_comparison.parquet` (headline
  scenario-2 UW numbers + `day2_known_auc` for all trials in its `TRIALS` map).
- **One trial's full report (all 7 scenarios, both tracks):**
  ```python
  from automl.project import use_project; use_project("neobank_ncm")
  from automl.mlflow.trial import artifacts
  r = artifacts.load_eval("<run_id>", "oot_new_links")
  rep = next(m["value"] for m in r.metrics if m["name"] == "decision_report")
  rep["scenarios"]["2_income500_match_bad_rate"]["tracks"]["uw"]   # ΔAR etc.
  rep["scenarios"]["2_income500_match_bad_rate"]["ltv_per_link_d90"]
  rep["benchmark"]; rep["discrimination"]
  ```
- **Scalar in MLflow:** `eval.oot_new_links.day2_known_auc` is the only logged
  scalar (deliberately — decision metrics never drive selection, so it is never
  `set_as_primary_label`).

**What the names mean** is the contract in
[`docs/to-do/decision-metric-vocabulary.md`](to-do/decision-metric-vocabulary.md):
`candidate` = model under eval, `v3a` = incumbent production strategy, `cle` =
lenient track, `*_delta` = candidate − v3a. Scenarios keyed `2_income500_match_bad_rate`
etc. (legacy number + KO-gate + objective). Family 3 (ΔAR, swap sets) is per-track;
Family 4 (LTV-per-link) is per-scenario over the combined UW∪CLE population.
**Headline = scenario 2 (income>500, BR-match), UW track, `approval_rate_delta`.**

## Re-validated comparison (scenario 2, UW track)

| trial | run_id | family | test AUC | day2 AUC | ΔAR | swap-in BR |
|---|---|---|---|---|---|---|
| 11 | `b3e5efdb9a924157b4ca521022ccf816` | WoE scorecard | 0.656 | 0.632 | +0.37% | 0.221 |
| 12 | `9c6b2e176e9f45888d7489be3e38aedc` | spline GAM | 0.663 | 0.643 | +1.46% | 0.218 |
| 13 | `e4ea3b7256924bdc83e942e79bb85715` | compact MLP | 0.684 | 0.648 | **+4.23%** | 0.209 |
| 1 | `51bd38d4bcb845cbbad52dcacd637e1e` | XGBoost | 0.700 | 0.672 | +4.29% | 0.212 |
| 3 | `2f39e0ead13d4a588e4a385f272dc38f` | XGBoost | 0.701 | 0.673 | +4.65% | 0.211 |

(Trial 7 excluded — not deployable, 114 leaky features.) LTV-per-link is invariant
(~$3.0 D90 / ~$3.1 D120 everywhere) — dominated by population economics, not the
model. Eval dataset: `oot_new_links_with_ltv` = `ev_abb30380d8bc`.
Draft learning (local, not yet promoted):
`.cache/automl/learnings/auc-vs-business-divergence.md`.

## Chunked prediction — what it is, and what it supports

`evaluate()` predicts the **entire** 5.3M-row frame in one `model.predict` call
(`automl/eval/evaluate.py` `_predict_model`). For models whose preprocessing
materializes a big dense matrix (spline GAM, torch MLP) that exhausts RAM →
swap-thrash. Trees (XGB) are fine.

- **`scripts/score_trial_decision_chunked.py`** is the workaround: it scores via
  `scoring.score_daily` → `TrialModel` (250k-row chunks) over the **local cached
  frames** (no 2 GB GCS read), then records the same `eval/oot_new_links/report.json`.
- **It supports every model we have.** The chunker is model-agnostic — it slices
  the input and calls the pyfunc `.predict()` per chunk; XGB / scorecard / GAM / MLP
  all work identically. **Use it as the default for any decision re-eval; reserve
  native `evaluate()` for tree models only** (until the core fix lands).
- **Core fix parked:** [`docs/to-do/eval-chunked-prediction.md`](to-do/eval-chunked-prediction.md)
  — add chunked prediction to the eval path so any large eval dataset is safe
  generically. Promote this if decision re-eval becomes routine.

**How to score a new/non-tree trial:** off-VPN, one at a time, watching RSS:
`uv run python projects/neobank_ncm/scripts/score_trial_decision_chunked.py --model-run-id <id>`
(healthy = high CPU + bounded RSS; the thrash signature is *low* CPU + high RSS +
climbing swap — kill it and chunk instead).

## Next steps

### A. Explore new directions — *not just seeds, and not just architectures* (the main ask)
Tried so far: trees (XGB), linear (WoE scorecard), additive (spline GAM), neural
(compact MLP) — all hit the ~0.70 AUC ceiling. The directions below are **things to
LEARN whether they apply, not a checklist to apply blindly** — each has a
precondition or a data dependency that must be checked first; some may not be
feasible here. All work stays within the **168 deployable features**
(`projects/neobank_ncm/data/deployable_features.json`), `--max-budget-usd 5` per
run, **OFF-VPN**, and is read on **ΔAR + swap-in BR via the chunked decision
script**, not just AUC. Loop pattern:
`automl --project neobank_ncm experiment run --max-iter 1 --max-budget-usd 5 --instruction "..."`.

**Evidence framing (Home Credit + tabular-credit literature):** GBMs won that
competition; fancy nets (TabNet, FT-Transformer, DAE) did *not* reliably beat GBMs
and cost more compute — NN mostly added value inside **blends/stacks**, and the real
lever was **feature engineering over the relational/temporal tables**, not the model.
So the high-value bets here are **features + framing + blend**, *not* a deeper net.
Sources: [DAE+GBDT HC writeup](https://github.com/pklauke/Kaggle-HomeCreditDefaultRisk),
[2024 tabular ML/DL benchmark](https://arxiv.org/html/2408.14817v1),
[TabNet-stacking credit paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11506879/).

1. **Temporal / trajectory features over the D1–D30 window — IF the data supports it
   (verify FIRST).** We collapse the daily sequence to a min-score and discard the
   trajectory. Our analog of HC's aggregations would be slope/volatility/range/
   recent-vs-early deltas over the 30 days. **Precondition / may not apply:** the
   production daily *scoring* path must be able to compute cross-day features at score
   time — we may **not have** the per-user daily history available in production (some
   168 features are already lookback-windowed like `inflowsum14d`, but cross-*day*
   evolution is new). **Check deployability before investing**; if production only sees
   the current day, this is out.
2. **Blend / stack the existing deployable XGB + MLP — applies regardless (data is in
   hand).** This is where NN paid off in HC. Our MLP is decorrelated from XGB (ties
   trees on ΔAR at lower AUC = recovers different signal), so a simple stack may beat
   either *and* lift ΔAR. Cheap. The most reliable near-term bet.
3. **Reframe toward the decision, not global AUC — explore if the labels support it.**
   ΔAR rewards ranking the marginal *swap-in* cohort. Options: model the swap-in
   population's risk directly, a residual-vs-v3a framing (learn where to disagree with
   the incumbent), or a rank-at-threshold loss. **May not apply:** the swap-in cohort
   is small and partly *unlabeled* (rejects) — confirm there's enough labeled signal
   before framing the target this way.
4. **If touching the NN, keep it proven + light (don't go deep).** Skip TabNet/FT-T
   (underperform + compute). Cheap evidence-backed tweaks: RankGauss/quantile input
   normalization + categorical embeddings on the existing MLP; or a tiny 1D-CNN / small
   GRU over D1–D30 (only if direction #1's data check passes). No deep stacks.

### B. Harden the divergence finding before promoting it
One OOT snapshot, scenario 2 only. Confirm the MLP's parity-with-trees with a
**re-fit seed** and a **second scenario**, then promote
`auc-vs-business-divergence.md` to the experiment `000_overview` learnings.

### C. (Optional) the chunked-prediction core fix — see §Chunked above.

## Legacy fidelity — audited 2026-06-12 (don't redo)
The decision-eval helpers (`analysis/{policy,impact,scoring,data}.py`) are a
faithful port of `financial_impact_analysis.ipynb`; the one real bug (no-KO ΔAR
paired with a scenario-gated LTV) is fixed and validated — corrected trial-1 ≈
legacy sc2 (ΔAR +4.3% vs +4.5%). The native re-eval reproduces it (trial-1
day2_auc 0.6725, ΔAR +0.0429). Difference from production is the retrain vs the
production artifact, not a code gap. LPL has a live `user_ltv.sql` drift source
(non-snapshotted) — consider freezing it to a dated snapshot for reproducibility.

## Where things live
- **Decision-eval code:** `projects/neobank_ncm/analysis/report.py` (assembly),
  `projects/neobank_ncm/eval/metrics.py` (metrics + spec).
- **Scripts:** `scripts/prepare_oot_new_links_dataset.py` (one-time dataset
  materialize), `scripts/score_trial_financials.py` (native evaluate driver — trees),
  `scripts/score_trial_decision_chunked.py` (chunked — all models),
  `scripts/rebuild_decision_comparison.py` (cross-trial table),
  `scripts/evaluate_split.py` (the cross-population AUC precedent).
- **Docs:** [`to-do/decision-metric-vocabulary.md`](to-do/decision-metric-vocabulary.md)
  (naming + recording contract), [`to-do/native-decision-reeval-plan.md`](to-do/native-decision-reeval-plan.md)
  (the build), [`to-do/native-reeval-decision-metrics.md`](to-do/native-reeval-decision-metrics.md)
  (the larger "level-2" capability), [`to-do/eval-chunked-prediction.md`](to-do/eval-chunked-prediction.md).
- **Deployable asset:** `projects/neobank_ncm/data/deployable_features.json` (168).
- **Scratch cache (prunable, local):** `.cache/automl/fin/*.parquet` (daily 1.7 GB,
  user_ltv, ri_scores, `decision_comparison.parquet`), `.cache/automl/learnings/`.

## Gotchas
- **Run everything OFF-VPN.** GCS flaps on VPN (~0.5 MB/s, hangs). The decision eval
  is GCS-heavy (one ~2 GB dataset write + reads); none of it needs Snowflake (frames
  cached). The training loop also flaps + orphans trials on VPN.
- **Non-tree decision evals: use the chunked script, one at a time, watch RSS.**
  Native `evaluate()` will thrash them.
- **torch trials** SIGSEGV the eval without single-thread env (`OMP_NUM_THREADS=1`
  etc.); both decision scripts set this at import.
- **`.env` not auto-loaded** by `uv run python`; the scripts load it (GCS/MLflow
  creds + `REQUESTS_CA_BUNDLE`/`SSL_CERT_FILE` for the VPN TLS intercept).
- **OOT discipline:** the decision eval is a metric *study*, never trial selection.
  Keep selection on `test`; `day2_known_auc` is logged but never the primary label.
