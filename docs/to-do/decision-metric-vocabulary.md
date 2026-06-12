# Decision-metric vocabulary + recording shape — settled design

**Status: to-do (design, decided).** The agreed naming **and recording shape** for
the neobank_ncm decision/financial metrics, so they align with the harness
vocabulary instead of the foreign `newlinks.*` / `lpl` / `v3`/`v3a` shorthand the
ad-hoc script emits today. This is the **contract** the native re-eval effort
([`native-reeval-decision-metrics.md`](native-reeval-decision-metrics.md))
implements; it does not by itself change code. Current behavior:
`projects/neobank_ncm/scripts/score_trial_financials.py` +
`projects/neobank_ncm/analysis/{policy,impact,scoring,data}.py`.

Decided 2026-06-12 with the user. Supersedes the `newlinks.*` namespace.

## Why

The decision metrics are a faithful port of the legacy
`financial_impact_analysis.ipynb`, and the port kept the legacy's shorthand:
`newlinks.*` namespace, `lpl`, `d2_auc`, `v3`/`v3a` column prefixes. None of it
matches the harness's own vocabulary (`eval.<label>.<metric>` keys, spelled-out
names, population-as-label). This document fixes the names once, and pins **how
the results are recorded** so they read like every other evaluation in the system.

## The lineage (what the names describe)

The metrics are born in a fixed sequence; the names mirror it.

```
Stage 0  score_daily        → candidate model scores every (user, day) row
Stage 1  add_policy_columns → per-day gates: account-eligible, KO variants,
                              v3a-pass, cle-pass
Stage 2  collapse_to_users  → one row/user. User's policy score = MIN candidate
                              score over that user's eligible days (per KO variant);
                              + effective_bad, + v3a-approved / cle-approved flags
Stage 3  benchmarks         → the v3a incumbent's realized numbers on this
                              population (approval rate, bad rate) — the match targets
Stage 4  compute_thresholds → the candidate score cutoff that matches either the
                              v3a approval rate (AR-match) or bad rate (BR-match)
Stage 5  threshold_table    → per scenario, apply the cutoff: candidate vs v3a —
                              approval rate, ΔAR, swap-in / swap-out and their bad rates
Stage 6  impact.py (LTV)    → join per-user LTV → LTV-per-link ($/applicant), D90 / D120
Stage 7  d2_known_auc       → candidate score AUC on day-2 known users (scenario-free)
```

## Naming conventions

- **`candidate`** — the model under evaluation (the trial being scored). Replaces
  the overloaded `v3` prefix, which read as a variant of `v3a` when it is in fact
  the *opposing side* of the comparison.
- **`v3a`** — the live production strategy (`UNDERWRITING_NEOBANK_STRATEGY_V3A`):
  `v2_score ≤ 0.485` + triple knockout + account-eligible. Kept verbatim — the real
  strategy name, familiar to the DS team. The legacy "reference" (`ref_*`) columns
  *are* v3a; they collapse to `v3a_*`. (Version-specific by design; fine for this
  release, may generalize later — see `native-reeval-decision-metrics.md`.)
- **`cle`** — the lenient second approval track (`v2_score ≤ 0.64` & income>500),
  $25 loan vs the UW track's $50.
- **`*_delta`** — candidate − v3a, consistently, for every comparison. The sign
  carries the direction (no `gain` vs `delta` mix).
- All acronyms spelled out: `ar → approval_rate`, `br → bad_rate`,
  `lpl → ltv_per_link`, `vol → count`, `thr → score_cutoff`, `d2 → day2`,
  `d1 → day1`.

### swap-in / swap-out vs bad_rate_delta (read this — it is subtle)

`v3a` and `candidate` approve two **overlapping** user sets. Most approved users
are in both (same decision); the **swap set** is who *changes*:

- **swap-in** = candidate approves, v3a rejects → the *marginal new approvals*.
- **swap-out** = v3a approves, candidate rejects → the *dropped*.

`bad_rate_delta` (= `candidate_bad_rate` − `v3a_bad_rate`) compares the two **whole
portfolios**, which are dominated by the shared core — so it is *diluted*, and in a
**BR-match scenario it is ≈ 0 by construction** (we set the cutoff to match v3a's
bad rate). `swap_in_bad_rate` isolates **only the marginal new approvals** — the
real underwriting question, "of the extra people I'd approve by switching, how many
go bad?" Example: shared core 10,000 @ 14% + 1,000 swap-in @ 25% →
`candidate_bad_rate` 15.0%, `bad_rate_delta` +1.0pt (looks harmless), but
`swap_in_bad_rate` 25% (the real risk). **In a matched-BR world `bad_rate_delta` is
near-useless; `swap_in_bad_rate` is the signal.**

## Population, EvalDataset, and label

Distinct from the existing **`oot`** split (known-only, user-grain, used for AUC by
`scripts/evaluate_split.py`). The decision frame is a different population and grain
— the full daily scoring population, including unknown/rejected users.

- **`oot_new_links`** — the OOT-window daily new-links *scoring* frame (everyone
  production scores), day grain, ~5.3M rows. The upstream data artifact.
- **`oot_new_links_with_ltv`** — that frame collapsed to **user grain** with per-user
  LTV pre-joined. This is the **managed `EvalDataset`** the evaluation consumes: the
  superset that makes every metric family computable in one pass (the daily→user
  collapse and the LTV join are done when materializing it, per the to-do's "pre-join
  LTV" note).
- **Eval label = `oot_new_links`.** One evaluation, one label = the population. Metric
  keys are `eval.oot_new_links.<metric>`. (The label is the short population name; its
  backing EvalDataset is the `_with_ltv` superset. Keeping the label short avoids
  `eval.oot_new_links_with_ltv.*` noise on every key.)

## Scenarios

A scenario = **KO gate × match objective**. All seven are already computed
(`policy.scenario_map`, an exact legacy port); the ad-hoc script merely *picks* one.
We compute and record **all seven — nothing cherry-picked** — as nested entries in
one report (see "Recording" below), keyed by the legacy number + a descriptive name:

| Scenario key | KO gate | objective | family |
|---|---|---|---|
| `1_no_ko_match_bad_rate` | none | match bad rate | BR-match (the "no-KO" baseline) |
| `2_income500_match_bad_rate` | income>500 | match bad rate | BR-match — **headline** (legacy PICKED) |
| `3_v3a_ko_match_bad_rate` | V3A KOs | match bad rate | BR-match |
| `7_income500_broad_match_bad_rate` | income>500 broad | match bad rate | BR-match |
| `4_no_ko_match_approval_rate` | none | match approval rate | AR-match (dual) |
| `5_income500_match_approval_rate` | income>500 | match approval rate | AR-match (dual) |
| `6_v3a_ko_match_approval_rate` | V3A KOs | match approval rate | AR-match (dual) |

- Numbers are the **legacy `scenario_map` ids** (preserve the DS team's "scenario 2"
  muscle memory); the descriptive suffix is the clarity. `no_ko` *is* the no-KO-gate
  baseline — no decoder ring.
- **BR-match (1, 2, 3, 7)** is the business objective: hold DPD45 at the v3a level,
  read how much more we approve (ΔAR + swap-in bad rate). The headline.
- **AR-match (4, 5, 6)** is the dual cross-check: hold approvals, read the bad-rate
  change.
- **The legacy parameter block carries nothing new for us:** `scenario_map` is an
  exact port; `LAM_UW=50` / `LAM_CLE=25` are already `policy.py` constants;
  `PICKED_SCENARIO=2` just marks which scenario is the headline (scenario 2). The
  **only** un-ported piece is `SAMPLING_RATE` (partial-rollout fraction) — we report
  the full-rollout (1.0) projection only; it is a linear scalar on projected volume,
  not a compute knob, and can be added later.

## The metric names

`candidate` = model under eval · `v3a` = incumbent · `cle` = lenient track ·
`*_delta` = candidate − v3a.

### Family 1 — Discrimination (scenario-free)
| Today | Name |
|---|---|
| `d2_auc` | `day2_known_auc` |
| `d2_n` | `day2_known_count` |
| `d2_bad_rate` | `day2_known_bad_rate` |

### Family 2 — Incumbent benchmark (scenario-free; the match targets)
| Today | Name |
|---|---|
| `v3a_ar` / `incumbent_ar` | `v3a_approval_rate` |
| `v3a_br` / `incumbent_br` | `v3a_bad_rate` |
| `cle_ar` | `cle_approval_rate` |
| `cle_br` | `cle_bad_rate` |

### Family 3 — Approval comparison (per scenario × per track: UW, CLE)
| Today | Name |
|---|---|
| `v3_thr` | `candidate_score_cutoff` |
| `n_v3` | `candidate_approved_count` |
| `v3_ar` / `ar_at_matched_br` | `candidate_approval_rate` |
| `ref_ar` | `v3a_approval_rate` |
| `delta_ar` | `approval_rate_delta` |
| `v3_br` | `candidate_bad_rate` |
| `ref_br` | `v3a_bad_rate` |
| `delta_br` | `bad_rate_delta` |
| `swap_in_br` | `swap_in_bad_rate` |
| `swap_out_br` | `swap_out_bad_rate` |
| `swap_in_vol` | `swap_in_count` |
| `swap_out_vol` | `swap_out_count` |
| `d1_ar` | `day1_approval_rate` |

### Family 4 — Financial (per scenario; needs the LTV join)
| Today | Name |
|---|---|
| `lpl_90` | `ltv_per_link_d90` |
| `lpl_120` | `ltv_per_link_d120` |

`v3a_approval_rate` / `v3a_bad_rate` appear in both Family 2 and Family 3 — they are
the same reference numbers, repeated per row for readability.

## Recording (Design B — one eval, scenarios nested, no selection primary)

We route through the existing eval/re-eval flow (`evaluate(..., eval_spec=...)`) —
**no core automl change** — but deliberately use only the half that fits.

**As built (2026-06-12):** the `oot_new_links_with_ltv` EvalDataset is materialized
**once** as an `external` dataset via `prepare_eval_dataset` (model-independent: the
daily frame + `add_daily_derived_features` + per-user LTV broadcast per daily row;
`unique_key=(user_id, day_number)`, `target_col=went_dpd45`). Every trial points at
that one id (e.g. `ev_abb30380d8bc`). The driver
(`scripts/score_trial_financials.py`) calls `evaluate(...)` per trial; the dataset is
read back, the model scores it, and `DecisionReport` runs. Validated: trial-1 native
re-eval reproduced the legacy/handoff numbers (`day2_known_auc` 0.6725,
`approval_rate_delta` sc2/uw +0.0429, `swap_in_bad_rate` 0.2125, `ltv_per_link_d90`
3.036). **Caveat:** `evaluate()` predicts the whole 5.3M-row frame in one call (ran
fine for the XGBoost trial; the reliability fix is
[`eval-chunked-prediction.md`](eval-chunked-prediction.md)).

- **One evaluation**, `label = oot_new_links`, on the `oot_new_links_with_ltv`
  EvalDataset. **Not** one-eval-per-scenario: scenario is a *policy parameter*, not a
  population, and `label` means population everywhere else in the system. Overloading
  the label axis with scenario would duplicate the 5.3M-row predictions per scenario
  and explode combinatorially against real populations.
- **Scenario is a dimension *inside* the report**, not a label and not repeated across
  metric names. The full per-scenario × per-track table is a **structured (non-scalar)
  metric** whose value nests `{scenario_key: {uw: {...}, cle: {...}}}`. The harness
  supports this directly: non-primary metric values are `Any` and survive into
  `report.json` (`automl/eval/base.py:79–87`, `automl/eval/results.py:32`); only the
  MLflow *scalar* promotion filters to finite floats.
- **No selection primary.** Decision metrics must **never** drive trial selection
  ("keep selection on `test`"). The schema forces us to *name* a scalar primary and
  always logs it, so we fill that slot honestly with **`day2_known_auc`** — the one
  genuinely scenario-free single-number summary (pure discrimination) — and we
  **never** call `set_as_primary_label`. It is logged as `eval.oot_new_links.day2_known_auc`
  but is never the run's selection metric. Nothing scenario-specific is ever promoted
  to a primary.
- **Scalar promotion of decision numbers → deferred.** Beyond the discrimination
  primary, no scenario-specific scalars are logged yet. Which headline to promote
  later (likely `2_income500_match_bad_rate` → `approval_rate_delta` + `swap_in_bad_rate`)
  is an additive decision once we have read the full report.
- **The linking "one report" is the eval index** — it lists this evaluation and the
  report path; the structured `report.json` *is* the per-scenario detail, organized so
  it reads as sections rather than one flat overwhelming list.

### `report.json` shape (target)

```json
{
  "provenance": {
    "candidate_run_id": "<trial run id>",
    "eval_dataset_id": "<oot_new_links_with_ltv snapshot id>",
    "ltv_pull_date": "YYYY-MM-DD",
    "daily_rows": 5269592,
    "population_users": 177528,
    "headline_scenario": "2_income500_match_bad_rate",
    "sampling_rate": 1.0
  },
  "discrimination": { "day2_known_auc": 0.0, "day2_known_count": 0, "day2_known_bad_rate": 0.0 },
  "benchmark": {
    "v3a_approval_rate": 0.0, "v3a_bad_rate": 0.0,
    "cle_approval_rate": 0.0, "cle_bad_rate": 0.0
  },
  "scenarios": {
    "2_income500_match_bad_rate": {
      "ko_gate": "income500",
      "objective": "match_bad_rate",
      "ltv_per_link_d90": 0.0, "ltv_per_link_d120": 0.0,
      "tracks": {
        "uw":  { "candidate_score_cutoff": 0.0, "candidate_approved_count": 0,
                 "candidate_approval_rate": 0.0, "v3a_approval_rate": 0.0,
                 "approval_rate_delta": 0.0, "candidate_bad_rate": 0.0,
                 "v3a_bad_rate": 0.0, "bad_rate_delta": 0.0,
                 "swap_in_bad_rate": 0.0, "swap_out_bad_rate": 0.0,
                 "swap_in_count": 0, "swap_out_count": 0, "day1_approval_rate": 0.0 },
        "cle": { "...": "same keys" }
      }
    },
    "1_no_ko_match_bad_rate": { "...": "same shape" }
  }
}
```

- **Confidence intervals / power view** (the legacy `sample_size_analysis`, ported in
  `impact.py`) are a planned addition (per-scenario and/or provenance); exact shape
  decided during implementation. The to-do requires CIs, not point estimates only.
- **LTV-per-link is per-scenario, over the combined UW∪CLE approved population**
  (legacy-aligned — the legacy computes one LTV number per scenario, not per track), so
  it lives at the scenario level, *outside* the `tracks`. Family 3 (ΔAR, swap sets) is
  per-track; Family 4 (LTV) is per-scenario. This also keeps the track-scope honest: a
  per-track ΔAR is never conflated with a combined-population LTV.

## The fork we chose, and what we parked

- **(B) chosen:** the above — route through `evaluate()`, one eval, structured report,
  discrimination-only primary, no selection role. Minimal, no core change.
- **(C) parked:** treat the decision study as a *first-class report capability* with
  its own schema (EvalDataset in, structured report out) that does **not** route
  through the Metric/primary machinery — because that abstraction (one scalar per
  `(df, y_pred)`) fits awkwardly against "needs an incumbent benchmark + a daily→user
  collapse + a scenario sweep." This is the larger architecture move; it lives in
  [`native-reeval-decision-metrics.md`](native-reeval-decision-metrics.md) as the
  "level 2 (eval-machinery awareness)" option.

## Out of scope here

Naming + recording shape only. The mechanics — materializing the
`oot_new_links_with_ltv` EvalDataset, the project `Metric` subclass(es) that emit the
structured report, the daily→user collapse, running via `evaluate(..., eval_spec=...)`
— are the [`native-reeval-decision-metrics.md`](native-reeval-decision-metrics.md)
effort. The names and shape above are the contract it implements.
