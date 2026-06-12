# Native Decision Re-eval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the neobank_ncm decision/financial evaluation a native, dataset-first eval — one scoring pass per trial produces a structured all-scenario report (settled vocabulary) recorded through the harness eval flow — and re-validate the AUC↔business divergence across trials 1/3/11/12/13 with the corrected numbers.

**Architecture:** Materialize the **model-independent** `oot_new_links_with_ltv` external `EvalDataset` once (daily frame + derived features + per-user LTV broadcast); every trial points at it. Per trial, route through `evaluate(eval_spec=...)` with a project `EvalSpec` = `Day2KnownAuc` (scalar primary, never `set_as_primary_label`) + `DecisionReport` (structured non-scalar metric carrying all 7 scenarios). The settled naming and report shape come from [`decision-metric-vocabulary.md`](decision-metric-vocabulary.md).

**Tech Stack:** Python, `uv`, pandas, `automl.eval` (`prepare_eval_dataset`, `evaluate`, `EvalSpec`, `Metric`), the existing `projects/neobank_ncm/analysis/*` helpers, pytest with the synthetic daily fixture.

**Known risk (accepted):** `evaluate()` predicts the whole frame in one `model.predict` call. The 5.3M-row frame may stress RAM (~7 GB dense transform). We accept this for now; the core fix is [`eval-chunked-prediction.md`](eval-chunked-prediction.md). If it OOMs in practice, promote that to-do.

---

## File structure

- **Create** `projects/neobank_ncm/analysis/report.py` — `build_decision_report(daily_scored, *, headline_scenario=2, provenance=None)`: pure assembly of the full structured report (all 7 scenarios, settled names, per-scenario LTV) from a **scored** daily frame (with LTV columns present). The one home for the legacy→settled rename + scenario sweep.
- **Create** `projects/neobank_ncm/eval/__init__.py` — exports `decision_eval_spec`.
- **Create** `projects/neobank_ncm/eval/metrics.py` — `Day2KnownAuc(Metric)`, `DecisionReport(Metric)`, `decision_eval_spec(...)`.
- **Create** `projects/neobank_ncm/scripts/prepare_oot_new_links_dataset.py` — one-time materialization of the external eval dataset.
- **Rewrite** `projects/neobank_ncm/scripts/score_trial_financials.py` — thin driver: resolve dataset id, `evaluate()` per trial, print summary.
- **Create** `projects/neobank_ncm/scripts/rebuild_decision_comparison.py` — read each trial's report.json, assemble the cross-trial comparison.
- **Modify** `docs/to-do/decision-metric-vocabulary.md` — recording section → as-built; LTV per-scenario; cross-link this plan.
- **Test** `projects/neobank_ncm/tests/test_decision_report.py` — `build_decision_report` + the metric subclasses on the synthetic fixture.

---

## Task 1: `build_decision_report` — the settled-vocabulary assembly

**Files:**
- Create: `projects/neobank_ncm/analysis/report.py`
- Test: `projects/neobank_ncm/tests/test_decision_report.py`

- [ ] **Step 1: Write the failing test**

```python
# projects/neobank_ncm/tests/test_decision_report.py
from __future__ import annotations

import numpy as np
import pandas as pd

from projects.neobank_ncm.analysis import policy, report, scoring
from projects.neobank_ncm.tests.fixtures import make_synthetic_daily


def _scored_daily_with_ltv():
    daily = make_synthetic_daily()                 # synthetic D1–D30 daily frame
    daily["v3_score"] = np.clip(                   # stand-in candidate score
        daily["v2_score"].astype(float) + np.random.default_rng(0).normal(0, 0.05, len(daily)),
        0, 1,
    )
    # broadcast cheap synthetic LTV columns per user (as the real frame will carry)
    for h in (30, 60, 90, 120):
        daily[f"total_revenue_{h}"] = 2.0
        daily[f"total_ltv_lite_{h}"] = 1.5
        daily[f"ltv_{h}_elig"] = True
    daily["loan_amount_max"] = 50.0
    daily["underwriting_strategy"] = "UNDERWRITING_NEOBANK_STRATEGY_V3A"
    daily["first_activation_date"] = pd.Timestamp("2026-01-15")
    return daily


def test_decision_report_structure_and_naming():
    rep = report.build_decision_report(_scored_daily_with_ltv(), headline_scenario=2)

    # the four families are present
    assert set(rep) >= {"discrimination", "benchmark", "scenarios"}
    assert set(rep["discrimination"]) == {"day2_known_auc", "day2_known_count", "day2_known_bad_rate"}
    assert set(rep["benchmark"]) == {
        "v3a_approval_rate", "v3a_bad_rate", "cle_approval_rate", "cle_bad_rate",
    }

    # all seven scenarios, keyed by the settled names
    assert set(rep["scenarios"]) == {
        "1_no_ko_match_bad_rate", "2_income500_match_bad_rate", "3_v3a_ko_match_bad_rate",
        "4_no_ko_match_approval_rate", "5_income500_match_approval_rate",
        "6_v3a_ko_match_approval_rate", "7_income500_broad_match_bad_rate",
    }

    sc = rep["scenarios"]["2_income500_match_bad_rate"]
    # LTV is per-scenario (combined UW∪CLE), NOT per-track — legacy-aligned
    assert {"ltv_per_link_d90", "ltv_per_link_d120", "ko_gate", "objective", "tracks"} <= set(sc)
    assert set(sc["tracks"]) == {"uw", "cle"}

    uw = sc["tracks"]["uw"]
    # settled Family-3 names, no legacy v3_/ref_/_br shorthand
    assert {
        "candidate_score_cutoff", "candidate_approved_count", "candidate_approval_rate",
        "v3a_approval_rate", "approval_rate_delta", "candidate_bad_rate", "v3a_bad_rate",
        "bad_rate_delta", "swap_in_bad_rate", "swap_out_bad_rate",
        "swap_in_count", "swap_out_count", "day1_approval_rate",
    } == set(uw)
    # delta identity holds
    assert uw["approval_rate_delta"] == (uw["candidate_approval_rate"] - uw["v3a_approval_rate"])
```

> If `make_synthetic_daily` does not yet exist in `fixtures.py`, add a thin wrapper there that returns the existing `_fixture_users_cached`-style daily frame (the same frame `test_analysis.py` builds via `make_synthetic_base_table` → daily). Reuse, do not duplicate, the fixture's daily construction.

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest projects/neobank_ncm/tests/test_decision_report.py::test_decision_report_structure_and_naming -v`
Expected: FAIL — `module 'report' has no attribute 'build_decision_report'`.

- [ ] **Step 3: Implement `build_decision_report`**

```python
# projects/neobank_ncm/analysis/report.py
"""Assemble the full decision report (all scenarios, settled vocabulary).

Pure function over a *scored* daily frame (``v3_score`` present) that also
carries the per-user LTV columns (broadcast). Reuses the policy/impact helpers;
this module owns the legacy→settled rename and the all-scenario sweep. See
docs/to-do/decision-metric-vocabulary.md for the contract.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from projects.neobank_ncm.analysis import impact, policy, scoring

# Family-3 legacy column -> settled name
_TRACK_RENAME = {
    "v3_thr": "candidate_score_cutoff",
    "n_v3": "candidate_approved_count",
    "v3_ar": "candidate_approval_rate",
    "ref_ar": "v3a_approval_rate",
    "delta_ar": "approval_rate_delta",
    "v3_br": "candidate_bad_rate",
    "ref_br": "v3a_bad_rate",
    "delta_br": "bad_rate_delta",
    "swap_in_br": "swap_in_bad_rate",
    "swap_out_br": "swap_out_bad_rate",
    "swap_in_vol": "swap_in_count",
    "swap_out_vol": "swap_out_count",
    "d1_ar": "day1_approval_rate",
}

# scenario id -> (settled key, ko_gate label, objective, all_threshold_tables key)
_SCENARIOS = {
    1: ("1_no_ko_match_bad_rate", "none", "match_bad_rate", "no_ko/match_br"),
    2: ("2_income500_match_bad_rate", "income500", "match_bad_rate", "income500/match_br"),
    3: ("3_v3a_ko_match_bad_rate", "v3a_ko", "match_bad_rate", "v3a_ko/match_br"),
    7: ("7_income500_broad_match_bad_rate", "income500_broad", "match_bad_rate", "income500_broad/match_br"),
    4: ("4_no_ko_match_approval_rate", "none", "match_approval_rate", "no_ko/match_ar"),
    5: ("5_income500_match_approval_rate", "income500", "match_approval_rate", "income500/match_ar"),
    6: ("6_v3a_ko_match_approval_rate", "v3a_ko", "match_approval_rate", "v3a_ko/match_ar"),
}

# UW / CLE row labels produced by policy.threshold_table
_UW_LABEL = "UW  (v2<=0.485 + KOs)"
_CLE_LABEL = "CLE (v2<=0.64 + inc>500)"


def _rename_track(row: pd.Series) -> dict:
    return {settled: _scalar(row[legacy]) for legacy, settled in _TRACK_RENAME.items()}


def _scalar(value) -> float | int:
    if isinstance(value, (np.integer,)):
        return int(value)
    return float(value)


def _scenario_ltv(daily: pd.DataFrame, users: pd.DataFrame, thresholds: dict, scenario_id: int) -> dict:
    """LTV-per-link over the combined UW∪CLE approved population (legacy cells 19–24)."""
    us_raw = impact.merge_ltv(users, _user_ltv_from_daily(daily))
    ref = impact.historical_reference(us_raw)
    us, lkp = impact.build_lookup(us_raw)
    _, thr_uw, thr_cle, ko_uw, ko_cle = policy.scenario_map(thresholds)[scenario_id]
    arr = policy.first_approval_days(daily, users["user_id"], thr_uw, thr_cle, ko_uw, ko_cle)
    uw_mask = us["user_id"].isin(set(users["user_id"][arr["uw"] <= 30]))
    cle_mask = us["user_id"].isin(set(users["user_id"][arr["cle"] <= 30])) & ~uw_mask
    lam = pd.Series(
        np.where(uw_mask, policy.LAM_UW, np.where(cle_mask, policy.LAM_CLE, np.nan)),
        index=us.index,
    )
    agg = impact.monthly_aggregate(
        impact.infer_financials(us, lkp, uw_mask | cle_mask, lam),
        len(us), ref["act_rate"], ref["monthly_vol"],
    )
    return {"ltv_per_link_d90": round(float(agg["lpl_90"]), 4),
            "ltv_per_link_d120": round(float(agg["lpl_120"]), 4)}


def _user_ltv_from_daily(daily: pd.DataFrame) -> pd.DataFrame:
    """Recover the user-grain LTV frame from the broadcast daily columns."""
    cols = (["user_id", "loan_amount_max", "underwriting_strategy", "first_activation_date"]
            + [f"total_revenue_{h}" for h in impact.HORIZONS]
            + [f"total_ltv_lite_{h}" for h in impact.HORIZONS]
            + [f"ltv_{h}_elig" for h in impact.HORIZONS])
    ltv = daily[cols].drop_duplicates("user_id").reset_index(drop=True)
    ltv["is_activated"] = ltv["first_activation_date"].notna()
    return ltv


def build_decision_report(daily: pd.DataFrame, *, headline_scenario: int = 2,
                          provenance: dict | None = None) -> dict:
    policy.add_policy_columns(daily)
    users = policy.collapse_to_users(daily)
    bench = policy.benchmarks(users)
    thresholds = policy.compute_thresholds(users, bench)
    tables = policy.all_threshold_tables(users, thresholds)
    auc = scoring.d2_known_auc(daily)

    scenarios = {}
    for scenario_id, (key, ko_gate, objective, table_key) in _SCENARIOS.items():
        table = tables[table_key]
        scenarios[key] = {
            "ko_gate": ko_gate,
            "objective": objective,
            **_scenario_ltv(daily, users, thresholds, scenario_id),
            "tracks": {
                "uw": _rename_track(table.loc[_UW_LABEL]),
                "cle": _rename_track(table.loc[_CLE_LABEL]),
            },
        }

    report = {
        "discrimination": {
            "day2_known_auc": round(float(auc["d2_auc"]), 5),
            "day2_known_count": int(auc["d2_n"]),
            "day2_known_bad_rate": round(float(auc["d2_bad_rate"]), 5),
        },
        "benchmark": {
            "v3a_approval_rate": round(float(bench["v3a_ar"]), 5),
            "v3a_bad_rate": round(float(bench["v3a_br"]), 5),
            "cle_approval_rate": round(float(bench["cle_ar"]), 5),
            "cle_bad_rate": round(float(bench["cle_br"]), 5),
        },
        "scenarios": scenarios,
    }
    if provenance is not None:
        report["provenance"] = {**provenance, "headline_scenario": _SCENARIOS[headline_scenario][0]}
    return report
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest projects/neobank_ncm/tests/test_decision_report.py::test_decision_report_structure_and_naming -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add projects/neobank_ncm/analysis/report.py projects/neobank_ncm/tests/test_decision_report.py projects/neobank_ncm/tests/fixtures.py
git commit -m "feat(neobank_ncm): build_decision_report — all scenarios, settled vocabulary"
```

---

## Task 2: project eval metrics (`Day2KnownAuc`, `DecisionReport`)

**Files:**
- Create: `projects/neobank_ncm/eval/__init__.py`, `projects/neobank_ncm/eval/metrics.py`
- Test: `projects/neobank_ncm/tests/test_decision_report.py` (extend)

- [ ] **Step 1: Write the failing test (extend the test file)**

```python
def test_decision_metrics_and_spec():
    from projects.neobank_ncm.eval import decision_eval_spec
    from projects.neobank_ncm.eval.metrics import DecisionReport, Day2KnownAuc

    daily = _scored_daily_with_ltv()
    y_pred = daily["v3_score"]

    auc_rec = Day2KnownAuc().evaluate(daily, y_pred, "went_dpd45")
    assert auc_rec["name"] == "day2_known_auc"
    assert isinstance(auc_rec["value"], float) and 0.0 <= auc_rec["value"] <= 1.0

    rep_rec = DecisionReport().evaluate(daily, y_pred, "went_dpd45")
    assert rep_rec["name"] == "decision_report"
    assert "scenarios" in rep_rec["value"]            # structured (non-scalar) survives

    spec = decision_eval_spec()
    assert spec.primary_name == "day2_known_auc"      # scalar primary
    out = spec.evaluate(daily, y_pred, "went_dpd45")
    assert out["primary"] == "day2_known_auc"
    names = {m["name"] for m in out["metrics"]}
    assert names == {"day2_known_auc", "decision_report"}
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest projects/neobank_ncm/tests/test_decision_report.py::test_decision_metrics_and_spec -v`
Expected: FAIL — `No module named 'projects.neobank_ncm.eval'`.

- [ ] **Step 3: Implement the metrics + spec**

```python
# projects/neobank_ncm/eval/__init__.py
from projects.neobank_ncm.eval.metrics import decision_eval_spec

__all__ = ["decision_eval_spec"]
```

```python
# projects/neobank_ncm/eval/metrics.py
"""Project decision metrics for the native re-eval (see decision-metric-vocabulary.md)."""
from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.metrics import roc_auc_score

from automl.eval import EvalSpec, Metric
from projects.neobank_ncm.analysis import report

# columns the metrics need present in the eval frame (validated by evaluate())
_REQUIRED = (
    "user_id", "day_number", "is_known", "synthetic_score", "v2_score",
    "account_approval_state", "dailyincomemean", "highestpaydepositmean",
    "noactivityrate", report.policy.PLAID_INFLOW_30D, "loan_amount_max",
    "underwriting_strategy", "first_activation_date",
)


class Day2KnownAuc(Metric):
    name = "day2_known_auc"
    required_columns = ("day_number", "is_known", "went_dpd45")

    def compute(self, df: pd.DataFrame, y_pred: Any, target_col: str) -> float:
        mask = (df["day_number"] == 2) & df["is_known"] & df[target_col].notna()
        return float(roc_auc_score(df.loc[mask, target_col].astype(int), pd.Series(y_pred)[mask]))


class DecisionReport(Metric):
    name = "decision_report"
    required_columns = _REQUIRED

    def __init__(self, *, headline_scenario: int = 2, provenance: dict | None = None) -> None:
        self._headline = headline_scenario
        self._provenance = provenance

    def compute(self, df: pd.DataFrame, y_pred: Any, target_col: str) -> dict:
        scored = df.copy()
        scored["v3_score"] = pd.Series(y_pred).to_numpy()
        return report.build_decision_report(
            scored, headline_scenario=self._headline, provenance=self._provenance
        )


def decision_eval_spec(*, headline_scenario: int = 2, provenance: dict | None = None) -> EvalSpec:
    return EvalSpec(
        primary=Day2KnownAuc(),
        metrics=[DecisionReport(headline_scenario=headline_scenario, provenance=provenance)],
    )
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest projects/neobank_ncm/tests/test_decision_report.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add projects/neobank_ncm/eval/
git commit -m "feat(neobank_ncm): Day2KnownAuc + DecisionReport eval metrics"
```

---

## Task 3: materialize the `oot_new_links_with_ltv` external eval dataset (one-time)

**Files:**
- Create: `projects/neobank_ncm/scripts/prepare_oot_new_links_dataset.py`

- [ ] **Step 1: Write the script**

```python
"""One-time materialization of the oot_new_links_with_ltv external EvalDataset.

Model-INDEPENDENT: the daily scoring frame + derived features + per-user LTV
(broadcast per daily row). Every trial's decision re-eval points at this one
dataset. Idempotent — re-running returns the existing id unless --overwrite.

Run OFF-VPN (writes ~GB to GCS). Cached frames are reused; --refresh re-pulls.

    uv run python projects/neobank_ncm/scripts/prepare_oot_new_links_dataset.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

CACHE = Path(".cache/automl/fin")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    from automl.eval import prepare_eval_dataset
    from automl.project import use_project
    from projects.neobank_ncm.analysis import data, impact, scoring

    session = use_project("neobank_ncm")

    daily = pd.read_parquet(CACHE / "daily.parquet")
    daily.columns = [c.lower() for c in daily.columns]
    daily["is_known"] = daily["went_dpd45"].notna()
    scoring.add_daily_derived_features(daily)          # model expects these as inputs

    ltv = pd.read_parquet(CACHE / "user_ltv.parquet")
    ltv.columns = [c.lower() for c in ltv.columns]
    ltv_cols = (["user_id", "loan_amount_max", "underwriting_strategy", "first_activation_date"]
                + [f"total_revenue_{h}" for h in impact.HORIZONS]
                + [f"total_ltv_lite_{h}" for h in impact.HORIZONS]
                + [f"ltv_{h}_elig" for h in impact.HORIZONS])
    frame = daily.merge(ltv[ltv_cols], on="user_id", how="left")   # broadcast LTV per daily row

    ds, existed = prepare_eval_dataset(
        session=session,
        kind="external",
        frame=frame,
        target_col="went_dpd45",
        unique_key=("user_id", "day_number"),
        provenance={"population": "oot_new_links_with_ltv", "ltv_pull_date": "2026-06-11"},
        overwrite=args.overwrite,
    )
    print(f"eval_dataset_id={ds.id}  (existed={existed})  rows={len(frame)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it (off-VPN; one-time)**

Run: `uv run python projects/neobank_ncm/scripts/prepare_oot_new_links_dataset.py`
Expected: prints `eval_dataset_id=<id>  (existed=False)  rows=5269592`. Record the id.

> If it OOMs on the GCS write or the merge, that is the [`eval-chunked-prediction.md`](eval-chunked-prediction.md) territory / a frame too large to hold — note it and stop; do not add a workaround.

- [ ] **Step 3: Commit**

```bash
git add projects/neobank_ncm/scripts/prepare_oot_new_links_dataset.py
git commit -m "feat(neobank_ncm): materialize oot_new_links_with_ltv external eval dataset"
```

---

## Task 4: rewrite `score_trial_financials.py` as a thin evaluate() driver

**Files:**
- Modify (rewrite): `projects/neobank_ncm/scripts/score_trial_financials.py`

- [ ] **Step 1: Replace the file**

```python
"""Decision/financial re-evaluation of trial model(s) on oot_new_links.

Routes through the harness eval flow: one evaluate() per trial against the
oot_new_links_with_ltv external dataset, with the project decision EvalSpec.
Records eval/oot_new_links/report.json + index + the day2_known_auc scalar
(never set as the run's primary label — decision metrics never drive selection).

    uv run python projects/neobank_ncm/scripts/score_trial_financials.py \
        --eval-dataset-id <id> --model-run-id <id> [--model-run-id <id> ...]
"""
from __future__ import annotations

import argparse
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # torch trials: avoid the OpenMP SIGSEGV


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-dataset-id", required=True)
    ap.add_argument("--model-run-id", action="append", required=True, dest="model_run_ids")
    args = ap.parse_args()

    from automl.eval import evaluate
    from automl.project import use_project
    from projects.neobank_ncm.eval import decision_eval_spec

    session = use_project("neobank_ncm")
    spec = decision_eval_spec()

    for run_id in args.model_run_ids:
        result = evaluate(
            session=session,
            model_run_id=run_id,
            eval_dataset_id=args.eval_dataset_id,
            label="oot_new_links",
            eval_spec=spec,
            set_as_primary_label=False,   # never a selection metric
            overwrite=True,
        )
        report = next(m["value"] for m in result.metrics if m["name"] == "decision_report")
        head = report["scenarios"]["2_income500_match_bad_rate"]["tracks"]["uw"]
        print(f"{run_id}  day2_known_auc={result.primary_value if hasattr(result,'primary_value') else '?'} "
              f"approval_rate_delta(sc2,uw)={head['approval_rate_delta']:.4f} "
              f"swap_in_bad_rate={head['swap_in_bad_rate']:.4f}")


if __name__ == "__main__":
    main()
```

> `result.primary_value` may not exist on `EvalResult`; read the scalar from `result.metrics` instead — find the record named `day2_known_auc` and print its `value`. Adjust the print line to: `auc = next(m["value"] for m in result.metrics if m["name"] == "day2_known_auc")`.

- [ ] **Step 2: Smoke-test on trial 1 (the validated baseline)**

Run: `uv run python projects/neobank_ncm/scripts/score_trial_financials.py --eval-dataset-id <id> --model-run-id 51bd38d4bcb845cbbad52dcacd637e1e`
Expected: prints a line with `approval_rate_delta(sc2,uw)` ≈ **0.043** (the validated corrected trial-1 ΔAR from the handoff). If it reproduces ~0.043, the native path is correct.

- [ ] **Step 3: Commit**

```bash
git add projects/neobank_ncm/scripts/score_trial_financials.py
git commit -m "refactor(neobank_ncm): score_trial_financials -> native evaluate() driver"
```

---

## Task 5: rebuild the cross-trial comparison + re-validate the finding

**Files:**
- Create: `projects/neobank_ncm/scripts/rebuild_decision_comparison.py`

- [ ] **Step 1: Write the comparison rebuild**

```python
"""Assemble the cross-trial decision comparison from each trial's report.json.

Reads eval/oot_new_links/report.json off each trial run (via the artifacts
seam), pulls the headline scenario-2 UW numbers + day2_known_auc, and writes
.cache/automl/fin/decision_comparison.parquet with the settled column names.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

TRIALS = {  # trial -> run_id (from docs/HANDOFF.md; trial 7 excluded — not deployable)
    1: "51bd38d4bcb845cbbad52dcacd637e1e",
    3: "2f39e0ead13d4a588e4a385f272dc38f",
    11: "b3e5efdb9a924157b4ca521022ccf816",
    12: "9c6b2e176e9f45888d7489be3e38aedc",
    13: "e4ea3b7256924bdc83e942e79bb85715",
}


def main() -> None:
    from automl.mlflow.trial import artifacts
    from automl.project import use_project

    use_project("neobank_ncm")
    rows = []
    for trial, run_id in TRIALS.items():
        result = artifacts.load_eval(run_id, "oot_new_links")
        rep = next(m["value"] for m in result.metrics if m["name"] == "decision_report")
        auc = next(m["value"] for m in result.metrics if m["name"] == "day2_known_auc")
        uw = rep["scenarios"]["2_income500_match_bad_rate"]["tracks"]["uw"]
        sc = rep["scenarios"]["2_income500_match_bad_rate"]
        rows.append({
            "trial": trial,
            "day2_known_auc": auc,
            "candidate_approval_rate": uw["candidate_approval_rate"],
            "approval_rate_delta": uw["approval_rate_delta"],
            "swap_in_bad_rate": uw["swap_in_bad_rate"],
            "ltv_per_link_d90": sc["ltv_per_link_d90"],
            "ltv_per_link_d120": sc["ltv_per_link_d120"],
        })
    df = pd.DataFrame(rows).sort_values("trial").reset_index(drop=True)
    out = Path(".cache/automl/fin/decision_comparison.parquet")
    df.to_parquet(out)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the full re-validation**

```bash
uv run python projects/neobank_ncm/scripts/score_trial_financials.py --eval-dataset-id <id> \
  --model-run-id 51bd38d4bcb845cbbad52dcacd637e1e \
  --model-run-id 2f39e0ead13d4a588e4a385f272dc38f \
  --model-run-id b3e5efdb9a924157b4ca521022ccf816 \
  --model-run-id 9c6b2e176e9f45888d7489be3e38aedc \
  --model-run-id e4ea3b7256924bdc83e942e79bb85715
uv run python projects/neobank_ncm/scripts/rebuild_decision_comparison.py
```
Expected: a 5-row table with corrected `approval_rate_delta` per trial. Compare to the OLD no-KO numbers in the draft learning (MLP +7.4, GAM +3.7) — the corrected scenario-2 ΔAR will differ.

- [ ] **Step 3: Update the draft learning with corrected numbers**

Edit `.cache/automl/learnings/auc-vs-business-divergence.md`: replace the ΔAR / swap-in / LPL columns with the corrected scenario-2 numbers; re-write findings 1–2 based on whether the divergence survives. Keep it DRAFT until confirmed.

- [ ] **Step 4: Commit**

```bash
git add projects/neobank_ncm/scripts/rebuild_decision_comparison.py
git commit -m "feat(neobank_ncm): rebuild decision comparison from native report.json"
```

---

## Task 6: align the vocabulary doc with the as-built design

**Files:**
- Modify: `docs/to-do/decision-metric-vocabulary.md`

- [ ] **Step 1: Update the recording section**

Edit `decision-metric-vocabulary.md`:
- Recording: state it routes through `evaluate(eval_spec=...)` against an **`external` EvalDataset** materialized once via `prepare_eval_dataset` (model-independent; LTV broadcast per daily row); cross-link this plan and `eval-chunked-prediction.md`.
- Move **`ltv_per_link_d90` / `ltv_per_link_d120` to the scenario level** (combined UW∪CLE), not inside the track records — fixing the report.json shape to match the legacy-aligned build.
- Note the accepted prediction-RAM risk + the chunked-prediction to-do.

- [ ] **Step 2: Commit**

```bash
git add docs/to-do/decision-metric-vocabulary.md docs/to-do/native-decision-reeval-plan.md docs/to-do/eval-chunked-prediction.md
git commit -m "docs(neobank_ncm): align decision-metric vocabulary with as-built native re-eval"
```

---

## Self-review notes (already folded in)

- **Spec coverage:** populations (Task 3), scenario keys + all-7 (Task 1), settled names (Task 1), Design-B recording / no selection primary (Tasks 2, 4), `report.json` shape (Task 1), re-validation (Task 5), doc alignment (Task 6). The deferred scalar-promotion stays deferred — only `day2_known_auc` is logged, by construction.
- **LTV grain:** corrected to per-scenario (legacy-aligned), reflected in Task 1 code, the test, and the Task 6 doc fix.
- **Prediction RAM:** explicitly NOT worked around (no `_model` hook); the core fix is `eval-chunked-prediction.md`.
- **Open verification during execution:** confirm `EvalResult` field names when printing the primary (Task 4 note); confirm `make_synthetic_daily` exists or add the thin fixture wrapper (Task 1 note).
