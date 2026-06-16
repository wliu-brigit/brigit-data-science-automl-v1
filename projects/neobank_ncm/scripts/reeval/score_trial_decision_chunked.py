"""Decision/financial re-eval for memory-heavy models, via CHUNKED scoring.

SHIM — works around a core gap; retire when core `evaluate()` can predict in
chunks. The native evaluate() path predicts the whole 5.3M-row frame at once,
which thrashes swap for non-tree models (the spline GAM, the torch MLP). This
script is the fallback chosen for those trials: it scores with the proven
chunked scorer (scoring.score_daily → TrialModel, 250k-row chunks) over the
LOCAL cached frames (no 2 GB GCS read), builds the same decision report via the
project decision EvalSpec, and records it onto the run in the SAME eval-artifact
format evaluate() uses — so any cross-trial reader loads these trials
identically to the XGB ones. Output is identical to score_trial_financials.py;
only the prediction path differs.

When core grows chunked prediction (docs/to-do/eval-chunked-prediction.md),
delete this script and route these trials through score_trial_financials.py.

    uv run python projects/neobank_ncm/scripts/reeval/score_trial_decision_chunked.py \
        --eval-dataset-id <id> --model-run-id <id> [--model-run-id <id> ...]
"""
from __future__ import annotations

import argparse
import os
from datetime import UTC, datetime
from pathlib import Path

# torch trials: single-thread BLAS/OpenMP before numpy/torch import (SIGSEGV guard)
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

CACHE = Path(".cache/automl/fin")
PRED_KEY = ["day_number", "user_id"]  # predictions unique key (parity with native evaluate())


def _load_env() -> None:
    envp = Path(".env")
    if not envp.exists():
        return
    for line in envp.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            if k.strip().isidentifier():
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _build_local_frame():
    """The same oot_new_links_with_ltv frame, assembled from the local cache.

    Raw daily (derived features are added by score_daily at scoring time) + LTV
    broadcast per user; daily's duplicate loan_amount_max dropped so the LTV one
    survives (matches prepare_oot_new_links_dataset.py).
    """
    import pandas as pd

    from projects.neobank_ncm.analysis import impact

    daily = pd.read_parquet(CACHE / "daily.parquet")
    daily.columns = [c.lower() for c in daily.columns]
    daily["is_known"] = daily["went_dpd45"].notna()

    ltv = pd.read_parquet(CACHE / "user_ltv.parquet")
    ltv.columns = [c.lower() for c in ltv.columns]
    ltv_cols = (
        ["user_id", "loan_amount_max", "underwriting_strategy", "first_activation_date"]
        + [f"total_revenue_{h}" for h in impact.HORIZONS]
        + [f"total_ltv_lite_{h}" for h in impact.HORIZONS]
        + [f"ltv_{h}_elig" for h in impact.HORIZONS]
    )
    daily = daily.drop(columns=["loan_amount_max"], errors="ignore")
    return daily.merge(ltv[ltv_cols], on="user_id", how="left")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-dataset-id", required=True,
                    help="provenance stamp on the recorded eval (the frame is built from local cache)")
    ap.add_argument("--model-run-id", action="append", required=True, dest="model_run_ids")
    args = ap.parse_args()

    _load_env()

    from automl.eval.results import EvalResult, Predictions
    from automl.mlflow import trial as mlflow_trial
    from automl.mlflow.trial import artifacts
    from automl.project import use_project
    from projects.neobank_ncm.analysis import scoring
    from projects.neobank_ncm.eval import decision_eval_spec

    use_project("neobank_ncm")
    spec = decision_eval_spec()

    for run_id in args.model_run_ids:
        frame = _build_local_frame()                       # fresh frame per trial (freed after)
        model = scoring.TrialModel(run_id)
        try:
            import torch

            torch.set_num_threads(1)
        except ImportError:
            pass
        scoring.score_daily(frame, model)                  # chunked at 250k -> bounded memory
        evaluated = spec.evaluate(frame, frame["v3_score"], "went_dpd45")
        # persist raw daily predictions so reruns/blends reuse them (parity with native evaluate())
        pred_frame = frame[PRED_KEY].copy()
        pred_frame["y_pred"] = frame["v3_score"].to_numpy()
        pref = artifacts.write_predictions(
            run_id, "oot_new_links",
            Predictions(
                trial_run_id=run_id, eval_dataset_id=args.eval_dataset_id,
                eval_dataset_kind="external", label="oot_new_links",
                unique_key=tuple(PRED_KEY), frame=pred_frame, augmentations_used=(),
                written_at=datetime.now(UTC).isoformat(),
            ),
            overwrite=True,
        )
        del pred_frame
        result = EvalResult(
            label="oot_new_links",
            eval_dataset_id=args.eval_dataset_id,
            eval_dataset_kind="external",
            predictions_uri=pref.uri,
            predictions_manifest_uri=pref.manifest_uri,
            augmentations_used=(),
            primary=str(evaluated["primary"]),
            metrics=evaluated["metrics"],
            computed_at=datetime.now(UTC).isoformat(),
        )
        artifacts.write_eval(run_id, "oot_new_links", result, overwrite=True)
        metrics = {m["name"]: m["value"] for m in result.metrics}
        mlflow_trial.log_metrics(run_id, {"eval.oot_new_links.day2_known_auc": metrics["day2_known_auc"]})
        uw = metrics["decision_report"]["scenarios"]["2_income500_match_bad_rate"]["tracks"]["uw"]
        print(
            f"{run_id}  day2_known_auc={metrics['day2_known_auc']:.5f}  "
            f"approval_rate_delta(sc2,uw)={uw['approval_rate_delta']:+.4f}  "
            f"swap_in_bad_rate={uw['swap_in_bad_rate']:.4f}  (chunked, recorded)"
        )
        del frame, model


if __name__ == "__main__":
    main()
