"""Evaluate a risk model over the OOT new-links daily snapshot.

The first half of the legacy financial impact pipeline as a repeatable
script: load the frozen D1–D30 daily snapshot, score it with the chosen
model, and log the QA/eval metrics to one MLflow run — coverage,
synthetic-score calibration (decile table + Brier), D2 known-only AUC,
score percentiles, and the RI-consistency parity stats. The full scenario /
financial analysis lives in notebooks/financial_impact_analysis.ipynb.

Model (exactly one):
    --legacy-artifacts DIR   parity mode — the production v3 artifacts
                             (legacy repo artifacts/ folder)
    --model-run-id ID        harness mode — an MLflow-logged trial model

Data: warehouse by default (VPN); offline pass --daily-parquet (and
optionally --ri-scores-parquet/--syn-oot-parquet for the parity QA).

Namespace: defaults to a dated qa/ namespace (sweepable). The real on-VPN
analysis run is keep-worthy — pass the project namespace explicitly with
--namespace "".

    uv run python projects/neobank_ncm/scripts/evaluate_new_links_daily.py \
        --legacy-artifacts ../data-science/models/underwriting/neobank/new_user/v3.0/artifacts
"""

from __future__ import annotations

import argparse
import json
import tempfile
from datetime import date
from pathlib import Path

from automl import mlflow as automl_mlflow
from automl.mlflow import experiment as mlflow_experiment
from automl.mlflow.client import log_artifact_file
from automl.project import use_project
from projects.neobank_ncm.analysis import data as analysis_data
from projects.neobank_ncm.analysis import scoring


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--legacy-artifacts", help="legacy artifacts dir (parity mode)")
    model_group.add_argument("--model-run-id", help="MLflow run id of a trial model")
    parser.add_argument("--daily-parquet", default=None, help="offline daily snapshot")
    parser.add_argument("--ri-scores-parquet", default=None)
    parser.add_argument("--syn-oot-parquet", default=None)
    parser.add_argument(
        "--namespace",
        default=f"qa/neobank-newlinks-eval-{date.today():%Y%m%d}",
        help='MLflow namespace; default is a sweepable qa/ one, "" = project default',
    )
    args = parser.parse_args()

    session = use_project("neobank_ncm", namespace=args.namespace)

    daily = analysis_data.load_daily(parquet=args.daily_parquet)
    if args.legacy_artifacts:
        model = scoring.LegacyArtifactsModel(args.legacy_artifacts)
        model_ref = f"legacy:{args.legacy_artifacts}"
    else:
        model = scoring.TrialModel(args.model_run_id)
        model_ref = f"trial:{args.model_run_id}"
    scoring.score_daily(daily, model)

    # ── metrics (legacy cells 6, 7, 10 + the RI-scoring QA cell) ─────────
    by_user = daily.groupby("user_id")["day_number"].count()
    known_users = daily.loc[daily["is_known"], "user_id"].nunique()
    metrics: dict[str, float] = {
        "n_rows": float(len(daily)),
        "n_users": float(by_user.size),
        "n_users_known": float(known_users),
        "n_users_unknown": float(by_user.size - known_users),
        "avg_days_per_user": float(by_user.mean()),
        "day_number_min": float(daily["day_number"].min()),
        "day_number_max": float(daily["day_number"].max()),
        "v3_score_min": float(daily["v3_score"].min()),
        "v3_score_max": float(daily["v3_score"].max()),
        "v3_score_p50": float(daily["v3_score"].quantile(0.50)),
        "v3_score_p90": float(daily["v3_score"].quantile(0.90)),
    }
    metrics.update(scoring.d2_known_auc(daily))

    calibration = scoring.calibration_table(daily)
    d2_known = daily[
        daily["is_known"] & daily["synthetic_score"].notna() & (daily["day_number"] == 2)
    ]
    metrics["calibration_overall_syn"] = float(d2_known["synthetic_score"].mean())
    metrics["calibration_overall_br"] = float(d2_known["went_dpd45"].astype(float).mean())
    metrics["calibration_brier"] = float(
        ((d2_known["synthetic_score"] - d2_known["went_dpd45"].astype(float)) ** 2).mean()
    )

    parity: dict[str, float] | None = None
    try:
        ri_scores = analysis_data.load_ri_scores(parquet=args.ri_scores_parquet)
        syn_oot = analysis_data.load_synthetic_scores_oot(parquet=args.syn_oot_parquet)
        parity = analysis_data.ri_scores_parity(ri_scores, syn_oot)
        metrics.update({f"ri_parity_{k}": float(v) for k, v in parity.items()})
    except EnvironmentError as exc:
        print(f"RI parity QA skipped: {exc}")

    # ── one MLflow run under the session's experiment ─────────────────────
    with automl_mlflow.bound_for(session, experiment_id=session.active_experiment_id):
        mlflow_experiment.ensure()
        client = automl_mlflow.raw()
        run = client.create_run(
            mlflow_experiment.mlflow_experiment_id(),
            run_name=f"new_links_daily_eval-{date.today():%Y%m%d}",
        )
        run_id = run.info.run_id
        client.log_param(run_id, "model", model_ref)
        client.log_param(run_id, "data", args.daily_parquet or "warehouse")
        for key, value in metrics.items():
            client.log_metric(run_id, key, value)
        with tempfile.TemporaryDirectory(prefix="newlinks_eval_") as tmp:
            cal_path = Path(tmp) / "calibration_deciles.csv"
            calibration.to_csv(cal_path, index=False)
            log_artifact_file(run_id, "analysis/calibration_deciles.csv", cal_path)
        client.set_terminated(run_id)
        run_url = automl_mlflow.run_url(run_id)

    print(
        json.dumps(
            {
                "run_id": run_id,
                "run_url": run_url,
                "namespace": args.namespace,
                "model": model_ref,
                "metrics": metrics,
                "ri_parity": parity,
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
