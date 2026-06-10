"""Supervised model as a discovery lens on the gated residual.

The unsupervised sweep (IF / GMM / autoencoder) all landed at ~base rate on
DPD45 in the residual, while a supervised GBM reaches ~0.16 AP / 5.9x at the
top 1%. So the GBM — not an anomaly score — is the instrument that actually
ranks residual risk. This script asks the discovery question the unsupervised
trials could not answer: among the GBM's highest-risk residual rows, is there a
fraud-shaped, generalizable cluster worth turning into a scenario, or is it
diffuse credit risk?

For the top-k% of GBM-scored test-residual-mature rows it reports:
  - never-paid rate (fraud-leaning) vs DPD45 rate (credit+fraud), vs residual base
  - heuristic band composition (LOW = genuinely heuristic-missed; the discovery
    win condition is fraud-shaped rows the heuristic called clean)
  - the driver-feature profile (median) vs the residual baseline, to see whether
    the cohort is a coherent velocity x amount x newness pattern

Read-only, pinned snapshot, no warehouse. Not a production model.

    uv run python -m projects.fraud_anomaly_detection.analysis.supervised_lens
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

DATASET_ID = "v1_42baf0ba"
COHORTS = (0.005, 0.01, 0.02)
DRIVERS = [
    "avg_prior_advances_per_day",
    "total_disbursed",
    "loan_amount",
    "hours_since_previous_advance_on_account",
    "prior_loan_amount_avg_30d",
    "days_since_plaid_account_created",
    "days_between_identity_and_bank_account_creation",
    "origination_hour",
]


def _load_env() -> None:
    p = Path(".env")
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def main() -> None:
    _load_env()
    from sklearn.ensemble import HistGradientBoostingClassifier

    from automl.data.registry import load_dataset_by_id
    from automl.project.session import use_project
    from projects.fraud_anomaly_detection.scenarios import residual_mask

    sess = use_project("fraud_anomaly_detection", dry_run=True)
    loaded = load_dataset_by_id(DATASET_ID, session=sess)
    df, registry = loaded.df, loaded.registry

    feats = sorted(set(registry.get_by_dtype("num", flag="feature"))
                   | set(registry.get_by_dtype("bool", flag="feature")))
    res = residual_mask(df).to_numpy()
    mat = (df["label_mature_d45"].astype(float) == 1).to_numpy()
    dpd45 = (df["label_gross_dpd45"].astype(float) == 1).to_numpy()
    never = dpd45 & (df["label_repaid_current_snapshot"].astype(float) == 0).to_numpy()
    split = df["SPLIT_PCT"].astype(float).to_numpy()
    band = df["heuristic_fraud_band"].astype(str).to_numpy()

    X = df[feats].apply(pd.to_numeric, errors="coerce").astype("float64").to_numpy()
    tr = res & mat & (split < 80)
    te = res & mat & (split >= 80)

    clf = HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=400, early_stopping=True,
        validation_fraction=0.1, l2_regularization=1.0, random_state=0,
    )
    clf.fit(X[tr], dpd45[tr].astype(int))
    scores = clf.predict_proba(X[te])[:, 1]

    te_idx = np.where(te)[0]
    order = te_idx[np.argsort(-scores)]
    n_te = len(te_idx)

    base_np = never[te].mean()
    base_dpd = dpd45[te].mean()
    print(f"test residual-mature: {n_te} rows | base never-paid {base_np:.2%} | base DPD45 {base_dpd:.2%}\n")

    # residual baseline driver medians
    base_med = {c: np.nanmedian(df[c].to_numpy(dtype=float)[te]) for c in DRIVERS if c in df.columns}

    for q in COHORTS:
        k = max(1, int(round(n_te * q)))
        cohort = order[:k]
        np_rate = never[cohort].mean()
        dpd_rate = dpd45[cohort].mean()
        bands = pd.Series(band[cohort]).value_counts().to_dict()
        print(f"=== top {q:.1%}  (n={k}) ===")
        print(f"  never-paid rate : {np_rate:.1%}  ({np_rate/base_np:.1f}x base)")
        print(f"  DPD45 rate      : {dpd_rate:.1%}  ({dpd_rate/base_dpd:.1f}x base)")
        low = bands.get("LOW", 0)
        print(f"  band mix        : LOW={low} ({low/k:.0%})  " +
              "  ".join(f"{b}={bands[b]}" for b in ("POSSIBLE", "LIKELY") if b in bands))
        # fraud-shape within the heuristic-missed (LOW) slice — the real discovery win
        lowmask = cohort[band[cohort] == "LOW"]
        if len(lowmask):
            print(f"  LOW-slice never-paid: {never[lowmask].mean():.1%}  (n={len(lowmask)})  <- heuristic-missed fraud-shape")
        print("  driver medians (cohort vs residual base):")
        for c in DRIVERS:
            if c in df.columns:
                cm = np.nanmedian(df[c].to_numpy(dtype=float)[cohort])
                print(f"    {c:<46} {cm:>12.3g}  vs {base_med[c]:>12.3g}")
        print()


if __name__ == "__main__":
    main()
