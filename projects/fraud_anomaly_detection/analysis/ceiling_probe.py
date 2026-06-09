"""Supervised information-ceiling probe on the scenario-gated residual.

Discovery diagnostic, not a production model. Round-3's unsupervised Isolation
Forest scored ~0.075 AP (~1.3x base) on the gated residual and only re-ranked
the heuristic's own velocity band. The question this answers: is that the
*algorithm's* ceiling or the *feature space's* ceiling?

A gradient-boosted classifier is forced (supervised) to predict the DPD45
outcome on the exact gated-residual rows IF scored. If it also tops out near
IF, the feature space is exhausted and the lever is new data (device/IP,
monetization speed). If it pulls meaningfully ahead, there is structure IF's
geometry misses and a different unsupervised lens is worth a trial. The feature
importances map *what* carries any residual signal (known newness/speed aliases
vs. a new axis).

Read it through top-k% precision and lift vs. base rate — not cross-run
comparison. Features are used AS-IS (no cleanup): HistGBM is scale-invariant,
handles NaNs natively, and is robust to the redundant aliases, so this measures
the information content, not preprocessing choices. DPD45 in the residual is
mostly credit risk now that the ring is rule-handled, so treat the level as a
ceiling on *learnable outcome structure*, and read the importances for whether
it is the axis we already know.

    uv run python -m projects.fraud_anomaly_detection.analysis.ceiling_probe
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

DATASET_ID = "v1_42baf0ba"
DEPTHS = (0.005, 0.01, 0.02, 0.05)
IF_TEST_AP = 0.0751  # round-3 iforest_residual_baseline, for reference


def _load_env() -> None:
    env_path = Path(".env")
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _depth_table(y_true: np.ndarray, scores: np.ndarray, base: float) -> list[dict]:
    n = len(scores)
    order = np.argsort(-scores, kind="stable")
    total_pos = int(y_true.sum())
    rows = []
    for d in DEPTHS:
        k = max(1, int(round(n * d)))
        idx = order[:k]
        tp = int(y_true[idx].sum())
        prec = tp / k
        rows.append(
            {
                "depth": d,
                "n_reviewed": k,
                "true_positives": tp,
                "precision": prec,
                "lift_vs_base": (prec / base) if base else None,
                "recall": (tp / total_pos) if total_pos else None,
            }
        )
    return rows


def main() -> None:
    _load_env()
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import average_precision_score

    from automl.data.registry import load_dataset_by_id
    from automl.project.session import use_project
    from projects.fraud_anomaly_detection.scenarios import residual_mask

    sess = use_project("fraud_anomaly_detection", dry_run=True)
    df = load_dataset_by_id(DATASET_ID, session=sess).df
    registry = load_dataset_by_id(DATASET_ID, session=sess).registry

    # Same feature space IF saw: numeric + bool registry features.
    num_cols = registry.get_by_dtype("num", flag="feature")
    bool_cols = registry.get_by_dtype("bool", flag="feature")
    feature_cols = sorted(set(num_cols) | set(bool_cols))

    residual = residual_mask(df).to_numpy()
    mature = (df["label_mature_d45"].astype(float) == 1).to_numpy()
    dpd45 = (df["label_gross_dpd45"].astype(float) == 1).to_numpy()
    never = dpd45 & (df["label_repaid_current_snapshot"].astype(float) == 0).to_numpy()
    split = df["SPLIT_PCT"].astype(float).to_numpy()  # group-safe (hash of user_id)

    keep_train = residual & mature & (split < 80)
    keep_test = residual & mature & (split >= 80)

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").astype("float64")
    Xtr, ytr = X[keep_train].to_numpy(), dpd45[keep_train].astype(int)
    Xte = X[keep_test].to_numpy()
    yte_dpd45 = dpd45[keep_test].astype(int)
    yte_never = never[keep_test].astype(int)

    base = yte_dpd45.mean()
    print(f"features: {len(feature_cols)} (num+bool, as-is)")
    print(f"train residual-mature: {keep_train.sum()} rows, {ytr.sum()} DPD45 positives")
    print(f"test  residual-mature: {keep_test.sum()} rows, {yte_dpd45.sum()} DPD45 positives")
    print(f"test DPD45 base rate: {base:.4%}\n")

    clf = HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=400, early_stopping=True,
        validation_fraction=0.1, l2_regularization=1.0, random_state=0,
    )
    clf.fit(Xtr, ytr)
    scores = clf.predict_proba(Xte)[:, 1]

    ap_dpd45 = average_precision_score(yte_dpd45, scores)
    ap_never = average_precision_score(yte_never, scores)
    print("=== CEILING (supervised GBM on test residual-mature) ===")
    print(f"  DPD45 AP      : {ap_dpd45:.4f}   (IF round-3: {IF_TEST_AP:.4f} | base {base:.4f} | "
          f"GBM lift vs base {ap_dpd45/base:.2f}x, vs IF {ap_dpd45/IF_TEST_AP:.2f}x)")
    print(f"  never-paid AP : {ap_never:.4f}\n")

    print("=== depth precision / lift vs base (DPD45) ===")
    print(f"  {'depth':>6} {'reviewed':>9} {'TP':>5} {'precision':>10} {'lift':>7} {'recall':>8}")
    for r in _depth_table(yte_dpd45, scores, base):
        print(f"  {r['depth']:>6.3f} {r['n_reviewed']:>9} {r['true_positives']:>5} "
              f"{r['precision']:>9.2%} {r['lift_vs_base']:>6.2f}x {r['recall']:>7.2%}")

    print("\n=== permutation importance (held-out test, scoring=AP, top 18) ===")
    imp = permutation_importance(
        clf, Xte, yte_dpd45, scoring="average_precision",
        n_repeats=5, random_state=0, n_jobs=-1,
    )
    rank = np.argsort(-imp.importances_mean)[:18]
    for i in rank:
        print(f"  {feature_cols[i]:<42} {imp.importances_mean[i]:+.4f} ± {imp.importances_std[i]:.4f}")


if __name__ == "__main__":
    main()
