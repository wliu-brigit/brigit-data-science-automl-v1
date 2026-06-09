"""Unsupervised discovery on the v2 feature base (dataset v1_76d3ad45).

Round-3 closed the unsupervised sweep as NEGATIVE on the gated residual: on the
OLD feature space (107k dry-run, v1_42baf0ba) Isolation Forest / GMM / autoencoder
all sat at ~base rate and only re-ranked the heuristic's own velocity band. This
re-opens the question on the NEW feature space — the v2 build added scarce-resource
sharing edges (device / persistent-id / address / phone / email, all as-of) plus
name-match and the neobank flag. Those edges are exactly the "abnormal shape"
(a value shared by >=2 fresh identities is a sub-0.1% anomaly), so the honest
test is: now that they exist as features, does an anomaly model on the gated
residual surface fraud-shaped rows the registered rules don't yet catch?

Why unsupervised (per wendao): a supervised target on DPD45 blends genuine fraud
with normal-but-defaulted credit risk; an anomaly score keys on SHAPE, which is
closer to fraud intent. So we read the cohort, not a single AP number.

Frame (mirrors ceiling_probe / supervised_lens for comparability):
  - population: scenario-gated residual (the two bank-account rings removed)
  - fit: residual train rows (SPLIT_PCT < 80); IF/GMM are unsupervised so labels
    are not used to fit and maturity is not required for the geometry
  - eval: residual + mature (label_mature_d45 == 1) test rows (SPLIT_PCT >= 80)
  - score read: AP / depth-precision vs DPD45 and never-paid, vs residual base
  - discovery read: top-cohort never-paid rate, heuristic-band mix (LOW = missed),
    NEW-EDGE prevalence (cohort vs base), and per-feature shape (cohort vs base)

Three runs:
  A. IF on the full feature space (the direct round-3 re-open)
  B. IF withholding the bank-account ring family — isolates whether the NEW
     edges + other features carry independent discovery leverage (the parked
     "withhold experiment")
  C. GMM (density) on the full space — a second geometry, robustness check

Read-only, pinned snapshot, no warehouse. Discovery diagnostic, not a model.

    uv run python -m projects.fraud_anomaly_detection.analysis.unsupervised_lens
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

DATASET_ID = "v1_76d3ad45"
DEPTHS = (0.005, 0.01, 0.02, 0.05)
COHORTS = (0.005, 0.01, 0.02)
IF_ROUND3_AP = 0.0751  # round-3 iforest_residual_baseline (OLD features, 107k) — directional only

# Excluded from the feature space here because the stored registry (baked at
# materialize time) still flags it as a feature. name_match_official is noise
# (account product type, not holder name) — also now in config.exclude_cols.
DROP_FEATURES = {"name_match_official"}

# The bank-account ring family — withheld in run B to test whether the new
# edges carry leverage independent of the signal the heuristic already owns.
RING_FAMILY = {
    "users_on_bank_account_72h", "users_on_bank_account_7d", "users_on_bank_account_30d",
    "users_on_bank_account_90d", "users_on_bank_account_lifetime_asof",
    "flag_3_users_on_bank_account_72h", "flag_5_users_on_bank_account_ever_asof",
    "flag_10_users_on_bank_account_ever_asof",
    "avg_users_created_per_day_asof", "avg_users_created_per_month_asof",
    "user_creation_days_span_asof",
    "network_user_count_asof", "network_account_count_asof", "network_score_asof",
    "prior_advances_on_bank_account_24h", "prior_advances_on_bank_account_72h",
    "prior_advances_on_bank_account_7d", "prior_advances_on_bank_account_30d",
    "prior_advances_on_bank_account_lifetime",
}

# New-edge / candidate-scenario indicators tabulated in every cohort.
EDGE_PREDICATES = {
    "device_7d>=2":       ("users_on_device_id_7d", ">=", 2),
    "persistent_7d>=2":   ("users_on_persistent_account_id_7d", ">=", 2),
    "address_7d>=2":      ("users_on_address_7d", ">=", 2),
    "phone_7d>=2":        ("users_on_phone_7d", ">=", 2),
    "email_7d>=2":        ("users_on_email_7d", ">=", 2),
    "bankacct_7d>=2":     ("users_on_bank_account_7d", ">=", 2),  # sub-ring-threshold reuse
    "name_last<80":       ("name_match_last", "<", 80),
    "neobank":            ("is_neobank_high_risk_institution", "==", 1),
}

DRIVERS = [
    "users_on_device_id_7d", "users_on_persistent_account_id_7d",
    "users_on_address_7d", "users_on_phone_7d", "users_on_email_7d",
    "users_on_bank_account_7d", "name_match_last", "name_match_first",
    "avg_prior_advances_per_day", "loan_amount", "total_disbursed",
    "hours_since_identity_created", "days_since_plaid_account_created",
    "prior_min_hours_between_advances_on_account", "is_neobank_high_risk_institution",
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


def _edge_mask(df: pd.DataFrame, col: str, op: str, val: float) -> np.ndarray:
    if col not in df.columns:
        return np.zeros(len(df), dtype=bool)
    s = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    if op == ">=":
        return s >= val
    if op == "<":
        return s < val
    if op == "==":
        return s == val
    raise ValueError(op)


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
        rows.append({
            "depth": d, "n": k, "tp": tp, "precision": prec,
            "lift": (prec / base) if base else float("nan"),
            "recall": (tp / total_pos) if total_pos else float("nan"),
        })
    return rows


def _fit_score(name: str, feats: list[str], df: pd.DataFrame,
               tr: np.ndarray, te: np.ndarray, geometry: str) -> np.ndarray:
    """Fit an unsupervised geometry on train rows, return anomaly score for test
    rows (higher = more anomalous). NaN/inf imputed by train median."""
    from sklearn.impute import SimpleImputer

    X = df[feats].apply(pd.to_numeric, errors="coerce").to_numpy(dtype="float64")
    X[~np.isfinite(X)] = np.nan
    imp = SimpleImputer(strategy="median").fit(X[tr])
    Xtr, Xte = imp.transform(X[tr]), imp.transform(X[te])

    if geometry == "iforest":
        from sklearn.ensemble import IsolationForest
        clf = IsolationForest(n_estimators=300, max_samples=min(256, Xtr.shape[0]),
                              contamination="auto", random_state=0, n_jobs=-1)
        clf.fit(Xtr)
        return -clf.score_samples(Xte)  # higher = more anomalous
    if geometry == "gmm":
        from sklearn.mixture import GaussianMixture
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(Xtr)
        Ztr, Zte = scaler.transform(Xtr), scaler.transform(Xte)
        # subsample the GMM fit for speed; score all test rows
        rng = np.random.RandomState(0)
        if Ztr.shape[0] > 250_000:
            sel = rng.choice(Ztr.shape[0], 250_000, replace=False)
            Ztr = Ztr[sel]
        gmm = GaussianMixture(n_components=8, covariance_type="diag",
                              reg_covar=1e-3, random_state=0, max_iter=200)
        gmm.fit(Ztr)
        return -gmm.score_samples(Zte)  # neg log-likelihood: higher = rarer
    raise ValueError(geometry)


def _report(name: str, scores: np.ndarray, df: pd.DataFrame, te_idx: np.ndarray,
            dpd45: np.ndarray, never: np.ndarray, band: np.ndarray) -> None:
    yte_dpd = dpd45[te_idx].astype(int)
    yte_never = never[te_idx].astype(int)
    base_dpd = yte_dpd.mean()
    base_never = yte_never.mean()

    from sklearn.metrics import average_precision_score
    ap_dpd = average_precision_score(yte_dpd, scores) if yte_dpd.sum() else float("nan")
    ap_never = average_precision_score(yte_never, scores) if yte_never.sum() else float("nan")

    print(f"\n########## {name} ##########")
    print(f"test residual-mature: {len(te_idx):,} rows | base DPD45 {base_dpd:.3%} | base never-paid {base_never:.3%}")
    print(f"  anomaly-score AP vs DPD45     : {ap_dpd:.4f}   (round-3 IF ref {IF_ROUND3_AP:.4f}; base {base_dpd:.4f}; "
          f"lift {ap_dpd/base_dpd:.2f}x)")
    print(f"  anomaly-score AP vs never-paid: {ap_never:.4f}   (base {base_never:.4f}; lift {ap_never/base_never:.2f}x)")

    print("  depth precision / lift (DPD45):")
    print(f"    {'depth':>6} {'n':>7} {'tp':>5} {'prec':>8} {'lift':>7} {'recall':>8}")
    for r in _depth_table(yte_dpd, scores, base_dpd):
        print(f"    {r['depth']:>6.3f} {r['n']:>7} {r['tp']:>5} {r['precision']:>7.2%} "
              f"{r['lift']:>6.2f}x {r['recall']:>7.2%}")

    # local frame for the test rows, ordered by anomaly score
    order = np.argsort(-scores, kind="stable")
    sub = df.iloc[te_idx].reset_index(drop=True)
    never_te = never[te_idx]
    band_te = band[te_idx]
    n_te = len(te_idx)

    # base edge prevalence over the whole test residual-mature set
    base_edge = {k: _edge_mask(sub, *spec).mean() for k, spec in EDGE_PREDICATES.items()}
    base_med = {c: np.nanmedian(pd.to_numeric(sub[c], errors="coerce").to_numpy(float))
                for c in DRIVERS if c in sub.columns}

    for q in COHORTS:
        k = max(1, int(round(n_te * q)))
        cidx = order[:k]
        c_never = never_te[cidx].mean()
        bands = pd.Series(band_te[cidx]).value_counts().to_dict()
        low = bands.get("LOW", 0)
        print(f"\n  === top {q:.1%} anomalous (n={k}) ===")
        print(f"    never-paid rate : {c_never:.1%}  ({c_never/base_never:.1f}x base)")
        print(f"    band mix        : LOW={low} ({low/k:.0%})  " +
              "  ".join(f"{b}={bands[b]}" for b in ("POSSIBLE", "LIKELY", "EXTREMELY_LIKELY") if b in bands))
        low_idx = cidx[band_te[cidx] == "LOW"]
        if len(low_idx):
            print(f"    LOW-slice never-paid: {never_te[low_idx].mean():.1%} (n={len(low_idx)})  <- heuristic-missed fraud-shape")
        csub = sub.iloc[cidx]
        print("    new-edge prevalence (cohort vs residual base):")
        for kk, spec in EDGE_PREDICATES.items():
            cp = _edge_mask(csub, *spec).mean()
            bp = base_edge[kk]
            print(f"      {kk:<18} {cp:>7.2%}  vs {bp:>7.2%}  ({cp/bp:>5.1f}x)" if bp else
                  f"      {kk:<18} {cp:>7.2%}  vs {bp:>7.2%}")
        print("    driver medians (cohort vs residual base):")
        for c in DRIVERS:
            if c in sub.columns:
                cm = np.nanmedian(pd.to_numeric(csub[c], errors="coerce").to_numpy(float))
                print(f"      {c:<46} {cm:>12.4g}  vs {base_med[c]:>12.4g}")


def main() -> None:
    _load_env()
    from automl.data.registry import load_dataset_by_id
    from automl.project.session import use_project
    from projects.fraud_anomaly_detection.scenarios import residual_mask

    sess = use_project("fraud_anomaly_detection", dry_run=False)
    loaded = load_dataset_by_id(DATASET_ID, session=sess)
    df, registry = loaded.df, loaded.registry

    feats_all = sorted((set(registry.get_by_dtype("num", flag="feature"))
                        | set(registry.get_by_dtype("bool", flag="feature")))
                       - DROP_FEATURES)
    feats_withhold = [c for c in feats_all if c not in RING_FAMILY]

    res = residual_mask(df).to_numpy()
    mat = (df["label_mature_d45"].astype(float) == 1).to_numpy()
    dpd45 = (df["label_gross_dpd45"].astype(float) == 1).to_numpy()
    never = dpd45 & (df["label_repaid_current_snapshot"].astype(float) == 0).to_numpy()
    split = df["SPLIT_PCT"].astype(float).to_numpy()
    band = df["heuristic_fraud_band"].astype(str).to_numpy()

    tr = res & (split < 80)            # unsupervised fit: labels unused, maturity not required
    te = res & mat & (split >= 80)     # honest eval: held-out, mature
    te_idx = np.where(te)[0]

    print(f"dataset {DATASET_ID}: {len(df):,} rows | features: {len(feats_all)} (withhold variant: {len(feats_withhold)})")
    print(f"residual: {res.sum():,} | gated (ring) rows: {(~res).sum():,}")
    print(f"fit (residual, train): {tr.sum():,} rows | eval (residual+mature, test): {te.sum():,} rows")
    print(f"test positives: DPD45={dpd45[te].sum()}  never-paid={never[te].sum()}")

    s_if = _fit_score("A", feats_all, df, tr, te, "iforest")
    _report("RUN A — Isolation Forest, full feature space", s_if, df, te_idx, dpd45, never, band)

    s_if_wh = _fit_score("B", feats_withhold, df, tr, te, "iforest")
    _report("RUN B — Isolation Forest, ring family WITHHELD", s_if_wh, df, te_idx, dpd45, never, band)

    s_gmm = _fit_score("C", feats_all, df, tr, te, "gmm")
    _report("RUN C — GMM (density), full feature space", s_gmm, df, te_idx, dpd45, never, band)


if __name__ == "__main__":
    main()
