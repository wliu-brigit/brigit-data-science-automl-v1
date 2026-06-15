"""Subgroup discovery (beam search) on the gated residual (dataset v1_76d3ad45).

The "proven algorithm" answer to wendao's question — how to find the best
conjunctive permutations of features, rigorously, instead of a greedy tree.

Subgroup discovery is the established method (Klosgen/Wrobel; pysubgroup is the
standard library, not installed here, so this is a self-contained, auditable
implementation). It enumerates SELECTORS (feature op threshold), then BEAM
SEARCHES conjunctions of them, keeping the top-W partial rules by quality at each
depth and extending them — so it explores cross-feature combinations the greedy
tree's first-split ordering structurally cannot reach.

Rigor (what makes it "proven", not data-dredging):
  - DISCOVER on train, VALIDATE every reported subgroup on held-out TEST. A
    conjunction that only looks good in-sample is exposed by its test precision.
  - minimum support floor (no 3-row "100%" pockets).
  - report candidates_evaluated + a one-sided binomial p-value of the TEST
    precision vs base, so search-inflated luck is visible (eyeball Bonferroni:
    divide your alpha by candidates_evaluated).

Target: never-paid (fraud-leaning, DPD45 & not repaid); mature rows only
(supervised needs matured labels). DPD45 reported alongside for the fraud smell.

    uv run python -m projects.fraud_anomaly_detection.analysis.subgroup_discovery \
        [--depth 3] [--beam 80] [--min-support 80] [--min-test 30] [--top 20]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

# Count-edges get exact integer selectors; continuous features get quantile
# thresholds in both directions; binaries get ==1.
EDGE_COLS = [
    "users_on_device_id_72h", "users_on_device_id_7d",
    "users_on_persistent_account_id_72h", "users_on_persistent_account_id_7d",
    "users_on_phone_72h", "users_on_email_7d", "users_on_address_72h",
    "users_on_address_7d", "users_on_bank_account_7d",
]
CONT_COLS = [
    "hours_since_identity_created", "days_since_plaid_account_created",
    "days_between_identity_and_bank_account_creation", "loan_amount",
    "total_disbursed", "avg_prior_advances_per_day",
    "prior_min_hours_between_advances_on_account",
    "prior_advances_on_bank_account_lifetime", "name_match_last",
    "bank_accounts_per_user_asof", "hours_since_socure_created",
]
BIN_COLS = ["is_neobank_high_risk_institution", "has_kyc",
            "signup_ip_matches_latest_ip", "is_joint"]


def _load_env() -> None:
    p = Path(".env")
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def build_selectors(sub: pd.DataFrame) -> list[tuple[str, np.ndarray]]:
    sels: list[tuple[str, np.ndarray]] = []
    def num(c):
        return pd.to_numeric(sub[c], errors="coerce").to_numpy(float)
    for c in EDGE_COLS:
        if c in sub.columns:
            v = num(c)
            for t in (2, 3):
                sels.append((f"{c} >= {t}", v >= t))
    for c in CONT_COLS:
        if c in sub.columns:
            v = num(c)
            qs = np.nanquantile(v, [0.33, 0.5, 0.67])
            for q in np.unique(np.round(qs, 4)):
                sels.append((f"{c} <= {q:.4g}", v <= q))
                sels.append((f"{c} > {q:.4g}", v > q))
    for c in BIN_COLS:
        if c in sub.columns:
            v = num(c)
            sels.append((f"{c} == 1", v == 1))
    return sels


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--beam", type=int, default=80)
    ap.add_argument("--min-support", type=int, default=80, help="min TRAIN rows for a subgroup")
    ap.add_argument("--min-test", type=int, default=30, help="min TEST rows to report")
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    _load_env()
    from automl.data.registry import load_dataset_by_id
    from automl.project.session import use_project
    from projects.fraud_anomaly_detection.scenarios import SCENARIOS, residual_mask

    sess = use_project("fraud_anomaly_detection", dry_run=False)
    df = load_dataset_by_id("v1_76d3ad45", session=sess).df
    print(f"gated out: {[s.name for s in SCENARIOS]}")

    res = residual_mask(df).to_numpy()
    mat = (df["label_mature_d45"].astype(float) == 1).to_numpy()
    dpd = (df["label_gross_dpd45"].astype(float) == 1).to_numpy()
    never = (dpd & (df["label_repaid_current_snapshot"].astype(float) == 0).to_numpy())
    split = pd.to_numeric(df["SPLIT_PCT"], errors="coerce").to_numpy(float)
    keep = res & mat
    sub = df[keep].reset_index(drop=True)
    y = never[keep].astype(int)
    dpd_k = dpd[keep].astype(int)
    tr = (split[keep] < 80)
    te = (split[keep] >= 80)

    base_tr = y[tr].mean()
    base_te = y[te].mean()
    print(f"post-lock residual+mature: {keep.sum():,} | train {tr.sum():,} (base {base_tr:.3%}) "
          f"| test {te.sum():,} (base {base_te:.3%})")

    sels = build_selectors(sub)
    print(f"selectors: {len(sels)} | beam search depth {args.depth}, width {args.beam}, "
          f"min train support {args.min_support}\n")

    from projects.fraud_anomaly_detection.analysis.subgroup_core import (
        beam_search,
        validate_rules,
    )

    ytr, yte = y[tr], y[te]
    sel_tr = [(name, m[tr]) for name, m in sels]
    sel_te = {name: m[te] for name, m in sels}

    all_rules, evaluated = beam_search(
        sel_tr, ytr, depth=args.depth, beam_width=args.beam,
        min_support=args.min_support)
    rows = validate_rules(
        all_rules, sel_te, yte, dpd_test=dpd_k[te], base_test=base_te,
        y_train=ytr, min_test=args.min_test)
    print(f"candidates evaluated: {evaluated}  (Bonferroni: significant if p < alpha/{evaluated})")
    print(f"top {args.top} subgroups by held-out TEST never-paid precision (min {args.min_test} test rows):\n")
    for r in rows[: args.top]:
        smell = (r["never_te"] / r["dpd_te"]) if r["dpd_te"] else float("nan")
        print(f"  • {r['conds']}")
        print(f"      test n={r['n_te']}  never-paid={r['never_te']:.1%} ({r['lift']:.1f}x)  "
              f"[train {r['never_tr']:.1%}]  DPD45={r['dpd_te']:.1%}  fraud-smell={smell:.0%}  p={r['p']:.1e}")


if __name__ == "__main__":
    main()
