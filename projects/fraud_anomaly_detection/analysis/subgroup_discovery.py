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


def _binom_sf(k: int, n: int, p: float) -> float:
    """One-sided P(X >= k) under Binomial(n, p), normal approximation with a
    continuity correction — significance of the test precision vs base."""
    from math import erfc, sqrt
    if n == 0 or p <= 0 or p >= 1:
        return 1.0
    z = (k - 0.5 - n * p) / sqrt(n * p * (1 - p))
    return 0.5 * erfc(z / sqrt(2))


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

    ytr, yte = y[tr], y[te]
    sel_tr = [(name, m[tr]) for name, m in sels]
    sel_te = {name: m[te] for name, m in sels}

    # beam of (selector-name-tuple, train_mask) — quality = train never-paid precision
    def precision(mask, yy):
        n = int(mask.sum())
        return (yy[mask].mean() if n else 0.0), n

    evaluated = 0
    beam: list[tuple[tuple[str, ...], np.ndarray]] = []
    for name, m in sel_tr:
        evaluated += 1
        p, n = precision(m, ytr)
        if n >= args.min_support:
            beam.append(((name,), m))
    beam.sort(key=lambda b: precision(b[1], ytr)[0], reverse=True)
    beam = beam[: args.beam]

    seen: set[frozenset] = set(frozenset(c) for c, _ in beam)
    all_rules: dict[frozenset, np.ndarray] = {frozenset(c): m for c, m in beam}

    for _ in range(args.depth - 1):
        cand: list[tuple[tuple[str, ...], np.ndarray]] = []
        for conds, m in beam:
            for name, sm in sel_tr:
                if name in conds:
                    continue
                key = frozenset(conds + (name,))
                if key in seen:
                    continue
                nm = m & sm
                evaluated += 1
                n = int(nm.sum())
                if n < args.min_support:
                    continue
                seen.add(key)
                cand.append((tuple(sorted(key)), nm))
                all_rules[key] = nm
        if not cand:
            break
        cand.sort(key=lambda b: precision(b[1], ytr)[0], reverse=True)
        beam = cand[: args.beam]

    # validate every discovered rule on held-out TEST
    rows = []
    for key, _m in all_rules.items():
        mask_te = np.ones(te.sum(), bool)
        for name in key:
            mask_te = mask_te & sel_te[name]
        n_te = int(mask_te.sum())
        if n_te < args.min_test:
            continue
        prec_te = yte[mask_te].mean()
        dpd_te = dpd_k[te][mask_te].mean()
        prec_tr = ytr[all_rules[key]].mean() if all_rules[key].sum() else float("nan")
        rows.append({
            "conds": " AND ".join(sorted(key)), "n_te": n_te,
            "never_te": prec_te, "lift": prec_te / base_te,
            "never_tr": prec_tr, "dpd_te": dpd_te,
            "p": _binom_sf(int(round(prec_te * n_te)), n_te, base_te),
        })
    # Dedup: collapse subgroups with an identical test footprint (same n_te and
    # precision = an inert extra condition), keeping the SHORTEST conjunction.
    rows.sort(key=lambda r: (r["n_te"], round(r["never_te"], 6), r["conds"].count(" AND ")))
    deduped: dict[tuple, dict] = {}
    for r in rows:
        sig = (r["n_te"], round(r["never_te"], 6))
        if sig not in deduped or r["conds"].count(" AND ") < deduped[sig]["conds"].count(" AND "):
            deduped[sig] = r
    rows = sorted(deduped.values(), key=lambda r: r["never_te"], reverse=True)
    print(f"candidates evaluated: {evaluated}  (Bonferroni: significant if p < alpha/{evaluated})")
    print(f"top {args.top} subgroups by held-out TEST never-paid precision (min {args.min_test} test rows):\n")
    for r in rows[: args.top]:
        smell = (r["never_te"] / r["dpd_te"]) if r["dpd_te"] else float("nan")
        print(f"  • {r['conds']}")
        print(f"      test n={r['n_te']}  never-paid={r['never_te']:.1%} ({r['lift']:.1f}x)  "
              f"[train {r['never_tr']:.1%}]  DPD45={r['dpd_te']:.1%}  fraud-smell={smell:.0%}  p={r['p']:.1e}")


if __name__ == "__main__":
    main()
