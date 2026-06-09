"""Validate the graph-discovery winners (v1_76d3ad45).

The sweep found a >=90% net-new pocket: nb_comp>=2 & types>=3 -> 100% never-paid,
n=15. Before trusting n=15 at 100%, this script checks the two ways it could be a
mirage:
  1. ONE RING counted many times (anecdote, not a rule). Count the distinct
     connected components / users / resources behind the matched advances; report
     their time spread. A real rule recurs across many rings over time.
  2. OVERFIT to the snapshot. Out-of-time split: features are already as-of (each
     advance sees only its past), so precision on a held-out LATE period is an
     honest generalisation check. Tune-free -- just measure each rule early vs late.

Also reports Wilson 95% CIs (small-n honesty).

    uv run python -m projects.fraud_anomaly_detection.analysis.graph_validate_winner
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from projects.fraud_anomaly_detection.analysis.graph_discovery_sweep import (
    DATASET_ID, WARMUP_DAYS, RESOURCE_COLS, _clean, _load_env, build_graph_features)


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def main():
    _load_env()
    from automl.data.registry import load_dataset_by_id
    from automl.project.session import use_project
    from projects.fraud_anomaly_detection.scenarios import residual_mask

    sess = use_project("fraud_anomaly_detection", dry_run=False)
    df = load_dataset_by_id(DATASET_ID, session=sess).df
    n = len(df)

    ts = pd.to_datetime(df["feature_as_of_ts"])
    ts_int = ts.astype("int64").to_numpy()
    day_idx = (ts.dt.normalize() - ts.dt.normalize().min()).dt.days.to_numpy().astype(int)
    mat_dt = pd.to_datetime(df["expected_dpd45_date"], errors="coerce").astype("int64").to_numpy()

    res_m = residual_mask(df).to_numpy()
    mat = (df["label_mature_d45"].astype(float) == 1).to_numpy()
    dpd = (df["label_gross_dpd45"].astype(float) == 1).to_numpy()
    never = dpd & (df["label_repaid_current_snapshot"].astype(float) == 0).to_numpy()
    keep = res_m & mat & (day_idx >= WARMUP_DAYS)
    base = never[keep].mean()

    f = build_graph_features(df, ts_int, mat_dt, dpd)
    cu, ct, nbc, nbd, br = f["cu"], f["ct"], f["nb_comp"], f["nb_d1"], f["bad_rate"]

    rules = {
        # seed-based (lagged by DPD45 maturity)
        "nb_comp>=2 & types>=3 (the gem)": (nbc >= 2) & (ct >= 3),
        "nb_comp>=3": nbc >= 3,
        "nb_d1>=2": nbd >= 2,
        "nb_comp>=1 & types>=3": (nbc >= 1) & (ct >= 3),
        "nb_comp>=1 (review-tier volume)": nbc >= 1,
        # STRUCTURAL (no seed -> no maturity lag, usable real-time)
        "comp>=5 & types>=2 (structural)": (cu >= 5) & (ct >= 2),
        "comp>=3 & types>=3 (structural)": (cu >= 3) & (ct >= 3),
        "comp>=4 & types>=3 (structural)": (cu >= 4) & (ct >= 3),
    }

    # ── 1. distinct-ring / concentration check (on the matched, kept rows) ──
    print("=== concentration: distinct rings / users / time behind each rule ===")
    present = {c: df[c].to_numpy(object) for c in RESOURCE_COLS}
    users = df["user_id"].astype(str).to_numpy()
    dates = ts.dt.date.to_numpy()
    for lbl, m in rules.items():
        mm = m & keep
        idx = np.where(mm)[0]
        if len(idx) == 0:
            print(f"  {lbl}: n=0"); continue
        # small union-find over matched advances via shared resources -> # rings
        parent = {}
        def find(x):
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x
        def uni(a, b):
            parent[find(a)] = find(b)
        for i in idx:
            anchor = f"A:{i}"; find(anchor)
            for c in RESOURCE_COLS:
                cv = _clean(present[c][i])
                if cv is not None:
                    uni(anchor, f"{c[0]}:{cv}")
        rings = len({find(f"A:{i}") for i in idx})
        uu = len({users[i] for i in idx})
        dd = sorted({dates[i] for i in idx})
        print(f"  {lbl}: n={len(idx)} | distinct rings={rings} | distinct users={uu} | "
              f"days={len(dd)} | span {dd[0]}..{dd[-1]}")

    # ── 2. out-of-time split (median feature_as_of_ts on the kept rows) ──
    kts = ts_int[keep]
    cut = np.median(kts)
    early = keep & (ts_int <= cut)
    late = keep & (ts_int > cut)
    print(f"\n=== out-of-time split (cut={pd.to_datetime(int(cut))}) ===")
    print(f"  early kept={int(early.sum()):,} (base {never[early].mean():.2%})  "
          f"late kept={int(late.sum()):,} (base {never[late].mean():.2%})")
    print(f"\n  {'rule':<34}{'period':<7}{'n':>5}{'never%':>9}{'lift':>7}{'  Wilson95%CI'}")
    print("  " + "-" * 78)
    for lbl, m in rules.items():
        for tag, seg, b in [("early", early, never[early].mean()), ("late", late, never[late].mean())]:
            mm = m & seg
            k = int(mm.sum())
            if k == 0:
                print(f"  {lbl:<34}{tag:<7}{k:>5}"); continue
            kk = int(never[mm].sum()); p = kk / k
            lo, hi = wilson(kk, k)
            print(f"  {lbl:<34}{tag:<7}{k:>5}{p:>8.1%}{p/b:>6.1f}x   [{lo:.0%}, {hi:.0%}]")


if __name__ == "__main__":
    main()
