"""Coverage / seed-availability diagnostic for the graph proximity signal.

Answers two questions raised in review:
  (1) Are we evaluating only MATURED rows? (yes -- this script makes the
      denominator explicit and shows what immature rows would do.)
  (2) What FRACTION of transactions actually carry a bad-neighbour count, and is
      the early->late precision decay caused by seed STARVATION of recent
      advances (my earlier claim) or by small-n early optimism regressing to the
      true rate as seeds accumulate? Late advances have MORE prior history, so if
      they fire on MORE rows, the starvation story is wrong.

    uv run python -m projects.fraud_anomaly_detection.analysis.graph_seed_coverage
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from projects.fraud_anomaly_detection.analysis.graph_discovery_sweep import (
    DATASET_ID, WARMUP_DAYS, RESOURCE_COLS, _clean, _load_env, build_graph_features)


def _ring_count(idx, present, df):
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for i in idx:
        a = f"A:{i}"; find(a)
        for c in RESOURCE_COLS:
            cv = _clean(present[c][i])
            if cv is not None:
                parent[find(a)] = find(f"{c[0]}:{cv}")
    return len({find(f"A:{i}") for i in idx})


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
    warm = day_idx >= WARMUP_DAYS

    f = build_graph_features(df, ts_int, mat_dt, dpd)
    cu, ct, nbc, nbd = f["cu"], f["ct"], f["nb_comp"], f["nb_d1"]
    present = {c: df[c].to_numpy(object) for c in RESOURCE_COLS}

    # (1) maturity: how much of the residual is even mature?
    res_warm = res_m & warm
    print("=== population funnel (residual + post-warmup) ===")
    print(f"  residual+warmup rows:        {int(res_warm.sum()):,}")
    print(f"  ... of which MATURE (d45):   {int((res_warm & mat).sum()):,} "
          f"({(res_warm & mat).mean() / res_warm.mean():.1%} of residual+warmup)")
    print(f"  ... immature (excluded):     {int((res_warm & ~mat).sum()):,}  "
          f"(label undefined -> correctly NOT counted)\n")

    keep = res_warm & mat
    base = never[keep].mean()

    # (2a) coverage: what fraction of kept rows carry a bad-neighbour count?
    print("=== coverage on matured eval population ===")
    print(f"  kept (eval denom): {int(keep.sum()):,} | base never-paid {base:.3%}")
    for lbl, m in [("nb_comp>=1", nbc >= 1), ("nb_comp>=2", nbc >= 2), ("nb_comp>=3", nbc >= 3),
                   ("nb_d1>=1", nbd >= 1), ("structural comp>=5 & types>=2", (cu >= 5) & (ct >= 2))]:
        mk = m & keep
        k = int(mk.sum())
        print(f"  {lbl:<32} n={k:>5}  = {k / keep.sum():.4%} of kept   "
              f"never={never[mk].mean() if k else float('nan'):.1%}  n_np={int(never[mk].sum())}")

    # (2b) seed availability + firing, early vs late -> tests the starvation story
    kts = ts_int[keep]
    cut = np.median(kts)
    early = keep & (ts_int <= cut)
    late = keep & (ts_int > cut)
    # active seeds = DPD45 advances whose maturity precedes the period's median advance time
    seed_mat = mat_dt[dpd & np.isfinite(mat_dt)]
    em, lm = np.median(ts_int[early]), np.median(ts_int[late])
    print("\n=== seed availability: early vs late (tests 'recent advances are seed-starved') ===")
    print(f"  total DPD45-matured seeds: {len(seed_mat):,}")
    print(f"  seeds matured before EARLY median ({pd.to_datetime(int(em)).date()}): "
          f"{int((seed_mat < em).sum()):,}")
    print(f"  seeds matured before LATE  median ({pd.to_datetime(int(lm)).date()}): "
          f"{int((seed_mat < lm).sum()):,}")
    print("  -> if LATE >> EARLY, late advances have MORE seeds, so decay is NOT starvation\n")

    print("=== nb_comp>=1 firing + precision + distinct rings, by period ===")
    for tag, seg in [("early", early), ("late", late)]:
        m = (nbc >= 1) & seg
        idx = np.where(m)[0]
        b = never[seg].mean()
        rings = _ring_count(idx, present, df) if len(idx) else 0
        p = never[m].mean() if len(idx) else float("nan")
        print(f"  {tag}: fires n={len(idx):>3}  never={p:.1%}  (period base {b:.2%}, "
              f"lift {p/b:.1f}x)  distinct rings={rings}  mean nb_comp={nbc[seg].mean():.4f}")


if __name__ == "__main__":
    main()
