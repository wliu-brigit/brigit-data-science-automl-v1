"""Institution concentration screen on the gated residual (dataset v1_76d3ad45).

wendao asked about "chase card" / institution as a signal. is_neobank_high_risk
is a coarse flag (2.3x over 15% of the population); this asks the finer question
— do SPECIFIC institutions concentrate never-paid beyond the neobank flag? If
one institution is the locus, that is a sharper feature than the blanket flag.

institution_id / institution_name are already in the table (metadata-flagged),
so this is free. Card-link sharing and income/payroll are NOT here (Tier-3 pull).

    uv run python -m projects.fraud_anomaly_detection.analysis.institution_screen
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


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
    from automl.data.registry import load_dataset_by_id
    from automl.project.session import use_project
    from projects.fraud_anomaly_detection.scenarios import residual_mask

    sess = use_project("fraud_anomaly_detection", dry_run=False)
    df = load_dataset_by_id("v1_76d3ad45", session=sess).df

    res = residual_mask(df).to_numpy()
    mat = (df["label_mature_d45"].astype(float) == 1).to_numpy()
    dpd = (df["label_gross_dpd45"].astype(float) == 1).to_numpy()
    never = (dpd & (df["label_repaid_current_snapshot"].astype(float) == 0).to_numpy())
    keep = res & mat
    sub = df[keep].reset_index(drop=True)
    nv = never[keep]
    base = nv.mean()
    print(f"residual + mature: {keep.sum():,} rows | base never-paid {base:.3%}\n")

    g = (sub.assign(_never=nv)
            .groupby("institution_name")
            .agg(n=("_never", "size"), never=("_never", "mean"),
                 neobank=("is_neobank_high_risk_institution", "mean"))
            .reset_index())
    g = g[g["n"] >= 500].copy()
    g["lift"] = g["never"] / base
    g = g.sort_values("never", ascending=False)

    print("=== institutions by never-paid (>=500 residual-mature rows) — top 20 ===")
    print(f"  {'institution':<34} {'n':>7} {'never%':>8} {'lift':>7} {'neobank':>8}")
    for _, r in g.head(20).iterrows():
        print(f"  {str(r['institution_name'])[:33]:<34} {int(r['n']):>7} "
              f"{r['never']:>7.1%} {r['lift']:>6.1f}x {r['neobank']:>7.0%}")

    print(f"\n  (total institutions with >=500 rows: {len(g)}; "
          f"share of all residual-mature rows in top 20: "
          f"{g.head(20)['n'].sum()/keep.sum():.0%})")

    # how much of the never-paid volume sits in high-rate institutions
    for thr in (0.10, 0.15, 0.20):
        hot = g[g["never"] >= thr]
        n_rows = int(hot["n"].sum())
        n_never = int((hot["n"] * hot["never"]).sum())
        print(f"  institutions with never-paid >= {thr:.0%}: {len(hot)} insts, "
              f"{n_rows:,} rows, ~{n_never:,} never-paid ({n_never/int((nv).sum()):.0%} of all residual never-paid)")


if __name__ == "__main__":
    main()
