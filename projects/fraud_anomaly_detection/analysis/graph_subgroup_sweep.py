"""Subgroup discovery over the GRAPH feature space, from a built store.

Where subgroup_discovery.py sweeps the base-table columns of the pinned
dataset, THIS joins the leak-free per-advance graph features
(asof.leakfree_features: ring size, type diversity, bad-neighbour counts —
strictly-prior, maturity-activated) onto the store's own advances snapshot
and lets the same beam search hunt conjunctions across BOTH spaces. The
question it answers systematically: does graph position add rule-grade
signal beyond the as-of sharing counts, and in combination with what?

Rigor is inherited from subgroup_core: train/test split (SPLIT_PCT when the
snapshot carries it, otherwise an out-of-time split at --train-frac of the
timeline), min-support floor, held-out validation, Bonferroni eyeball.
Residual + mature pool only — the register owns what is already caught.

    uv run --group fraud python -m projects.fraud_anomaly_detection.analysis.graph_subgroup_sweep
    uv run --group fraud python -m ... --store .../fraud_graph_v3.duckdb
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from projects.fraud_anomaly_detection.analysis.subgroup_core import beam_search, validate_rules
from projects.fraud_anomaly_detection.analysis.subgroup_discovery import build_selectors
from projects.fraud_anomaly_detection.graph.asof import leakfree_features
from projects.fraud_anomaly_detection.scenarios import SCENARIOS, residual_mask

DEFAULT_STORE = Path("projects/fraud_anomaly_detection/data/graph/fraud_graph.duckdb")

# graph features are small counts — exact integer selectors
GRAPH_SELECTORS: dict[str, tuple[int, ...]] = {
    "nb_comp": (1, 2, 3),      # matured-bad users in the component (other people)
    "nb_d1": (1, 2),           # matured-bad users one entity away
    "comp_users": (2, 3, 5, 10),
    "comp_types": (2, 3),
}


def graph_selector_masks(sub: pd.DataFrame) -> list[tuple[str, np.ndarray]]:
    sels: list[tuple[str, np.ndarray]] = []
    for col, thresholds in GRAPH_SELECTORS.items():
        v = pd.to_numeric(sub[col], errors="coerce").to_numpy(float)
        for t in thresholds:
            sels.append((f"{col} >= {t}", v >= t))
    return sels


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--store", type=Path, default=DEFAULT_STORE)
    ap.add_argument("--degree-cap", type=int, default=20)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--beam", type=int, default=80)
    ap.add_argument("--min-support", type=int, default=80)
    ap.add_argument("--min-test", type=int, default=30)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--train-frac", type=float, default=0.7,
                    help="out-of-time split point when SPLIT_PCT is absent")
    args = ap.parse_args()
    if not args.store.exists():
        raise SystemExit(f"store not found: {args.store} — build it first")

    with duckdb.connect(str(args.store), read_only=True) as con:
        base = con.execute("SELECT * FROM advances").df()
    print(f"store: {args.store} | {len(base):,} advances")
    print(f"gated out: {[s.name for s in SCENARIOS]}")

    print(f"replaying leak-free graph features (cap={args.degree_cap})...")
    graph_feats = leakfree_features(args.store, degree_cap=args.degree_cap)
    df = base.merge(
        graph_feats[["advance_id", "comp_users", "comp_types", "nb_comp", "nb_d1"]],
        on="advance_id", how="left",
    )

    res = residual_mask(df).to_numpy()
    mat = (df["label_mature_d45"].astype(float) == 1).to_numpy()
    dpd = (df["label_gross_dpd45"].astype(float) == 1).to_numpy()
    never = dpd & (df["label_repaid_current_snapshot"].astype(float) == 0).to_numpy()
    keep = res & mat
    sub = df[keep].reset_index(drop=True)
    y = never[keep].astype(int)
    dpd_k = dpd[keep].astype(int)

    if "SPLIT_PCT" in sub.columns:
        split = pd.to_numeric(sub["SPLIT_PCT"], errors="coerce").to_numpy(float)
        tr, te = split < 80, split >= 80
        split_desc = "SPLIT_PCT <80/>=80"
    else:
        ts = pd.to_datetime(sub["feature_as_of_ts"])
        cut = ts.quantile(args.train_frac)
        tr = (ts <= cut).to_numpy()
        te = ~tr
        split_desc = f"out-of-time at {cut} (train-frac {args.train_frac})"

    base_tr, base_te = y[tr].mean(), y[te].mean()
    print(f"residual+mature pool: {keep.sum():,} | split: {split_desc}")
    print(f"train {tr.sum():,} (base {base_tr:.3%}) | test {te.sum():,} (base {base_te:.3%})")

    sels = build_selectors(sub) + graph_selector_masks(sub)
    n_graph = len(graph_selector_masks(sub))
    print(f"selectors: {len(sels)} ({n_graph} graph-positional) | depth {args.depth},"
          f" beam {args.beam}, min support {args.min_support}\n")

    sel_tr = [(name, m[tr]) for name, m in sels]
    sel_te = {name: m[te] for name, m in sels}
    all_rules, evaluated = beam_search(
        sel_tr, y[tr], depth=args.depth, beam_width=args.beam,
        min_support=args.min_support)
    rows = validate_rules(
        all_rules, sel_te, y[te], dpd_test=dpd_k[te], base_test=base_te,
        y_train=y[tr], min_test=args.min_test)

    graph_cols = tuple(GRAPH_SELECTORS)
    print(f"candidates evaluated: {evaluated}  (Bonferroni: significant if p < alpha/{evaluated})")
    print(f"top {args.top} subgroups by held-out TEST never-paid precision"
          f" (* = uses a graph-positional feature):\n")
    for row in rows[: args.top]:
        star = "*" if any(c in row["conds"] for c in graph_cols) else " "
        print(f" {star} {row['conds']}")
        print(f"      test n={row['n_te']}  never-paid={row['never_te']:.1%}"
              f" ({row['lift']:.1f}x)  [train {row['never_tr']:.1%}]"
              f"  DPD45={row['dpd_te']:.1%}  p={row['p']:.1e}")
    print("\nSample-store caveat: fraud-enriched + graph-thinned — rules here are"
          " workflow evidence; findings only on the full build, then through"
          " asof-measured register draft -> backtest -> sign-off.")


if __name__ == "__main__":
    main()
