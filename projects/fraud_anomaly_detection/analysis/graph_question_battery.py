"""Question battery against a built entity-graph store — the analysis tool.

Where graph_store_demo proves the plumbing, THIS answers the standing
questions on whatever store you point it at (sample now, v3 later):

  1. store contents          4. multi-type census + the RESIDUAL discovery cut
  2. pooled base rates       5. proximity to flagged users (bad-neighbour view)
  3. hubs vs all three truth columns (is_fraud / DPD45 / scenario coverage)

Truth columns are different things — keep them straight in every readout:
``is_fraud`` is the upstream confirmed-fraud label (the training target),
``label_gross_dpd45`` is the bad-outcome proxy (fraud SUBSET credit losses),
``scenario_<name>`` flags are OUR register, computed dynamically at run time.

Rates follow the settled pooling decision (HANDOFF 2026-06-09): user-level
labels aggregate only advances that are MATURE (label_mature_d45=1) and at or
after --pool-start (default 2025-08-01) — no undefined labels diluting rates,
~7-month graph warm-up before the first scored row. Edges always keep the
store's full history (connectivity is as-of history, not a scored row).

    uv run --group fraud python -m projects.fraud_anomaly_detection.analysis.graph_question_battery
    uv run --group fraud python -m ... --store projects/fraud_anomaly_detection/data/graph/fraud_graph_v3.duckdb
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import pandas as pd

from projects.fraud_anomaly_detection.graph.load import load_graph
from projects.fraud_anomaly_detection.graph.queries import (
    components,
    hub_report,
    near_flagged,
)
from projects.fraud_anomaly_detection.scenarios import assign

DEFAULT_STORE = Path("projects/fraud_anomaly_detection/data/graph/fraud_graph.duckdb")
RING_CAP = 20  # the v1 traversal finding: caps 10-50 gave identical net-new results


def _pct(x: float) -> str:
    return f"{x:.1%}" if pd.notna(x) else "n/a"


def banner(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def user_truth(base: pd.DataFrame, pool_start: str | None, mature_only: bool) -> pd.DataFrame:
    """User-level truth table over the POOLED advances (rates denominator)."""
    pooled = base
    if mature_only:
        pooled = pooled[pooled["label_mature_d45"].astype(float) == 1]
    if pool_start:
        pooled = pooled[pd.to_datetime(pooled["feature_as_of_ts"]) >= pd.Timestamp(pool_start)]
    flags = assign(pooled)
    per_advance = pd.DataFrame({
        "user_id": pooled["user_id"].astype(str),
        "is_fraud": pooled["is_fraud"].astype(int),
        "dpd45": pooled["label_gross_dpd45"].astype(int),
        "scen": flags["scenario_any"].astype(bool),
        "first_ts": pd.to_datetime(pooled["feature_as_of_ts"]),
    })
    return per_advance.groupby("user_id").agg(
        is_fraud=("is_fraud", "max"), dpd45=("dpd45", "max"),
        scen=("scen", "max"), first_ts=("first_ts", "min"),
    )


def _rates(sub: pd.DataFrame, label: str) -> str:
    if not len(sub):
        return f"  {label}: 0 users"
    return (f"  {label}: {len(sub):,} users | fraud {int(sub.is_fraud.sum()):,}"
            f" ({_pct(sub.is_fraud.mean())}) | dpd45 {int(sub.dpd45.sum()):,}"
            f" ({_pct(sub.dpd45.mean())}) | scenario-flagged {_pct(sub.scen.mean())}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--store", type=Path, default=DEFAULT_STORE)
    ap.add_argument("--pool-start", default="2025-08-01",
                    help="rate pool lower bound on feature_as_of_ts ('' disables)")
    ap.add_argument("--no-mature-only", action="store_true",
                    help="include label-immature advances in the rate pool")
    ap.add_argument("--top-hubs", type=int, default=15)
    ap.add_argument("--max-hops", type=int, default=3)
    args = ap.parse_args()
    if not args.store.exists():
        raise SystemExit(f"store not found: {args.store} — build it first "
                         "(graph_store_demo for the sample, graph_store_build for v3)")

    banner(f"1) STORE: {args.store}")
    con = duckdb.connect(str(args.store), read_only=True)
    print(con.execute("SELECT key, value FROM meta WHERE key IN"
                      " ('built_at','source','n_advances','n_users','n_edges')")
          .df().to_string(index=False, header=False))
    base = con.execute("SELECT * FROM advances").df()

    banner("2) POOLED BASE RATES (mature"
           + ("" if not args.no_mature_only else " OFF")
           + (f", from {args.pool_start}" if args.pool_start else "") + ")")
    truth = user_truth(base, args.pool_start or None, not args.no_mature_only)
    _all = truth
    print(_rates(_all, "pool"))

    banner(f"3) HUBS top-{args.top_hubs}: degree vs all three truth columns")
    hubs = hub_report(args.store, top_n=args.top_hubs)
    hub_users = con.execute(
        "SELECT DISTINCT entity_type, entity_value, user_id FROM edges").df()
    hub_users["user_id"] = hub_users.user_id.astype(str)
    m = (hub_users.merge(hubs[["entity_type", "entity_value", "n_users", "span_days"]],
                         on=["entity_type", "entity_value"])
                  .merge(truth, left_on="user_id", right_index=True))
    agg = (m.groupby(["entity_type", "entity_value"])
            .agg(n_users=("user_id", "nunique"), span_days=("span_days", "first"),
                 fraud_rate=("is_fraud", "mean"), dpd45_rate=("dpd45", "mean"),
                 scen_coverage=("scen", "mean"))
            .sort_values("n_users", ascending=False).reset_index())
    agg["entity_value"] = agg.entity_value.str[:20] + "…"
    for col in ("fraud_rate", "dpd45_rate", "scen_coverage"):
        agg[col] = (agg[col] * 100).round(1)
    print(agg.to_string(index=False))
    print("  (rates over POOLED users only; scen_coverage ~100% means the hub"
          " is already caught by the register, not a discovery)")
    con.close()

    banner(f"4) MULTI-TYPE CENSUS (cap={RING_CAP}) + RESIDUAL discovery cut")
    g = load_graph(args.store, base=base, degree_cap=RING_CAP,
                   node_attrs=("is_fraud",), scenarios=False)
    cc = components(g, flag="is_fraud")
    user_comp = {u: cid for cid, row in zip(cc.comp_id, cc.user_ids)
                 for u in row.split(",")}
    split_ts = truth.first_ts.median()  # early/late decay check (v1 lesson #2)
    for label, sel in [
        (">=3 users & >=2 types", (cc.n_users >= 3) & (cc.n_types >= 2)),
        (">=5 users & >=2 types (the v1 durable rule)", (cc.n_users >= 5) & (cc.n_types >= 2)),
    ]:
        comp_set = cc[sel]
        members = set(u for row in comp_set.user_ids for u in row.split(","))
        sub = truth.loc[truth.index.isin(members)]
        print(f"\n  {label}: {len(comp_set):,} components, {len(members):,} member users")
        print(_rates(sub, "    members (pooled)"))
        print(_rates(sub[sub.first_ts <= split_ts], "      early half"))
        print(_rates(sub[sub.first_ts > split_ts], "      late half"))
        residual = sub[~sub.scen]
        print(_rates(residual, "    RESIDUAL (no scenario fired)"))
        if len(residual):
            n_comps = len({user_comp[u] for u in residual.index})
            print(f"      spans {n_comps} distinct components"
                  " (v1 lesson #1: 1-3 rings = anecdote, not a rule)")
            print(_rates(residual[residual.first_ts <= split_ts], "      early half"))
            print(_rates(residual[residual.first_ts > split_ts], "      late half"))
    del g

    banner(f"5) PROXIMITY (<= {args.max_hops} user-hops, union graph, uncapped)")
    g = load_graph(args.store, base=base, node_attrs=("is_fraud",), scenarios=True)
    for flag in ("scenario_any", "is_fraud"):
        out = near_flagged(g, flag=flag, max_hops=args.max_hops)
        print(f"\n  seeds = {flag}: {len(out):,} users within {args.max_hops} hops")
        if len(out):
            print("    by hops: " + ", ".join(
                f"{h}:{n}" for h, n in out.hops.value_counts().sort_index().items()))
            near = truth.loc[truth.index.isin(set(out.user_id))]
            print(_rates(near, "    proximate users (pooled)"))
            unflagged = near[~near.scen]
            print(_rates(unflagged, "    proximate & NOT scenario-flagged"))
            if len(unflagged):
                split_ts = truth.first_ts.median()
                print(_rates(unflagged[unflagged.first_ts <= split_ts], "      early half"))
                print(_rates(unflagged[unflagged.first_ts > split_ts], "      late half"))

    print("\nDone. Sample-store caveat: fraud-enriched + graph-thinned —"
          " treat rates as workflow output, findings only on the full build.")


if __name__ == "__main__":
    main()
