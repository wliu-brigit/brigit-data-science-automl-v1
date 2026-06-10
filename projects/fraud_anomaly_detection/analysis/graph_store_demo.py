"""Capability demo for the persisted entity-graph store, on the local sample.

Proves the spec's question list end-to-end: build -> persist -> reopen ->
flag (current register, dynamically) -> proximity / per-layer / components /
hubs / ring deep-dive. Read-only toward the sample; writes only the store
file. Sample is fraud-enriched and graph-thinned: NUMBERS HERE ARE
CAPABILITY EVIDENCE, NOT TRANSFERABLE METRICS.

    uv run --group fraud python -m projects.fraud_anomaly_detection.analysis.graph_store_demo
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from projects.fraud_anomaly_detection.graph.build import build_store
from projects.fraud_anomaly_detection.graph.load import DEFAULT_LAYERS, load_graph
from projects.fraud_anomaly_detection.graph.queries import (
    components,
    hub_report,
    near_flagged,
    project_users,
    ring,
)

PROJECT = Path("projects/fraud_anomaly_detection")
SAMPLE = PROJECT / "data" / "sample" / "graph_sample.parquet"
STORE = PROJECT / "data" / "graph" / "fraud_graph.duckdb"


def banner(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main() -> None:
    banner("1) BUILD: sample parquet -> persisted store")
    summary = build_store(SAMPLE, STORE, source_label=f"sample:{SAMPLE.name}")
    for key, val in summary.items():
        print(f"  {key:<22}{val:>10,}")

    banner("2) REOPEN: fresh connection, plain SQL inspection")
    with duckdb.connect(str(STORE), read_only=True) as con:
        print(con.execute(
            "SELECT entity_type, count(*) n_edges, count(DISTINCT entity_value) n_values"
            " FROM edges GROUP BY 1 ORDER BY 2 DESC").df().to_string(index=False))

    banner("3) LOAD + dynamic scenario overlay (current register)")
    base = pd.read_parquet(SAMPLE)
    from projects.fraud_anomaly_detection.scenarios import TRIGGER_COLUMNS

    missing = sorted(set(TRIGGER_COLUMNS) - set(base.columns))
    if missing:
        raise SystemExit(f"sample is missing register trigger columns: {missing}")
    g = load_graph(STORE, base=base)
    flag_cols = sorted(a for a in g.vs.attributes() if a.startswith("scenario_"))
    for col in flag_cols:
        n = sum(bool(v[col]) for v in g.vs if v["kind"] == "user")
        print(f"  {col:<40}{n:>8,} users")

    banner("4) PROXIMITY: users within 1-3 user-hops of a scenario-flagged user")
    out = near_flagged(g, flag="scenario_any", max_hops=3)
    print(f"  union graph ({'+'.join(DEFAULT_LAYERS)}): {len(out):,} users near a flagged one")
    if len(out):
        print(out["hops"].value_counts().sort_index().rename("users").to_string())
    for layer in DEFAULT_LAYERS:
        gl = load_graph(STORE, base=base, layers=(layer,))
        nl = len(near_flagged(gl, flag="scenario_any", max_hops=3))
        print(f"  {layer:>12}-only: {nl:,}")

    banner("5) COMPONENT CENSUS: multi-type density (the v1 discriminator)")
    cap_g = load_graph(STORE, base=base, degree_cap=20)
    cc = components(cap_g, flag="is_fraud")
    multi = cc[(cc.n_users >= 3) & (cc.n_types >= 2)]
    print(f"  components (cap=20): {len(cc):,}; with >=3 users & >=2 types: {len(multi):,}")
    print(f"  fraud users inside multi-type comps: {int(multi.n_flagged.sum()):,}"
          f" / {int(multi.n_users.sum()):,} members")
    if len(multi):
        print(multi.sort_values("n_users", ascending=False).head(10)
              [["n_users", "n_types", "entity_types", "n_flagged"]].to_string(index=False))

    banner("6) HUB REPORT: no cap — farms vs infrastructure by time density")
    print(hub_report(STORE, top_n=15).to_string(index=False))

    banner("7) RING DEEP-DIVE: largest multi-type component")
    if len(multi):
        target_users = multi.sort_values("n_users", ascending=False).iloc[0]["user_ids"]
        center = target_users.split(",")[0]
        sub = ring(cap_g, center, hops=2)
        print(f"  ego graph around {center}: {sub.vcount()} vertices, {sub.ecount()} edges")
        proj = project_users(STORE, degree_cap=20)
        strong = proj[proj.n_types >= 2]
        print(f"  user-user projected pairs (cap=20): {len(proj):,};"
              f" multi-type pairs: {len(strong):,}")
    else:
        print("  no multi-type component >=3 users in the (thinned) sample")

    print("\nStore persisted at:", STORE)


if __name__ == "__main__":
    main()
