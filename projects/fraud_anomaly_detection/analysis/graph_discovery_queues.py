"""Run the seven snapshot discovery queues against a built store.

The actionable counterpart to graph_question_battery: where the battery
measures, THIS produces review queues — ranked unflagged users/entities with
the evidence attached. "Flagged" = the current scenario register (computed at
run time) plus the confirmed-fraud label; both stay caller-side inputs to the
queue functions, so the register remains the single owner of "known".

Outcome columns (dpd45/fraud rates per queue) are printed as a sanity check —
on the fraud-enriched sample they are workflow evidence, not findings.

    uv run --group fraud python -m projects.fraud_anomaly_detection.analysis.graph_discovery_queues
    uv run --group fraud python -m ... --store <path> --days 7 --top 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import pandas as pd

from projects.fraud_anomaly_detection.graph.dense import dense_blocks
from projects.fraud_anomaly_detection.graph.discover import (
    RING_CAP,
    bad_neighbours,
    emerging_farms,
    fresh_rings,
    multi_witness_pairs,
    residual_ring_members,
    suspicion_queue,
)
from projects.fraud_anomaly_detection.graph.load import load_graph
from projects.fraud_anomaly_detection.scenarios import assign

DEFAULT_STORE = Path("projects/fraud_anomaly_detection/data/graph/fraud_graph.duckdb")


def banner(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def outcome_check(users: pd.Series, truth: pd.DataFrame, label: str) -> None:
    sub = truth.loc[truth.index.isin(set(users.astype(str)))]
    if not len(sub):
        print(f"  {label}: queue empty")
        return
    print(f"  {label}: {len(sub):,} users | dpd45 {sub.dpd45.mean():.1%}"
          f" | confirmed-fraud {sub.is_fraud.mean():.1%}"
          f" (sample sanity check, not a finding)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--store", type=Path, default=DEFAULT_STORE)
    ap.add_argument("--top", type=int, default=10, help="rows shown per queue")
    ap.add_argument("--days", type=int, default=7, help="fresh-rings window")
    args = ap.parse_args()
    if not args.store.exists():
        raise SystemExit(f"store not found: {args.store} — build it first")

    with duckdb.connect(str(args.store), read_only=True) as con:
        base = con.execute("SELECT * FROM advances").df()
    flags = assign(base)
    truth = (pd.DataFrame({
        "user_id": base.user_id.astype(str),
        "scen": flags.scenario_any.astype(bool),
        "is_fraud": base.is_fraud.astype(int),
        "dpd45": base.label_gross_dpd45.astype(int),
    }).groupby("user_id").max())
    user_flags = (truth.scen.astype(bool) | truth.is_fraud.astype(bool))
    print(f"store: {args.store} | {len(truth):,} users |"
          f" flagged (register OR confirmed fraud): {user_flags.mean():.1%}")

    g_ring = load_graph(args.store, base=base, degree_cap=RING_CAP,
                        node_attrs=("is_fraud",), scenarios=True)

    banner("QUEUE 1 — residual ring members (unflagged users in multi-type rings)")
    q1 = residual_ring_members(g_ring, flag="scenario_any")
    print(q1.head(args.top).to_string(index=False))
    outcome_check(q1.user_id, truth, "all queue members")

    banner("QUEUE 2 — bad neighbours (near flagged/fraud, flagged by neither)")
    g_union = load_graph(args.store, base=base, node_attrs=("is_fraud",), scenarios=True)
    q2 = bad_neighbours(g_union, flags=("scenario_any", "is_fraud"), max_hops=2)
    print(q2.head(args.top).to_string(index=False))
    outcome_check(q2.user_id, truth, "all queue members")

    banner("QUEUE 3 — emerging farms (velocity hubs the flags haven't caught)")
    q3 = emerging_farms(args.store, user_flags=user_flags)
    show = q3.head(args.top).copy()
    show["entity_value"] = show.entity_value.str[:24] + "…"
    print(show[["entity_type", "entity_value", "n_users", "span_days",
                "users_per_day", "flagged_coverage", "score"]].to_string(index=False))

    banner("QUEUE 4 — multi-witness pairs (>=2 channels agree, both unflagged)")
    q4 = multi_witness_pairs(args.store, user_flags=user_flags)
    print(q4.head(args.top).to_string(index=False))
    members = pd.concat([q4.user_a, q4.user_b]) if len(q4) else pd.Series(dtype=str)
    outcome_check(members, truth, "all paired users")

    banner(f"QUEUE 5 — fresh rings (formed in the store's last {args.days} days)")
    q5 = fresh_rings(args.store, days=args.days)
    cols = ["comp_id", "n_users", "n_types", "entity_types", "n_flagged"]
    print(q5.head(args.top)[cols].to_string(index=False) if len(q5)
          else "  none in window")

    banner("QUEUE 6 — PPR suspicion (diffused guilt-by-association, unflagged)")
    q6 = suspicion_queue(g_union, seed_flag="is_fraud",
                         exclude_flags=("scenario_any", "is_fraud"))
    if len(q6):
        print(q6.head(args.top).to_string(index=False))
        outcome_check(q6.user_id, truth, "all queue members")
        outcome_check(q6.head(args.top).user_id, truth, f"top {args.top}")
    else:
        print("  no seeds or nothing unflagged in reach")

    banner("QUEUE 7 — dense blocks (Fraudar-style peeling, camouflage-resistant)")
    q7 = dense_blocks(args.store, top_k=args.top)
    if len(q7):
        print(q7.drop(columns="user_ids").to_string(index=False))
        for block in q7.head(3).itertuples(index=False):
            members = pd.Series(block.user_ids.split(","))
            unflagged = members[~members.isin(user_flags[user_flags].index)]
            outcome_check(members, truth, f"block {block.block_id} members")
            print(f"    unflagged members: {len(unflagged)}/{len(members)}")
    else:
        print("  none")

    print("\nDone. Queues are snapshot-semantics review lists; precision-grade"
          " measurement of any rule derived from them goes through asof.leakfree_features.")


if __name__ == "__main__":
    main()
