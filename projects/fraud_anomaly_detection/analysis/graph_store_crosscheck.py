"""Cross-check a built graph store against the scenario/warehouse semantics.

Second-eye validation harness (first run 2026-06-10, sample store). Five
checks, all read-only against the store; run it as the sanity gate after any
store build (sample or v3) and after the link-grain edge work lands:

  0. (optional --parquet) store vs source-parquet fidelity
  1. scenario engine (pandas) vs hand-compiled SQL over the store's advances
     — MUST match exactly; proves the DuckDB ingest + engine mask algebra
  2. data QA: label/maturity coherence, sharing-window monotonicity,
     per-user identity-timestamp constancy
  3. graph-edge recount of users_on_{device,bank}_72h vs the stored
     warehouse columns — quantifies the advance-grain blind spot
     (2026-06-10 sample result: graph reproduces only ~73-75% of trigger
     flags; ZERO overcounts — the graph is a strict subset until link-grain
     edges land; see TODO "LINK-GRAIN EDGES")
  4. naive 'past 72h' on edge timestamps vs the identity-creation window the
     scenarios use (sample: agree on 4.4% of rows — different quantities)

    uv run --group fraud python -m projects.fraud_anomaly_detection.analysis.graph_store_crosscheck
    uv run --group fraud python -m ... --store .../fraud_graph_v3.duckdb
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import pandas as pd

DEFAULT_STORE = Path("projects/fraud_anomaly_detection/data/graph/fraud_graph.duckdb")

# Hand-compiled SQL mirrors of register.yaml's triggers (engine semantics:
# NaN never matches; hours_between is exact elapsed, not boundary-counting).
# If the register changes, update these in the same change — check 1 is the
# proof the two stay equivalent.
SQL_TRIGGERS = {
    "ring_account_reuse": (
        "(epoch(feature_as_of_ts) - epoch(identity_created_time))/3600.0 <= 24 "
        "AND loan_amount > 100 AND prior_advances_on_bank_account_7d > 0"
    ),
    "ring_identity_burst": "users_on_bank_account_72h >= 3",
    "ring_shared_persistent_account": (
        "users_on_persistent_account_id_72h >= 2 "
        "AND NOT coalesce(is_joint = 1, FALSE)"
    ),
    "ring_device_burst": "users_on_device_id_72h >= 3",
}


def banner(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def check_fidelity(con: duckdb.DuckDBPyConnection, parquet: Path) -> None:
    banner(f"CHECK 0 — store vs parquet fidelity ({parquet})")
    n_store = con.execute("SELECT count(*) FROM advances").fetchone()[0]
    n_pq, overlap = con.execute(
        "SELECT (SELECT count(*) FROM read_parquet(?)), "
        "(SELECT count(*) FROM advances a JOIN read_parquet(?) p USING (advance_id))",
        [str(parquet), str(parquet)],
    ).fetchone()
    ok = n_store == n_pq == overlap
    print(f"  rows: store={n_store:,} parquet={n_pq:,} id-overlap={overlap:,}"
          f"  {'OK' if ok else '*** MISMATCH ***'}")


def check_engine_vs_sql(con: duckdb.DuckDBPyConnection, base: pd.DataFrame) -> None:
    banner("CHECK 1 — scenario engine (pandas) vs hand-compiled SQL (exact match required)")
    from projects.fraud_anomaly_detection.scenarios import assign

    flags = assign(base)
    for name, where in SQL_TRIGGERS.items():
        n_engine = int(flags[f"scenario_{name}"].sum())
        n_sql = con.execute(f"SELECT count(*) FROM advances WHERE {where}").fetchone()[0]
        print(f"  {name:34s} engine={n_engine:7,}  duckdb={n_sql:7,}  "
              f"{'OK' if n_engine == n_sql else '*** MISMATCH ***'}")
    union = " OR ".join(f"({w})" for w in SQL_TRIGGERS.values())
    n_any = con.execute(f"SELECT count(*) FROM advances WHERE {union}").fetchone()[0]
    n_any_engine = int(flags["scenario_any"].sum())
    print(f"  {'scenario_any (union)':34s} engine={n_any_engine:7,}  duckdb={n_any:7,}  "
          f"{'OK' if n_any_engine == n_any else '*** MISMATCH ***'}")


def check_data_qa(con: duckdb.DuckDBPyConnection) -> None:
    banner("CHECK 2 — data QA: labels, maturity, window monotonicity")
    q = lambda s: con.execute(s).fetchone()[0]  # noqa: E731
    n = q("SELECT count(*) FROM advances")
    print(f"  dpd45=1 but mature_d45=0 (label before maturity): "
          f"{q('SELECT count(*) FROM advances WHERE label_gross_dpd45=1 AND label_mature_d45=0'):,}")
    print(f"  immature rows (mature_d45=0, censored tail): "
          f"{q('SELECT count(*) FROM advances WHERE label_mature_d45=0'):,} / {n:,}")
    for a, b in [("users_on_device_id_72h", "users_on_device_id_7d"),
                 ("users_on_device_id_7d", "users_on_device_id_30d"),
                 ("users_on_bank_account_72h", "users_on_bank_account_7d"),
                 ("users_on_bank_account_7d", "users_on_bank_account_30d"),
                 ("users_on_persistent_account_id_72h", "users_on_persistent_account_id_7d")]:
        n_bad = q(f"SELECT count(*) FROM advances WHERE {a} > {b}")
        print(f"  window monotonicity {a} <= {b}: "
              f"{'OK' if n_bad == 0 else f'*** {n_bad} violations ***'}")
    print(f"  users with >1 identity_created_time: "
          f"{q('SELECT count(*) FROM (SELECT user_id FROM advances GROUP BY 1 HAVING count(DISTINCT identity_created_time) > 1)'):,}")
    print(f"  identity created AFTER advance ts (negative age): "
          f"{q('SELECT count(*) FROM advances WHERE identity_created_time > feature_as_of_ts'):,}")


def check_graph_recount(con: duckdb.DuckDBPyConnection) -> None:
    banner("CHECK 3 — graph recount of *_72h vs stored warehouse column")
    print("  (graph>stored = real bug; graph<stored = the advance-grain blind spot,\n"
          "   should shrink to ~0 once link-grain edges land)")
    # identity timestamps from the users table when the store carries them
    # (covers link-only users); fall back to the advances snapshot otherwise
    user_cols = {r[0] for r in con.execute("DESCRIBE users").fetchall()}
    uid_sql = (
        "SELECT user_id, identity_created_time AS ict FROM users"
        if "identity_created_time" in user_cols
        else "SELECT user_id, min(identity_created_time) AS ict FROM advances GROUP BY 1"
    )
    for etype, anchor_col, stored_col, thr in [
        ("device", "device_id", "users_on_device_id_72h", 3),
        ("bank", "bank_account_key", "users_on_bank_account_72h", 3),
    ]:
        r = con.execute(f"""
            WITH uid AS ({uid_sql}),
            anchors AS (
                SELECT advance_id, feature_as_of_ts AS ts,
                       CAST({anchor_col} AS VARCHAR) AS ev, {stored_col} AS stored
                FROM advances WHERE {anchor_col} IS NOT NULL),
            e AS (SELECT DISTINCT user_id, entity_value, ts FROM edges
                  WHERE entity_type = '{etype}'),
            rc AS (
                SELECT a.advance_id, a.stored,
                       count(DISTINCT CASE WHEN u.ict >= a.ts - INTERVAL 72 HOUR
                                            AND u.ict <= a.ts AND e.ts <= a.ts
                                           THEN e.user_id END) AS graph_asof
                FROM anchors a
                LEFT JOIN e ON e.entity_value = a.ev
                LEFT JOIN uid u ON u.user_id = e.user_id
                GROUP BY 1, 2)
            SELECT count(*),
                   sum(CASE WHEN graph_asof = stored THEN 1 ELSE 0 END),
                   sum(CASE WHEN graph_asof < stored THEN 1 ELSE 0 END),
                   sum(CASE WHEN graph_asof > stored THEN 1 ELSE 0 END),
                   sum(CASE WHEN stored >= {thr} THEN 1 ELSE 0 END),
                   sum(CASE WHEN graph_asof >= {thr} THEN 1 ELSE 0 END)
            FROM rc
        """).fetchone()
        n, eq, lower, higher, stored_t, graph_t = (int(x) for x in r)
        print(f"  {etype:8s} n={n:,} | equal={eq:,} ({eq / n:.1%}) | graph<stored={lower:,}"
              f" | graph>stored={higher:,}{'' if higher == 0 else ' *** INVESTIGATE ***'}")
        print(f"           trigger >={thr}: stored={stored_t:,}  graph={graph_t:,}"
              f"  (graph sees {graph_t / stored_t:.0%})" if stored_t else "")


def check_window_semantics(con: duckdb.DuckDBPyConnection) -> None:
    banner("CHECK 4 — 'past 72h': edge-ts window vs identity-creation window")
    n, eq, stored_t, edge_t, both = (int(x) for x in con.execute("""
        WITH anchors AS (
            SELECT advance_id, feature_as_of_ts AS ts,
                   CAST(device_id AS VARCHAR) AS ev, users_on_device_id_72h AS stored
            FROM advances WHERE device_id IS NOT NULL),
        e AS (SELECT DISTINCT user_id, entity_value, ts FROM edges
              WHERE entity_type = 'device'),
        rc AS (
            SELECT a.advance_id, a.stored,
                   count(DISTINCT CASE WHEN e.ts >= a.ts - INTERVAL 72 HOUR
                                        AND e.ts <= a.ts THEN e.user_id END) AS ew
            FROM anchors a LEFT JOIN e ON e.entity_value = a.ev
            GROUP BY 1, 2)
        SELECT count(*),
               sum(CASE WHEN ew = stored THEN 1 ELSE 0 END),
               sum(CASE WHEN stored >= 3 THEN 1 ELSE 0 END),
               sum(CASE WHEN ew >= 3 THEN 1 ELSE 0 END),
               sum(CASE WHEN ew >= 3 AND stored >= 3 THEN 1 ELSE 0 END)
        FROM rc
    """).fetchone())
    print(f"  device: edge-ts window == stored column on {eq:,}/{n:,} rows ({eq / n:.1%})"
          f" — different definitions, do NOT use edge-ts for scenario replication")
    print(f"  trigger >=3: stored={stored_t:,}  edge-ts-window={edge_t:,}  overlap={both:,}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--store", type=Path, default=DEFAULT_STORE)
    ap.add_argument("--parquet", type=Path, default=None,
                    help="optional source parquet for the fidelity check")
    args = ap.parse_args()
    if not args.store.exists():
        raise SystemExit(f"store not found: {args.store} — build it first")

    con = duckdb.connect(str(args.store), read_only=True)
    try:
        if args.parquet is not None:
            check_fidelity(con, args.parquet)
        base = con.execute("SELECT * FROM advances").df()
        check_engine_vs_sql(con, base)
        check_data_qa(con)
        check_graph_recount(con)
        check_window_semantics(con)
    finally:
        con.close()
    print("\nDone. Any '*** MISMATCH/INVESTIGATE ***' line is a real setup error;"
          " graph<stored gaps are the documented advance-grain blind spot.")


if __name__ == "__main__":
    main()
