"""DuckPGQ probe — scoped experiment, nothing depends on it (spec: probe only).

Question under test: can SQL/PGQ MATCH answer "users within 3 hops of a
fraud user" on our store, and does it agree with queries.near_flagged?
Builds single-key vertex/edge tables (DuckPGQ wants simple keys), creates a
property graph, runs the hop query, compares user sets and wall time.

    uv run --group fraud python -m projects.fraud_anomaly_detection.analysis.graph_pgq_probe
"""

from __future__ import annotations

import time
from pathlib import Path

import duckdb

from projects.fraud_anomaly_detection.graph.load import DEFAULT_LAYERS, load_graph
from projects.fraud_anomaly_detection.graph.queries import near_flagged

STORE = Path("projects/fraud_anomaly_detection/data/graph/fraud_graph.duckdb")

# Syntax variants tried (in order) for GRAPH_TABLE quantifier.
# DuckPGQ support is partial; we record what failed before giving up.
_ATTEMPTS: list[str] = []


def _pgq_query_v1(con: duckdb.DuckDBPyConnection) -> tuple[set[str], float]:
    """Attempt 1: standard quantifier {1,6} undirected bipartite path."""
    q = """
        SELECT DISTINCT t.id FROM GRAPH_TABLE (fraud_pg
            MATCH (u:account)-[e:touches]-{1,6}(f:account)
            WHERE f.is_fraud = 1 AND u.is_fraud = 0
            COLUMNS (u.id)
        ) t(id)
    """
    _ATTEMPTS.append("v1: -[e:touches]-{1,6} undirected")
    t0 = time.perf_counter()
    rows = con.execute(q).fetchall()
    elapsed = time.perf_counter() - t0
    return {r[0] for r in rows}, elapsed


def _pgq_query_v2(con: duckdb.DuckDBPyConnection) -> tuple[set[str], float]:
    """Attempt 2: quantifier with space {1, 6}."""
    q = """
        SELECT DISTINCT t.id FROM GRAPH_TABLE (fraud_pg
            MATCH (u:account)-[e:touches]-{1, 6}(f:account)
            WHERE f.is_fraud = 1 AND u.is_fraud = 0
            COLUMNS (u.id)
        ) t(id)
    """
    _ATTEMPTS.append("v2: -[e:touches]-{1, 6} with space")
    t0 = time.perf_counter()
    rows = con.execute(q).fetchall()
    elapsed = time.perf_counter() - t0
    return {r[0] for r in rows}, elapsed


def _pgq_query_v3(con: duckdb.DuckDBPyConnection) -> tuple[set[str], float]:
    """Attempt 3: any-length wildcard -[e]-* then filter hop depth separately."""
    q = """
        SELECT DISTINCT t.id FROM GRAPH_TABLE (fraud_pg
            MATCH (u:account)-[e:touches]*-(f:account)
            WHERE f.is_fraud = 1 AND u.is_fraud = 0
            COLUMNS (u.id)
        ) t(id)
    """
    _ATTEMPTS.append("v3: -[e:touches]*- wildcard (unbounded, filtered post)")
    t0 = time.perf_counter()
    rows = con.execute(q).fetchall()
    elapsed = time.perf_counter() - t0
    return {r[0] for r in rows}, elapsed


def main() -> None:  # noqa: C901
    con = duckdb.connect(str(STORE))  # writable: probe creates pg_* tables
    try:
        # ── Extension ──────────────────────────────────────────────────────────
        try:
            con.execute("INSTALL duckpgq FROM community")
            con.execute("LOAD duckpgq")
        except Exception as exc:  # noqa: BLE001 — verdict, not control flow
            print(f"VERDICT: SKIPPED — extension unavailable: {exc}")
            return

        # ── Build vertex / edge tables + property graph ────────────────────────
        try:
            layer_list = ", ".join(f"'{layer}'" for layer in DEFAULT_LAYERS)
            con.execute("""
                CREATE OR REPLACE TABLE pg_users AS
                SELECT u.user_id AS id, coalesce(max(a.is_fraud), 0) AS is_fraud
                FROM users u LEFT JOIN advances a USING (user_id) GROUP BY 1;
            """)
            con.execute(f"""
                CREATE OR REPLACE TABLE pg_entities AS
                SELECT DISTINCT entity_type || ':' || entity_value AS id
                FROM edges WHERE entity_type IN ({layer_list});
            """)
            con.execute(f"""
                CREATE OR REPLACE TABLE pg_edges AS
                SELECT DISTINCT user_id AS src, entity_type || ':' || entity_value AS dst
                FROM edges WHERE entity_type IN ({layer_list});
            """)

            # ── Property graph — drop first to survive re-runs ─────────────────
            try:
                con.execute("DROP PROPERTY GRAPH IF EXISTS fraud_pg")
            except Exception:  # noqa: BLE001 — older builds may not support DROP PG
                pass

            con.execute("""
                CREATE PROPERTY GRAPH fraud_pg
                VERTEX TABLES (
                    pg_users PROPERTIES (id, is_fraud) LABEL account,
                    pg_entities PROPERTIES (id) LABEL resource
                )
                EDGE TABLES (
                    pg_edges SOURCE KEY (src) REFERENCES pg_users (id)
                             DESTINATION KEY (dst) REFERENCES pg_entities (id)
                             LABEL touches
                );
            """)
        except Exception as exc:  # noqa: BLE001
            print(f"VERDICT: SKIPPED — property-graph setup failed: {exc}")
            return

        # ── PGQ hop query — try up to 3 syntax variants ────────────────────────
        pgq_users: set[str] = set()
        pgq_secs: float = 0.0
        pgq_ok = False
        last_exc: Exception | None = None

        for attempt_fn in (_pgq_query_v1, _pgq_query_v2, _pgq_query_v3):
            try:
                pgq_users, pgq_secs = attempt_fn(con)
                pgq_ok = True
                break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                print(f"  [attempt {len(_ATTEMPTS)}] {_ATTEMPTS[-1]} — FAILED: {exc}")

        if not pgq_ok:
            tried = "; ".join(_ATTEMPTS)
            print(f"VERDICT: SKIPPED — syntax/unsupported after {len(_ATTEMPTS)} attempts"
                  f" ({tried}): {last_exc}")
            return

        print(f"SQL/PGQ 3-user-hop neighbours of fraud: {len(pgq_users):,}"
              f" users in {pgq_secs:.2f}s"
              f"  [syntax: {_ATTEMPTS[-1]}]")

        # ── igraph baseline ────────────────────────────────────────────────────
        con.close()  # load_graph opens read_only on the same file; DuckDB forbids mixed configs
        t0 = time.perf_counter()
        g = load_graph(STORE, scenarios=False, node_attrs=("is_fraud",))
        ig_users = set(near_flagged(g, flag="is_fraud", max_hops=3)["user_id"])
        ig_secs = time.perf_counter() - t0
        print(f"igraph near_flagged equivalent:        {len(ig_users):,}"
              f" users in {ig_secs:.2f}s (incl. load)")

        # ── Verdict ────────────────────────────────────────────────────────────
        if pgq_users == ig_users:
            print(f"VERDICT: AGREES — identical user sets; pgq {pgq_secs:.2f}s"
                  f" vs igraph {ig_secs:.2f}s")
        else:
            only_pgq = pgq_users - ig_users
            only_ig = ig_users - pgq_users
            # Print a few examples to aid investigation
            sample_pgq = sorted(only_pgq)[:5]
            sample_ig = sorted(only_ig)[:5]
            print(f"  only-pgq sample (up to 5): {sample_pgq}")
            print(f"  only-igraph sample (up to 5): {sample_ig}")
            print(f"VERDICT: DISAGREES — only-pgq {len(only_pgq)},"
                  f" only-igraph {len(only_ig)} (investigate before trusting pgq)")
    finally:
        con.close()


if __name__ == "__main__":
    main()
