"""Question-level helpers on a loaded graph view (or directly on the store).

This is the query surface (the engine has no query language): each function
answers one analysis question. Graph inputs are bipartite user<->entity views
from load.load_graph; `hops` always means USER-hops (2 bipartite steps).
project_users / hub_report run as SQL on the store: set math is the
database's home turf, traversal is the graph's.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import igraph as ig
import pandas as pd

from projects.fraud_anomaly_detection.graph.build import ENTITY_COLS
from projects.fraud_anomaly_detection.graph.load import DEFAULT_LAYERS


def _flagged_indices(g: ig.Graph, flag: str) -> list[int]:
    return [v.index for v in g.vs if v["kind"] == "user" and bool(v[flag])]


def near_flagged(g: ig.Graph, flag: str = "is_fraud", max_hops: int = 3) -> pd.DataFrame:
    """Users within max_hops USER-hops of any flagged user (flagged excluded).

    Single multi-source BFS via a virtual vertex attached to every flagged
    user: distance(virtual -> target) = 1 + bipartite-steps, and one
    user-hop = 2 bipartite steps, so hops = (d - 1) // 2.
    """
    seeds = _flagged_indices(g, flag)
    if not seeds:
        return pd.DataFrame(columns=["user_id", "hops", "nearest_flagged"])
    seed_set = set(seeds)
    gv = g.copy()
    virtual = gv.add_vertex(name="__virtual__", kind="__virtual__")
    gv.add_edges([(virtual.index, s) for s in seeds])
    dist = gv.distances(source=[virtual.index])[0]

    rows = []
    for v in g.vs:
        if v["kind"] != "user" or v.index in seed_set:
            continue
        d = dist[v.index]
        hops = (int(d) - 1) // 2 if d != float("inf") else None
        if hops is not None and 1 <= hops <= max_hops:
            path = gv.get_shortest_paths(virtual.index, to=v.index)[0]
            rows.append((v["raw_id"], hops, gv.vs[path[1]]["raw_id"]))
    return pd.DataFrame(rows, columns=["user_id", "hops", "nearest_flagged"])


def components(g: ig.Graph, flag: str = "is_fraud") -> pd.DataFrame:
    """Connected-component census with the multi-type density discriminator.

    n_types counts distinct ENTITY types in the component — the v1 finding:
    small components webbed across >=2 types are the ring signature.
    """
    comps = g.connected_components()
    rows = []
    for comp_id, members in enumerate(comps):
        kinds = [g.vs[i]["kind"] for i in members]
        users = [g.vs[i] for i, k in zip(members, kinds) if k == "user"]
        etypes = sorted({k for k in kinds if k != "user"})
        rows.append({
            "comp_id": comp_id,
            "n_users": len(users),
            "n_entities": len(members) - len(users),
            "entity_types": ",".join(etypes),
            "n_types": len(etypes),
            "n_flagged": sum(bool(v[flag]) for v in users),
            "user_ids": ",".join(sorted(v["raw_id"] for v in users)),
        })
    return pd.DataFrame(rows)


def ring(g: ig.Graph, user_id: str, hops: int = 2) -> ig.Graph:
    """Ego subgraph around a user, out to `hops` user-hops (deep-dive unit)."""
    center = g.vs.find(name=f"user:{user_id}")
    member_ids = g.neighborhood(center.index, order=2 * hops)
    return g.induced_subgraph(member_ids)


def project_users(
    store: Path | str,
    layers: tuple[str, ...] = DEFAULT_LAYERS,
    degree_cap: int | None = 20,
    as_of: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Weighted user<->user projection: n_shared entities + n_types distinct types.

    ALWAYS think before lifting the cap: one 136-user device alone emits
    9,180 pairs. Default cap matches the v1 ring-traversal finding (~20).
    """
    unknown = set(layers) - set(ENTITY_COLS)
    if unknown:
        raise ValueError(f"unknown layer(s) {sorted(unknown)}; expected {sorted(ENTITY_COLS)}")
    params: list = list(layers)
    time_cond = ""
    if as_of is not None:
        time_cond = " AND ts <= ?"
        params.append(as_of)
    cap_cond = ""
    if degree_cap is not None:
        cap_cond = (
            " AND (entity_type, entity_value) IN ("
            "SELECT entity_type, entity_value FROM entities WHERE n_users <= ?)"
        )
        params.append(degree_cap)

    sql = f"""
        WITH pairs AS (
            SELECT DISTINCT user_id, entity_type, entity_value FROM edges
            WHERE entity_type IN ({", ".join("?" * len(layers))}){time_cond}{cap_cond}
        )
        SELECT a.user_id AS user_a, b.user_id AS user_b,
               count(*) AS n_shared,
               count(DISTINCT a.entity_type) AS n_types
        FROM pairs a JOIN pairs b
          ON a.entity_type = b.entity_type AND a.entity_value = b.entity_value
         AND a.user_id < b.user_id
        GROUP BY 1, 2
    """
    with duckdb.connect(str(store), read_only=True) as con:
        return con.execute(sql, params).df()


def hub_report(
    store: Path | str,
    top_n: int = 20,
    layers: tuple[str, ...] = tuple(ENTITY_COLS),
) -> pd.DataFrame:
    """High-degree entities, NO cap — the bigger, the more interesting.

    The time axis separates fraud farms (many users in days, high attached
    fraud rate) from shared infrastructure (many users over years, base rate).
    """
    params: list = list(layers)
    sql = f"""
        WITH user_label AS (
            SELECT user_id, max(is_fraud) AS is_fraud FROM advances GROUP BY 1
        ), pairs AS (
            SELECT DISTINCT entity_type, entity_value, user_id FROM edges
            WHERE entity_type IN ({", ".join("?" * len(layers))})
        ), stats AS (
            SELECT entity_type, entity_value,
                   count(DISTINCT user_id) AS n_users,
                   count(*) AS n_edges,
                   date_diff('day', min(ts), max(ts)) AS span_days
            FROM edges
            WHERE entity_type IN ({", ".join("?" * len(layers))})
            GROUP BY 1, 2
        )
        SELECT s.*, round(s.n_users / greatest(s.span_days, 1), 3) AS users_per_day,
               round(avg(l.is_fraud), 3) AS fraud_user_rate
        FROM stats s
        JOIN pairs p USING (entity_type, entity_value)
        JOIN user_label l USING (user_id)
        GROUP BY ALL
        ORDER BY s.n_users DESC, s.entity_value
        LIMIT {int(top_n)}
    """
    with duckdb.connect(str(store), read_only=True) as con:
        return con.execute(sql, params + params).df()
