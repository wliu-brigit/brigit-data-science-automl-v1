"""Load parameterized graph views from the store.

The store is lossless; THIS is where opinions are applied — which layers,
which degree cap, which time slice, which metadata, which scenario register.
Returns an igraph bipartite multigraph: user vertices (kind='user') and
entity vertices (kind=entity_type), parallel edges kept with etype/ts attrs.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import igraph as ig
import pandas as pd

from projects.fraud_anomaly_detection.graph.build import ENTITY_COLS, USER_ID

# email is near-noise (max 6 users even on full v3) and raw IP is a
# NAT/household junk generator (v1 learning) — stored, but opt-in.
DEFAULT_LAYERS: tuple[str, ...] = ("device", "bank", "persistent", "phone", "address")
DEFAULT_NODE_ATTRS: tuple[str, ...] = (
    "is_fraud", "label_gross_dpd45", "label_mature_d45",
    "is_neobank_high_risk_institution",
)


def _read_store(store: Path | str, sql: str, params: list | None = None) -> pd.DataFrame:
    with duckdb.connect(str(store), read_only=True) as con:
        return con.execute(sql, params or []).df()


EDGE_SOURCES: tuple[str, ...] = ("advance", "link")


def load_graph(
    store: Path | str,
    base: pd.DataFrame | None = None,
    layers: tuple[str, ...] = DEFAULT_LAYERS,
    degree_cap: int | None = None,
    as_of: pd.Timestamp | None = None,
    window: tuple[pd.Timestamp, pd.Timestamp] | None = None,
    node_attrs: tuple[str, ...] = DEFAULT_NODE_ATTRS,
    scenarios: bool = True,
    register_path: Path | str | None = None,
    sources: tuple[str, ...] = EDGE_SOURCES,
) -> ig.Graph:
    """One opinionated view of the stored graph, as an igraph multigraph.

    base defaults to the store's own `advances` snapshot (the file is
    self-contained); pass a DataFrame to override. degree_cap drops entity
    vertices whose distinct-user count WITHIN THIS VIEW exceeds the cap
    (users always stay). scenarios=True runs the register (the bound one, or
    `register_path`) against `base` NOW and attaches user-level
    scenario_<name> / scenario_any flags — never persisted, always current.
    sources picks the edge provenance: 'advance' (an advance happened) and/or
    'link' (the user touched the entity — includes advance-less users).
    """
    if not layers:
        raise ValueError("layers must name at least one entity type")
    unknown = set(layers) - set(ENTITY_COLS)
    if unknown:
        raise ValueError(f"unknown layer(s) {sorted(unknown)}; expected {sorted(ENTITY_COLS)}")
    if not sources:
        raise ValueError("sources must name at least one edge source")
    bad_sources = set(sources) - set(EDGE_SOURCES)
    if bad_sources:
        raise ValueError(f"unknown source(s) {sorted(bad_sources)}; expected {sorted(EDGE_SOURCES)}")

    conds = [
        "entity_type IN (" + ", ".join("?" * len(layers)) + ")",
        "source IN (" + ", ".join("?" * len(sources)) + ")",
    ]
    params: list = list(layers) + list(sources)
    if as_of is not None:
        conds.append("ts <= ?")
        params.append(as_of)
    if window is not None:
        conds.append("ts >= ? AND ts <= ?")
        params.extend([window[0], window[1]])
    where = " AND ".join(conds)

    if degree_cap is not None:
        # Rewrite as subquery for DuckDB compatibility
        cap_params = params + [degree_cap]
        edges = _read_store(
            store,
            f"SELECT advance_id, user_id, entity_type, entity_value, ts, source "
            f"FROM (SELECT *, count(DISTINCT user_id) OVER "
            f"(PARTITION BY entity_type, entity_value) AS du FROM edges WHERE {where}) "
            f"WHERE du <= ?",
            cap_params,
        )
    else:
        edges = _read_store(
            store,
            f"SELECT advance_id, user_id, entity_type, entity_value, ts, source "
            f"FROM edges WHERE {where}",
            params,
        )

    users = _read_store(store, "SELECT user_id FROM users ORDER BY 1")
    if base is None:
        # At full scale, callers loading multiple views should read the snapshot
        # once and pass base= explicitly to avoid re-reading per call.
        base = _read_store(store, "SELECT * FROM advances")

    user_names = ("user:" + users[USER_ID].astype(str)).tolist()
    ent_ids = edges["entity_type"] + ":" + edges["entity_value"]
    ent_names = sorted(set(ent_ids))

    g = ig.Graph()
    g.add_vertices(user_names + ent_names)
    g.vs["kind"] = ["user"] * len(user_names) + [n.split(":", 1)[0] for n in ent_names]
    g.vs["raw_id"] = [n.split(":", 1)[1] for n in user_names + ent_names]

    index = {name: i for i, name in enumerate(g.vs["name"])}
    src = ("user:" + edges["user_id"].astype(str)).map(index)
    dst = ent_ids.map(index)
    g.add_edges(list(zip(src, dst)))
    g.es["etype"] = edges["entity_type"].tolist()
    g.es["ts"] = list(edges["ts"])
    g.es["source"] = edges["source"].tolist()

    flags = pd.DataFrame(index=base.index)
    if scenarios:
        if register_path is not None:
            from projects.fraud_anomaly_detection.scenarios import engine

            register = engine.load_register(register_path)
            flags = engine.evaluate(base, register.scenarios)
        else:
            from projects.fraud_anomaly_detection.scenarios import assign

            flags = assign(base)

    missing = [col for col in node_attrs if col not in base.columns]
    if missing:
        raise ValueError(f"node_attrs not found in base: {sorted(missing)}")
    per_user = pd.concat([base[[USER_ID]], base[list(node_attrs)], flags], axis=1)
    per_user[USER_ID] = per_user[USER_ID].astype(str)
    agg = per_user.groupby(USER_ID).max()  # labels/flags: any advance counts
    labels = [n.split(":", 1)[1] for n in user_names]
    for col in agg.columns:
        fill = False if agg[col].dtype == bool else 0
        values = agg[col].reindex(labels).fillna(fill)
        g.vs.select(range(len(user_names)))[col] = values.tolist()
    return g
