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
) -> ig.Graph:
    """One opinionated view of the stored graph, as an igraph multigraph.

    base defaults to the store's own `advances` snapshot (the file is
    self-contained); pass a DataFrame to override. degree_cap drops entity
    vertices whose distinct-user count WITHIN THIS VIEW exceeds the cap
    (users always stay). scenarios=True runs the register (the bound one, or
    `register_path`) against `base` NOW and attaches user-level
    scenario_<name> / scenario_any flags — never persisted, always current.
    """
    unknown = set(layers) - set(ENTITY_COLS)
    if unknown:
        raise ValueError(f"unknown layer(s) {sorted(unknown)}; expected {sorted(ENTITY_COLS)}")

    conds = ["entity_type IN (" + ", ".join("?" * len(layers)) + ")"]
    params: list = list(layers)
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
            f"SELECT advance_id, user_id, entity_type, entity_value, ts "
            f"FROM (SELECT *, count(DISTINCT user_id) OVER "
            f"(PARTITION BY entity_type, entity_value) AS du FROM edges WHERE {where}) "
            f"WHERE du <= ?",
            cap_params,
        )
    else:
        edges = _read_store(
            store,
            f"SELECT advance_id, user_id, entity_type, entity_value, ts "
            f"FROM edges WHERE {where}",
            params,
        )

    users = _read_store(store, "SELECT user_id FROM users ORDER BY 1")
    if base is None:
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

    flags = pd.DataFrame(index=base.index)
    if scenarios:
        if register_path is not None:
            from projects.fraud_anomaly_detection.scenarios import engine

            register = engine.load_register(register_path)
            flags = engine.evaluate(base, register.scenarios)
        else:
            from projects.fraud_anomaly_detection.scenarios import assign

            flags = assign(base)

    # Filter node_attrs to only columns that exist in base
    available_attrs = [col for col in node_attrs if col in base.columns]
    per_user = pd.concat([base[[USER_ID]], base[available_attrs], flags], axis=1)
    agg = per_user.groupby(USER_ID).max()  # labels/flags: any advance counts
    for col in agg.columns:
        values = agg[col].reindex([n.split(":", 1)[1] for n in user_names])
        g.vs.select(range(len(user_names)))[col] = values.tolist()
    return g
