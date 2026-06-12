"""Snapshot discovery queues — ranked lists of UNFLAGGED users and entities.

Each function answers one standing discovery question and returns a frame
sorted so the top rows are the queue: who/what to review first, with the
evidence attached (ring id and shape, hop distances, accumulation velocity,
agreeing channels). Deliberately SNAPSHOT semantics — review and clawback
act on current knowledge about past cases, so hindsight is legitimate here.
Gate-grade, at-decision-moment measurement lives in asof.leakfree_features.

Flags are inputs, not opinions of this module: pass a graph whose user
vertices carry the flag attributes (load.load_graph scenario overlay), or a
user-level boolean Series for the SQL-side queues — the caller decides what
"already known" means (scenario register, confirmed-fraud label, both).
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import igraph as ig
import pandas as pd

from projects.fraud_anomaly_detection.graph.load import DEFAULT_LAYERS, load_graph
from projects.fraud_anomaly_detection.graph.queries import (
    components,
    hub_report,
    near_flagged,
    ppr_suspicion,
    project_users,
)

RING_CAP = 20  # the v1 traversal finding; callers can rebuild views with their own


def residual_ring_members(
    g: ig.Graph,
    flag: str = "scenario_any",
    min_users: int = 3,
    min_types: int = 2,
) -> pd.DataFrame:
    """Unflagged members of multi-type rings — guilt-by-association queue.

    Rings qualify by the v1 discriminator (small, dense, MULTI-entity-type);
    every unflagged member is returned with the ring's evidence. Sorted so
    rings with the most flagged co-members (strongest association) come first.
    """
    cc = components(g, flag=flag)
    rings = cc[(cc.n_users >= min_users) & (cc.n_types >= min_types)]
    flagged_users = {v["raw_id"] for v in g.vs
                     if v["kind"] == "user" and bool(v[flag])}
    rows = []
    for ring in rings.itertuples(index=False):
        for user in ring.user_ids.split(","):
            if user not in flagged_users:
                rows.append({
                    "user_id": user, "comp_id": ring.comp_id,
                    "ring_users": ring.n_users, "ring_types": ring.n_types,
                    "ring_flagged": ring.n_flagged,
                    "entity_types": ring.entity_types,
                })
    out = pd.DataFrame(rows, columns=["user_id", "comp_id", "ring_users",
                                      "ring_types", "ring_flagged", "entity_types"])
    return out.sort_values(["ring_flagged", "ring_types", "ring_users"],
                           ascending=False, kind="stable").reset_index(drop=True)


def bad_neighbours(
    g: ig.Graph,
    flags: tuple[str, ...] = ("scenario_any", "is_fraud"),
    max_hops: int = 2,
) -> pd.DataFrame:
    """Users near ANY flagged user but flagged by NONE of the given flags.

    One hops_to_<flag> column per flag (NaN = not within reach). Sorted by
    closest approach: distance is the evidence strength.
    """
    per_flag = {}
    for flag in flags:
        out = near_flagged(g, flag=flag, max_hops=max_hops)
        per_flag[flag] = out.set_index("user_id")["hops"]
    merged = pd.DataFrame(per_flag).rename(
        columns={flag: f"hops_to_{flag}" for flag in flags})
    flagged_any = {v["raw_id"] for v in g.vs if v["kind"] == "user"
                   and any(bool(v[flag]) for flag in flags)}
    merged = merged[~merged.index.isin(flagged_any)]
    merged["closest"] = merged.min(axis=1)
    merged = merged.sort_values("closest", kind="stable").drop(columns="closest")
    return merged.rename_axis("user_id").reset_index()


def suspicion_queue(
    g: ig.Graph,
    seed_flag: str = "is_fraud",
    exclude_flags: tuple[str, ...] = ("scenario_any", "is_fraud"),
    top_n: int = 100,
) -> pd.DataFrame:
    """Unflagged users ranked by diffused suspicion from the flagged seeds.

    PPR generalizes bad_neighbours: instead of the nearest-hop count, every
    path to every seed contributes, with principled decay. Seeds and users
    matching ANY exclude flag are dropped — the queue is what is left to look
    at, ordered by how much known-bad mass flows to them.
    """
    out = ppr_suspicion(g, flag=seed_flag)
    if not len(out):
        return out.drop(columns=["seeded"], errors="ignore")
    flagged = {v["raw_id"] for v in g.vs if v["kind"] == "user"
               and any(bool(v[flag]) for flag in exclude_flags)}
    out = out[~out.seeded & ~out.user_id.isin(flagged)]
    return out.drop(columns="seeded").head(top_n).reset_index(drop=True)


def emerging_farms(
    store: Path | str,
    user_flags: pd.Series,
    min_users: int = 5,
    top_n: int = 50,
    layers: tuple[str, ...] = tuple(DEFAULT_LAYERS),
) -> pd.DataFrame:
    """Entities accumulating users FAST that flags haven't caught up with.

    The farm signature is velocity (users/day over a short span); the queue
    score discounts hubs whose users are already flagged (known farms).
    """
    hubs = hub_report(store, top_n=max(top_n * 20, 500), layers=layers)
    hubs = hubs[hubs.n_users >= min_users]
    with duckdb.connect(str(store), read_only=True) as con:
        pairs = con.execute(
            "SELECT DISTINCT entity_type, entity_value, user_id FROM edges").df()
    pairs["user_id"] = pairs.user_id.astype(str)
    pairs["flagged"] = pairs.user_id.map(user_flags).fillna(False).astype(bool)
    coverage = (pairs.groupby(["entity_type", "entity_value"])["flagged"]
                .mean().rename("flagged_coverage"))
    out = hubs.merge(coverage, on=["entity_type", "entity_value"])
    out["score"] = out.users_per_day * (1.0 - out.flagged_coverage)
    return (out.sort_values("score", ascending=False, kind="stable")
            .head(top_n).reset_index(drop=True))


def multi_witness_pairs(
    store: Path | str,
    user_flags: pd.Series,
    min_types: int = 2,
    degree_cap: int | None = RING_CAP,
) -> pd.DataFrame:
    """User pairs linked through >= min_types INDEPENDENT channels, both unflagged.

    One shared device can be a roommate; device AND bank AND phone agreeing
    is a ring (the v1 multi-type lesson, at pair grain).
    """
    proj = project_users(store, degree_cap=degree_cap)
    proj = proj[proj.n_types >= min_types]
    flagged = user_flags[user_flags].index
    proj = proj[~proj.user_a.isin(flagged) & ~proj.user_b.isin(flagged)]
    return (proj.sort_values(["n_types", "n_shared"], ascending=False, kind="stable")
            .reset_index(drop=True))


def fresh_rings(
    store: Path | str,
    days: int = 7,
    min_users: int = 2,
    layers: tuple[str, ...] = DEFAULT_LAYERS,
    degree_cap: int | None = RING_CAP,
) -> pd.DataFrame:
    """Rings whose edges formed within the last `days` of the store's history.

    A windowed view of the same census — 'what connected up just now'. Window
    is anchored to the newest edge in the store (not wall clock), so it works
    on any snapshot.
    """
    with duckdb.connect(str(store), read_only=True) as con:
        [(max_ts,)] = con.execute("SELECT max(ts) FROM edges").fetchall()
    window = (pd.Timestamp(max_ts) - pd.Timedelta(days=days), pd.Timestamp(max_ts))
    g = load_graph(store, layers=layers, degree_cap=degree_cap,
                   window=window, node_attrs=("is_fraud",), scenarios=False)
    cc = components(g, flag="is_fraud")
    out = cc[cc.n_users >= min_users].copy()
    out["window_start"], out["window_end"] = window
    return (out.sort_values(["n_types", "n_users"], ascending=False, kind="stable")
            .reset_index(drop=True))
