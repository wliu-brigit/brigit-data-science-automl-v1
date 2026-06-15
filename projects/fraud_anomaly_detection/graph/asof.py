"""Leak-free per-advance graph features: replay the store's edge stream in time.

The snapshot views (load.load_graph) answer "what does the graph look like at
time T" — right for investigation and review queues, where acting on current
knowledge about past cases is legitimate. THIS module answers the stricter
question a deployed rule or model faces: "what did the graph look like at THE
MOMENT of each advance, knowing only what was knowable then." Two disciplines
make it leak-free:

- **Strictly prior reads.** Advances replay in timestamp order; each one reads
  the graph BEFORE its own edges are added (read-then-add). Full timestamps,
  never day buckets — fraud rings burst intra-day (the v1 effort's worst bug).
- **Seeds activate at maturity, not at the event.** A bad outcome counts as a
  known-bad neighbour only from `expected_dpd45_date` onward — when it became
  KNOWABLE — and a user's own prior default is excluded (that's credit
  history, not a ring; the v1 north-star confound).

One pass over the stream is O(E α) via union-find — the graph is never
rebuilt per advance. Edges only ACCUMULATE here (cumulative prior graph);
sliding-window semantics live in the SQL features / windowed snapshot views.

Origin: rebuilt from the pruned v1 scripts (graph_discovery_sweep, git
`5ea6d3d`) as a consumer of the store instead of a dataset-pinned one-off.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import duckdb
import pandas as pd

from projects.fraud_anomaly_detection.graph.build import ENTITY_COLS
from projects.fraud_anomaly_detection.graph.load import DEFAULT_LAYERS
from projects.fraud_anomaly_detection.graph.queries import _check_layers

_LAYER_BIT = {etype: 1 << i for i, etype in enumerate(ENTITY_COLS)}


class _UnionFind:
    __slots__ = ("parent", "n_users", "type_mask", "n_seeds")

    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.n_users: dict[str, int] = {}
        self.type_mask: dict[str, int] = {}
        self.n_seeds: dict[str, int] = {}

    def add(self, node: str, *, user: bool, type_bit: int = 0) -> None:
        if node not in self.parent:
            self.parent[node] = node
            self.n_users[node] = 1 if user else 0
            self.type_mask[node] = type_bit
            self.n_seeds[node] = 0

    def find(self, node: str) -> str:
        parent = self.parent
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:
            parent[node], node = root, parent[node]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra
            self.n_users[ra] += self.n_users[rb]
            self.type_mask[ra] |= self.type_mask[rb]
            self.n_seeds[ra] += self.n_seeds[rb]


def leakfree_features(
    store: Path | str,
    layers: tuple[str, ...] = DEFAULT_LAYERS,
    degree_cap: int | None = 20,
    seed_label: str = "label_gross_dpd45",
    seed_activation: str = "expected_dpd45_date",
) -> pd.DataFrame:
    """Per-advance leak-free graph features, one row per distinct advance.

    Returns columns: advance_id, user_id, ts, comp_users (component user
    count at that moment, incoming user included), comp_types (distinct
    entity types webbing the component — the multi-type discriminator),
    nb_comp (OTHER matured-bad users in the component), nb_d1 (OTHER
    matured-bad users sharing an entity directly). degree_cap screens
    promiscuous entities from the replay (v1: 10-50 equivalent, 20 default).
    """
    _check_layers(layers)
    params: list = list(layers)
    cap_clause = ""
    if degree_cap is not None:
        # advance-edge-only on BOTH sides: the replay is per-advance, and the
        # cap must be computed over the same population it screens (link edges
        # would shift caps without ever entering the replay)
        cap_clause = (
            " AND (entity_type, entity_value) IN (SELECT entity_type, entity_value"
            " FROM (SELECT entity_type, entity_value, count(DISTINCT user_id) AS du"
            f" FROM edges WHERE source = 'advance'"
            f" AND entity_type IN ({', '.join('?' * len(layers))})"
            " GROUP BY 1, 2) WHERE du <= ?)"
        )
        params = params + list(layers) + [degree_cap]

    with duckdb.connect(str(store), read_only=True) as con:
        edges = con.execute(
            "SELECT advance_id, entity_type, entity_value FROM edges"
            f" WHERE source = 'advance'"
            f" AND entity_type IN ({', '.join('?' * len(layers))}){cap_clause}",
            params,
        ).df()
        scored = con.execute(
            f"""
            SELECT advance_id, min(user_id) AS user_id,
                   min(feature_as_of_ts) AS ts,
                   max({seed_label}) AS seed,
                   min({seed_activation}) AS seed_ts
            FROM advances GROUP BY 1
            """
        ).df()

    adv_entities: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for advance_id, etype, value in edges.itertuples(index=False):
        adv_entities[advance_id].append((f"{etype}:{value}", _LAYER_BIT[etype]))

    scored = scored.sort_values("ts", kind="stable").reset_index(drop=True)
    ts_int = scored["ts"].astype("int64").to_numpy()
    seed_ts = pd.to_datetime(scored["seed_ts"], errors="coerce").astype("int64").to_numpy()
    is_seed = scored["seed"].astype(float).to_numpy() == 1

    events: list[tuple[int, int, int]] = [(ts_int[i], 0, i) for i in range(len(scored))]
    for i in range(len(scored)):
        if is_seed[i] and seed_ts[i] != pd.Timestamp.min.value:  # NaT -> int64 min
            events.append((seed_ts[i], 1, i))
    events.sort()

    uf = _UnionFind()
    seed_touched: dict[str, set[str]] = {}
    seeded_users: set[str] = set()
    user_node = ("u:" + scored["user_id"].astype(str)).tolist()
    advance_ids = scored["advance_id"].tolist()
    comp_users = [0] * len(scored)
    comp_types = [0] * len(scored)
    nb_comp = [0] * len(scored)
    nb_d1 = [0] * len(scored)

    def add_advance(i: int) -> None:
        user = user_node[i]
        uf.add(user, user=True)
        for node, bit in adv_entities.get(advance_ids[i], ()):
            uf.add(node, user=False, type_bit=bit)
            uf.union(user, node)

    for _, kind, i in events:
        user = user_node[i]
        caps = adv_entities.get(advance_ids[i], ())
        if kind == 0:
            roots = {uf.find(node) for node, _ in caps if node in uf.parent}
            if user in uf.parent:
                roots.add(uf.find(user))
            in_own_comp = user in uf.parent and uf.find(user) in roots
            mask = 0
            for root in roots:
                mask |= uf.type_mask[root]
            for _, bit in caps:
                mask |= bit
            comp_users[i] = sum(uf.n_users[r] for r in roots) + (0 if in_own_comp else 1)
            comp_types[i] = bin(mask).count("1")
            self_seeded = 1 if (user in seeded_users and in_own_comp) else 0
            nb_comp[i] = sum(uf.n_seeds[r] for r in roots) - self_seeded
            direct = set()
            for node, _ in caps:
                direct |= seed_touched.get(node, set())
            direct.discard(user)
            nb_d1[i] = len(direct)
            add_advance(i)
        else:
            if user not in uf.parent:
                add_advance(i)
            for node, _ in caps:
                seed_touched.setdefault(node, set()).add(user)
            if user not in seeded_users:
                seeded_users.add(user)
                uf.n_seeds[uf.find(user)] += 1

    return pd.DataFrame({
        "advance_id": advance_ids,
        "user_id": scored["user_id"],
        "ts": scored["ts"],
        "comp_users": comp_users,
        "comp_types": comp_types,
        "nb_comp": nb_comp,
        "nb_d1": nb_d1,
    })
