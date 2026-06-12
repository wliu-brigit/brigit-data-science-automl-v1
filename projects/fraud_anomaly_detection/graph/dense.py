"""Dense-block mining on the bipartite user<->entity graph (Fraudar-style).

After Hooi et al., "FRAUDAR: Bounding Graph Fraud in the Face of Camouflage"
(KDD 2016): repeatedly peel the node of minimum weighted degree, tracking the
best average-weighted-degree block along the way; peel that block off and
repeat for the next one. The published algorithm — an exhaustive enumeration
of suspicious groups, not a hypothesis test.

Entity (column) weights 1/log(deg + 5) are the camouflage resistance: a
globally popular entity (NAT device, big institution) contributes almost
nothing per edge, so shared infrastructure cannot outscore a small ring
webbed through scarce resources. Weights are fixed from the full graph, per
the paper.

Snapshot semantics (review/clawback) like the other queues; any rule derived
from a block goes through asof.leakfree_features before precision is quoted.
Pure-Python peeling is O(E log V) — fine for the sample; at v3 scale (~14M
edges) expect minutes, and move to arrays/scipy if it grinds.
"""

from __future__ import annotations

import heapq
import math
from pathlib import Path

import duckdb
import pandas as pd

from projects.fraud_anomaly_detection.graph.load import DEFAULT_LAYERS
from projects.fraud_anomaly_detection.graph.queries import _check_layers


def _peel(
    adj_user: dict[str, set[str]],
    adj_ent: dict[str, set[str]],
    weight: dict[str, float],
) -> tuple[set[str], set[str], float]:
    """One greedy peel: returns (best users, best entities, best avg score)."""
    deg: dict[str, float] = {}
    for user, ents in adj_user.items():
        deg[user] = sum(weight[e] for e in ents)
    for ent, users in adj_ent.items():
        deg[ent] = weight[ent] * len(users)

    f = sum(deg[u] for u in adj_user)  # total edge weight, counted user-side
    n = len(adj_user) + len(adj_ent)
    heap: list[tuple[float, str]] = [(d, node) for node, d in deg.items()]
    heapq.heapify(heap)

    removed: set[str] = set()
    order: list[str] = []
    best_g, best_step = (f / n if n else 0.0), 0
    while n:
        d, node = heapq.heappop(heap)
        if node in removed or d != deg[node]:
            continue  # stale heap entry
        removed.add(node)
        order.append(node)
        if node in adj_user:
            for ent in adj_user[node]:
                f -= weight[ent]
                deg[ent] -= weight[ent]
                adj_ent[ent].discard(node)
                heapq.heappush(heap, (deg[ent], ent))
            adj_user.pop(node)
        else:
            for user in adj_ent[node]:
                f -= weight[node]
                deg[user] -= weight[node]
                adj_user[user].discard(node)
                heapq.heappush(heap, (deg[user], user))
            adj_ent.pop(node)
        n -= 1
        g = f / n if n else 0.0
        if g > best_g:
            best_g, best_step = g, len(order)

    survivors = set(deg) - set(order[:best_step])
    users = {x for x in survivors if not x.startswith("__ent__")}
    ents = {x[7:] for x in survivors if x.startswith("__ent__")}
    return users, ents, best_g


def dense_blocks(
    store: Path | str,
    layers: tuple[str, ...] = DEFAULT_LAYERS,
    top_k: int = 3,
    min_users: int = 3,
) -> pd.DataFrame:
    """Top suspicious dense blocks, peeled off one after another (disjoint).

    Returns one row per block: score (avg weighted degree), n_users,
    n_entities, n_types, entity_types, user_ids — same census shape as
    queries.components so the queues can consume either.
    """
    _check_layers(layers)
    with duckdb.connect(str(store), read_only=True) as con:
        pairs = con.execute(
            "SELECT DISTINCT user_id, entity_type,"
            " entity_type || ':' || entity_value AS ent FROM edges"
            f" WHERE entity_type IN ({', '.join('?' * len(layers))})",
            list(layers),
        ).df()

    # entity node ids are prefixed so user/entity never collide in one dict
    adj_user: dict[str, set[str]] = {}
    adj_ent: dict[str, set[str]] = {}
    for user, _etype, ent in pairs.itertuples(index=False):
        ent_node = f"__ent__{ent}"
        adj_user.setdefault(str(user), set()).add(ent_node)
        adj_ent.setdefault(ent_node, set()).add(str(user))
    weight = {ent: 1.0 / math.log(len(users) + 5) for ent, users in adj_ent.items()}

    rows = []
    block_id = 0
    while len(rows) < top_k and adj_user:
        # _peel consumes its adjacency copies; survivors are removed from the
        # originals afterwards so the next iteration sees a disjoint graph
        users, ents, score = _peel(
            {u: set(es) for u, es in adj_user.items()},
            {e: set(us) for e, us in adj_ent.items()},
            weight,
        )
        if not users and not ents:
            break
        for user in users:
            for ent_node in adj_user.pop(user, set()):
                adj_ent.get(ent_node, set()).discard(user)
        for ent in ents:
            ent_node = f"__ent__{ent}"
            for user in adj_ent.pop(ent_node, set()):
                adj_user.get(user, set()).discard(ent_node)
        adj_user = {u: es for u, es in adj_user.items() if es}
        adj_ent = {e: us for e, us in adj_ent.items() if us}
        if len(users) < min_users:
            continue  # peeled off, but too small to queue
        etypes = sorted({e.split(":", 1)[0] for e in ents})
        rows.append({
            "block_id": block_id,
            "score": round(score, 4),
            "n_users": len(users),
            "n_entities": len(ents),
            "n_types": len(etypes),
            "entity_types": ",".join(etypes),
            "user_ids": ",".join(sorted(users)),
        })
        block_id += 1

    out = pd.DataFrame(rows, columns=["block_id", "score", "n_users", "n_entities",
                                      "n_types", "entity_types", "user_ids"])
    return out.sort_values("score", ascending=False, kind="stable").reset_index(drop=True)
