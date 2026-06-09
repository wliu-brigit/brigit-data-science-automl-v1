"""Graph / entity-ring connected-component screen (dataset v1_76d3ad45).

THE QUESTION (TODO.md TIER 2): the locked ring scenarios are each a 1-HOP view
(how many users share THIS row's device / account / persistent-id). This screen
generalises that to the GRAPH: link every user to every resource it touches,
take connected components, and ask whether MULTI-HOP ring membership
(user1-device-user2-account-user3, where user1 and user3 share nothing directly)
buys never-paid precision the 1-hop edges miss.

Rung 1 of the agreed ladder (basic first, escalate only if it pays). Purely
STRUCTURAL features (component size / breadth / #resource-types) -- no labels in
the graph, so no maturity-lag trap. Distance-to-known-fraud / prior-bad-in-
component (rung 2) is deferred until this cut shows lift.

AS-OF DISCIPLINE (non-negotiable -- the trap that inflated the early device
screen). The graph for advance A at t_A is built STRICTLY PRIOR: only advances
with feature_as_of_ts in [t_A - W, start-of-day(t_A)) contribute edges; A's own
resources then connect A's user into that prior graph. A whole-snapshot graph is
massively leaky. Day-bucketed for speed: all advances on a day share one prior
union-find. Known approximations (refine in v2): (a) day granularity -- a 72h
window is treated as 3 prior days, and same-day-earlier bursts are NOT counted
(strictly-prior is conservative, never leaks); (b) Dec-1 history floor left-
censors early advances -> a 30-day warm-up is dropped from the screen.

    uv run python -m projects.fraud_anomaly_detection.analysis.graph_component_screen
"""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

DATASET_ID = "v1_76d3ad45"

# resource node types (NOT ip_address: NAT/households would merge thousands of
# unrelated users into one giant junk component -- actively harmful in a graph).
RESOURCE_COLS = {
    "device_id": "D",
    "bank_account_key": "B",
    "persistent_account_id": "P",
}
WINDOWS_DAYS = {"72h": 3, "7d": 7}  # day-bucketed; 72h ~ 3 prior days
WARMUP_DAYS = 30  # drop the left-censored head (Dec-1 floor)
_MISSING = {"", "none", "nan", "null", "nat", "0", "0-0", "none-none"}


# ── union-find with per-root component attributes (n_users, n_res, type mask) ─
class UnionFind:
    __slots__ = ("parent", "nu", "nr", "tm")

    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.nu: dict[str, int] = {}   # root -> # user nodes
        self.nr: dict[str, int] = {}   # root -> # resource nodes
        self.tm: dict[str, int] = {}   # root -> resource-type bitmask

    def add_user(self, x: str) -> None:
        if x not in self.parent:
            self.parent[x] = x
            self.nu[x], self.nr[x], self.tm[x] = 1, 0, 0

    def add_res(self, x: str, typebit: int) -> None:
        if x not in self.parent:
            self.parent[x] = x
            self.nu[x], self.nr[x], self.tm[x] = 0, 1, typebit

    def find(self, x: str) -> str:
        p = self.parent
        root = x
        while p[root] != root:
            root = p[root]
        while p[x] != root:
            p[x], x = root, p[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra
            self.nu[ra] += self.nu[rb]
            self.nr[ra] += self.nr[rb]
            self.tm[ra] |= self.tm[rb]


# ── graph feature build (as-of windowed component, EXACT intra-day ordering) ──
def component_features(
    day_idx: np.ndarray,
    order_rank: np.ndarray,
    user_node: list[str],
    res_nodes: list[list[tuple[str, int]]],
    window_days: int,
) -> dict[str, np.ndarray]:
    """Per-advance structural component features, strictly-prior windowed, with
    EXACT intra-day ordering (recovers same-day bursts that day-bucketing drops).

    For each day d: build a union-find over advances in days [d-window, d-1],
    then process day d's advances in (ts, advance_id) order. For each advance we
    read its component from the CURRENT graph (prior days + same-day-EARLIER
    advances), record it, and only THEN union the advance in -- so an advance
    never sees itself or any later advance (strictly prior, no leak), but it
    does see earlier same-day siblings (the burst).

    A's resources point into disjoint prior roots, so their user/res counts add;
    A's own user adds 1 unless it already sits in one of those roots.
    """
    n = len(user_node)
    n_users = np.ones(n, dtype=np.int32)
    n_res = np.zeros(n, dtype=np.int32)
    n_types = np.zeros(n, dtype=np.int32)

    by_day: dict[int, list[int]] = defaultdict(list)
    for i, d in enumerate(day_idx):
        by_day[int(d)].append(i)

    def _add_advance(uf: UnionFind, i: int) -> None:
        u = user_node[i]
        uf.add_user(u)
        for r, tb in res_nodes[i]:
            uf.add_res(r, tb)
            uf.union(u, r)

    for d in sorted(by_day):
        uf = UnionFind()
        for dd in range(d - window_days, d):
            for i in by_day.get(dd, ()):
                _add_advance(uf, i)

        for i in sorted(by_day[d], key=lambda j: order_rank[j]):
            u = user_node[i]
            roots = {uf.find(r) for r, _ in res_nodes[i] if r in uf.parent}
            pu = sum(uf.nu[rt] for rt in roots)
            pr = sum(uf.nr[rt] for rt in roots)
            tm = 0
            for rt in roots:
                tm |= uf.tm[rt]
            u_counted = u in uf.parent and uf.find(u) in roots
            own_new_res = sum(1 for r, _ in res_nodes[i] if r not in uf.parent)
            own_tm = 0
            for _, tb in res_nodes[i]:
                own_tm |= tb
            n_users[i] = pu + (0 if u_counted else 1)
            n_res[i] = pr + own_new_res
            n_types[i] = bin(tm | own_tm).count("1")
            _add_advance(uf, i)  # now visible to later same-day advances

    return {"comp_n_users": n_users, "comp_n_res": n_res, "comp_n_types": n_types}


# ── self-test: as-of windowing, multi-hop, AND intra-day burst (the guards) ───
_D, _B, _P = 1, 2, 4


def _self_test() -> None:
    # A1 U1{D1,B1} day0 t0; A2 U2{D1,B2} day0 t1 (SAME day, later);
    # A3 U3{D2,B1} day2; A4 U1{D1,B1} day5
    day = np.array([0, 0, 2, 5])
    rank = np.array([0, 1, 2, 3])  # global ts order
    users = ["U:1", "U:2", "U:3", "U:1"]
    res = [[("D:1", _D), ("B:1", _B)], [("D:1", _D), ("B:2", _B)],
           [("D:2", _D), ("B:1", _B)], [("D:1", _D), ("B:1", _B)]]
    f = component_features(day, rank, users, res, window_days=7)
    assert f["comp_n_users"][0] == 1, f["comp_n_users"][0]            # A1 alone
    # A2 same-day-later: sees A1 via D1 -> 2-user ring (v1 day-bucketing gave 1)
    assert f["comp_n_users"][1] == 2, f["comp_n_users"][1]
    # A3: B1 links U1-D1-U2-B2 -> U3 joins a 3-user device+account ring
    assert f["comp_n_users"][2] == 3, f["comp_n_users"][2]
    assert f["comp_n_types"][2] == 2, f["comp_n_types"][2]
    print("self-test OK (as-of + multi-hop + intra-day burst correct)\n")


# ── helpers ───────────────────────────────────────────────────────────────────
def _load_env() -> None:
    p = Path(".env")
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _node(prefix: str, val) -> str | None:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    s = str(val).strip()
    if s.lower() in _MISSING:
        return None
    return f"{prefix}:{s}"


def _row(label, mask, never, dpd, base_never, base_dpd) -> dict:
    n = int(mask.sum())
    if n == 0:
        return {"rule": label, "n": 0, "never%": float("nan"), "never_lift": float("nan"),
                "dpd45%": float("nan"), "dpd_lift": float("nan"), "n_never": 0}
    nv, dp = never[mask].mean(), dpd[mask].mean()
    return {"rule": label, "n": n, "n_never": int(never[mask].sum()),
            "never%": nv, "never_lift": nv / base_never if base_never else float("nan"),
            "dpd45%": dp, "dpd_lift": dp / base_dpd if base_dpd else float("nan")}


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    _self_test()
    _load_env()
    from automl.data.registry import load_dataset_by_id
    from automl.project.session import use_project
    from projects.fraud_anomaly_detection.scenarios import residual_mask

    sess = use_project("fraud_anomaly_detection", dry_run=False)
    df = load_dataset_by_id(DATASET_ID, session=sess).df
    print(f"loaded {len(df):,} rows\n")

    ts = pd.to_datetime(df["feature_as_of_ts"])
    ts_int = ts.astype("int64").to_numpy()
    day_idx = (ts.dt.normalize() - ts.dt.normalize().min()).dt.days.to_numpy().astype(int)
    # strict intra-day ordering: by timestamp, then advance_id as a stable tiebreak
    aid = df["advance_id"].astype(str).to_numpy()
    order_rank = np.lexsort((aid, ts_int)).argsort()

    typebit = {"device_id": _D, "bank_account_key": _B, "persistent_account_id": _P}
    user_node = [f"U:{u}" for u in df["user_id"].astype(str)]
    res_nodes: list[list[tuple[str, int]]] = []
    present = {c: df[c].to_numpy(object) for c in RESOURCE_COLS}
    for i in range(len(df)):
        rs = []
        for c, pre in RESOURCE_COLS.items():
            node = _node(pre, present[c][i])
            if node is not None:
                rs.append((node, typebit[c]))
        res_nodes.append(rs)

    # outcome masks (same construction as edge_precision_screen)
    res_m = residual_mask(df).to_numpy()
    mat = (df["label_mature_d45"].astype(float) == 1).to_numpy()
    dpd = (df["label_gross_dpd45"].astype(float) == 1).to_numpy()
    never = dpd & (df["label_repaid_current_snapshot"].astype(float) == 0).to_numpy()
    warm = day_idx >= WARMUP_DAYS
    keep = res_m & mat & warm
    never_k, dpd_k = never[keep], dpd[keep]
    base_never, base_dpd = never_k.mean(), dpd_k.mean()
    print(f"residual + mature + post-warmup population: {keep.sum():,} rows "
          f"(dropped first {WARMUP_DAYS}d)")
    print(f"base never-paid {base_never:.3%} | base DPD45 {base_dpd:.3%}\n")

    # 1-hop "already caught" set (the locked + screened block edges) -> for marginal
    def ge(col, t):
        return pd.to_numeric(df[col], errors="coerce").to_numpy() >= t if col in df else np.zeros(len(df), bool)
    caught_1hop = (
        ge("users_on_device_id_72h", 3) | ge("users_on_device_id_7d", 3)
        | ge("users_on_persistent_account_id_72h", 2) | ge("users_on_persistent_account_id_7d", 2)
        | ge("users_on_bank_account_72h", 3) | ge("users_on_bank_account_7d", 3)
        | ge("users_on_phone_72h", 3)
    )

    for wlabel, wdays in WINDOWS_DAYS.items():
        print(f"================  WINDOW = {wlabel} ({wdays}d, day-bucketed)  ================")
        feats = component_features(day_idx, order_rank, user_node, res_nodes, wdays)
        cu, ct = feats["comp_n_users"], feats["comp_n_types"]
        print(f"  max component user-count = {cu.max()} "
              f"(blowup check; rows with >=2 users: {(cu >= 2).sum():,})")

        cu_k, ct_k, c1_k = cu[keep], ct[keep], caught_1hop[keep]
        rules = [
            ("comp_n_users >= 2", cu_k >= 2),
            ("comp_n_users >= 3", cu_k >= 3),
            ("comp_n_users >= 5", cu_k >= 5),
            ("comp_n_users >= 10", cu_k >= 10),
            ("MULTI-HOP: users>=3 & types>=2", (cu_k >= 3) & (ct_k >= 2)),
            ("MULTI-HOP: users>=5 & types>=2", (cu_k >= 5) & (ct_k >= 2)),
            ("MULTI-HOP: users>=3 & types>=3", (cu_k >= 3) & (ct_k >= 3)),
        ]
        rows = [_row(lbl, m, never_k, dpd_k, base_never, base_dpd) for lbl, m in rules]
        res_df = pd.DataFrame(rows)
        with pd.option_context("display.width", 200, "display.max_rows", None):
            print(res_df.to_string(index=False, formatters={
                "never%": "{:.1%}".format, "dpd45%": "{:.1%}".format,
                "never_lift": "{:.1f}x".format, "dpd_lift": "{:.1f}x".format}))

        print("  --- MARGINAL vs 1-hop edges (graph fires AND no 1-hop block edge fires) ---")
        for lbl, m in rules:
            marg = m & ~c1_k
            r = _row(f"[marginal] {lbl}", marg, never_k, dpd_k, base_never, base_dpd)
            print(f"    {r['rule']:<42} n={r['n']:>5}  never={r['never%'] if r['n'] else float('nan'):.1%}"
                  f" ({r['never_lift']:.1f}x)  n_never={r['n_never']}")
        print()


if __name__ == "__main__":
    main()
