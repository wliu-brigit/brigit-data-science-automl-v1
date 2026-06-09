"""Seed-proximity (distance-to-known-fraud) study on the entity graph (v1_76d3ad45).

Rung 2 of the graph ladder. Instead of unsupervised component size, seed the
graph with users we ALREADY know are ring members and ask: does a residual user
who STAYS CLOSE to a known seed go bad? Hypothesis: proximity to a confirmed
ring is sharper (>=90%) than raw component size (~60%), and it surfaces the rest
of a ring the scenario itself missed.

SEEDS (two definitions):
  - scenario  : a prior advance matched by a locked scenario (~residual_mask).
                As-of CLEAN with no lag -- scenario flags are as-of features, so
                a prior flagged advance is known-suspicious at its own time.
  - dpd45     : a prior advance that hit gross-DPD45, activated at its
                expected_dpd45_date (the correct maturity bound -- the outcome is
                only observable then). DPD45, not never-paid: "repaid" is a
                current-snapshot fact, not knowable as-of.

PROXIMITY (two, both as-of / prior-only, on the degree-capped graph):
  - dist1         : the advance shares a resource directly with a prior seed.
  - in_component  : the advance sits in a connected component containing a prior
                    seed user.

Event-driven as-of engine: advances and seed-activations are merged in time
order; an advance READS proximity from strictly-prior seed state, then is added
to the graph. Degree cap drops shared-infra junk nodes (graph_giant_component.py).
Read-only on the pinned snapshot.

    uv run python -m projects.fraud_anomaly_detection.analysis.graph_seed_proximity
"""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

DATASET_ID = "v1_76d3ad45"
WARMUP_DAYS = 30
DEGREE_CAP = 20          # resource nodes shared by > CAP distinct users = junk
_MISSING = {"", "none", "nan", "null", "nat", "0", "0-0", "none-none"}
_D, _B, _P = 1, 2, 4
RESOURCE_COLS = {"device_id": _D, "bank_account_key": _B, "persistent_account_id": _P}


class UnionFind:
    __slots__ = ("parent", "nu", "tm", "seed")

    def __init__(self):
        self.parent, self.nu, self.tm, self.seed = {}, {}, {}, {}

    def add_user(self, x):
        if x not in self.parent:
            self.parent[x] = x; self.nu[x] = 1; self.tm[x] = 0; self.seed[x] = 0

    def add_res(self, x, tb):
        if x not in self.parent:
            self.parent[x] = x; self.nu[x] = 0; self.tm[x] = tb; self.seed[x] = 0

    def find(self, x):
        p = self.parent; root = x
        while p[root] != root:
            root = p[root]
        while p[x] != root:
            p[x], x = root, p[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra
            self.nu[ra] += self.nu[rb]; self.tm[ra] |= self.tm[rb]; self.seed[ra] += self.seed[rb]


def _load_env():
    p = Path(".env")
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _clean(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    s = str(val).strip()
    return None if s.lower() in _MISSING else s


def _stat(label, mask, never, base):
    n = int(mask.sum())
    if n == 0:
        return f"    {label:<40} n=     0"
    nv = never[mask].mean()
    return (f"    {label:<40} n={n:>6}  never={nv:.1%} ({nv/base:.1f}x)  n_never={int(never[mask].sum())}")


def run_seed(order_events, user_node, res_caps, n):
    """Single as-of pass. order_events: list of (time, kind, idx) sorted;
    kind 0 = advance (read then add), 1 = seed activation (mark seed).
    Returns: dist1 (bool), n_seeds_in_comp (int), comp_types (int),
    dist1_persistent (bool, shared a persistent-id node with a seed)."""
    uf = UnionFind()
    seed_touched: dict[str, set] = {}   # resource node -> set of seed user-nodes
    seeded_users: set[str] = set()      # users already counted as seed
    # OTHER-user (true ring) versions exclude the advance's own user as a seed.
    dist1o = np.zeros(n, bool)          # dist1 to ANOTHER bad user
    n_seeds_o = np.zeros(n, np.int32)   # # OTHER bad users in component
    ctypes = np.zeros(n, np.int32)
    self_hist = np.zeros(n, bool)       # own prior bad advance on a shared resource

    def add_advance(i):
        u = user_node[i]
        uf.add_user(u)
        for r, tb in res_caps[i]:
            uf.add_res(r, tb); uf.union(u, r)

    for _, kind, i in order_events:
        if kind == 0:  # advance: READ proximity from strictly-prior seed state
            caps = res_caps[i]
            u = user_node[i]
            dist1o[i] = any((seed_touched.get(r, ()) and (seed_touched[r] - {u}))
                            for r, _ in caps)
            self_hist[i] = any(u in seed_touched.get(r, ()) for r, _ in caps)
            roots = {uf.find(r) for r, _ in caps if r in uf.parent}
            if u in uf.parent:
                roots.add(uf.find(u))
            total = sum(uf.seed[rt] for rt in roots)
            self_in = 1 if (u in seeded_users and u in uf.parent and uf.find(u) in roots) else 0
            n_seeds_o[i] = total - self_in
            tm = 0
            for rt in roots:
                tm |= uf.tm[rt]
            for _, tb in caps:
                tm |= tb
            ctypes[i] = bin(tm).count("1")
            add_advance(i)
        else:          # seed activation: mark this seed's user + resources
            u = user_node[i]
            if u not in uf.parent:
                add_advance(i)  # ensure node exists (matured seed: already there)
            for r, _ in res_caps[i]:
                seed_touched.setdefault(r, set()).add(u)
            if u not in seeded_users:
                seeded_users.add(u)
                uf.seed[uf.find(u)] += 1
    return dist1o, n_seeds_o, ctypes, self_hist


def main():
    _load_env()
    from automl.data.registry import load_dataset_by_id
    from automl.project.session import use_project
    from projects.fraud_anomaly_detection.scenarios import residual_mask

    sess = use_project("fraud_anomaly_detection", dry_run=False)
    df = load_dataset_by_id(DATASET_ID, session=sess).df
    print(f"loaded {len(df):,} rows  (degree cap = {DEGREE_CAP})\n")
    n = len(df)

    ts = pd.to_datetime(df["feature_as_of_ts"])
    ts_int = ts.astype("int64").to_numpy()
    day_idx = (ts.dt.normalize() - ts.dt.normalize().min()).dt.days.to_numpy().astype(int)
    mat_dt = pd.to_datetime(df["expected_dpd45_date"], errors="coerce").astype("int64").to_numpy()

    res_m = residual_mask(df).to_numpy()
    mat = (df["label_mature_d45"].astype(float) == 1).to_numpy()
    dpd = (df["label_gross_dpd45"].astype(float) == 1).to_numpy()
    never = dpd & (df["label_repaid_current_snapshot"].astype(float) == 0).to_numpy()
    keep = res_m & mat & (day_idx >= WARMUP_DAYS)
    base = never[keep].mean()
    nv = never[keep]

    # nodes + lifetime degree (junk filter)
    user_node = [f"U:{u}" for u in df["user_id"].astype(str)]
    raw_nodes = []
    node_users = defaultdict(set)
    present = {c: df[c].to_numpy(object) for c in RESOURCE_COLS}
    for i in range(n):
        rs = []
        for c, tb in RESOURCE_COLS.items():
            cv = _clean(present[c][i])
            if cv is not None:
                node = f"{c[0].upper()}:{cv}"
                rs.append((node, tb)); node_users[node].add(user_node[i])
        raw_nodes.append(rs)
    banned = {k for k, v in node_users.items() if len(v) > DEGREE_CAP}
    res_caps = [[(r, tb) for r, tb in rs if r not in banned] for rs in raw_nodes]
    print(f"banned {len(banned)} junk resource nodes (>{DEGREE_CAP} users)\n")

    # 1-hop caught (for net-new view)
    def num(col):
        return pd.to_numeric(df[col], errors="coerce").to_numpy() if col in df else np.zeros(n)
    caught = np.zeros(n, bool)
    for e in ["users_on_device_id_7d", "users_on_bank_account_7d", "users_on_persistent_account_id_7d",
              "users_on_phone_7d", "users_on_email_7d", "users_on_address_7d",
              "users_on_device_id_72h", "users_on_bank_account_72h",
              "users_on_persistent_account_id_72h", "users_on_phone_72h"]:
        caught |= num(e) >= 2
    c1 = caught[keep]

    scenario_seed = ~res_m                       # matched by a locked scenario
    dpd_seed = dpd & np.isfinite(mat_dt)         # activates at maturity
    print(f"seed advances: scenario={int(scenario_seed.sum()):,} | dpd45(matured)={int(dpd_seed.sum()):,}\n")

    seed_defs = {
        "scenario": [(ts_int[i], 1, i) for i in np.where(scenario_seed)[0]],
        "dpd45": [(mat_dt[i], 1, i) for i in np.where(dpd_seed)[0]],
    }
    advance_events = [(ts_int[i], 0, i) for i in range(n)]

    for name, seed_events in seed_defs.items():
        # merge: at equal time, advances (kind 0) read BEFORE seeds (kind 1) activate -> strict prior
        events = sorted(advance_events + seed_events, key=lambda e: (e[0], e[1]))
        dist1o, n_seeds_o, ctypes, self_hist = run_seed(events, user_node, res_caps, n)
        d1, ns, ct, sh = dist1o[keep], n_seeds_o[keep], ctypes[keep], self_hist[keep]
        print(f"================  SEED = {name}  (OTHER-user / true-ring proximity)  ================")
        print(_stat("dist1: shares resource w/ ANOTHER bad user", d1, nv, base))
        print(_stat("in_comp: >=1 OTHER bad user", ns >= 1, nv, base))
        print(_stat("in_comp: >=2 OTHER bad users", ns >= 2, nv, base))
        print(_stat("in_comp: >=3 OTHER bad users", ns >= 3, nv, base))
        print(_stat("dist1(other) & comp types>=2", d1 & (ct >= 2), nv, base))
        print(_stat("dist1(other) & >=2 other seeds", d1 & (ns >= 2), nv, base))
        print(_stat("[contrast] self_hist: OWN prior bad on resource", sh, nv, base))
        print()


if __name__ == "__main__":
    main()
