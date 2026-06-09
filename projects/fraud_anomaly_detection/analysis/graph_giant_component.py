"""Giant-component / junk-node study for the entity graph (v1_76d3ad45).

graph_edge_study.py found the cumulative advance-co-occurrence graph is richly
connected (NOT sparse), but polluted by giant components (max 3,789 users) --
shared-infrastructure resources (default device_id, kiosks) that merge thousands
of unrelated users. The genuine net-new signal was SMALL, DENSE, MULTI-TYPE
components (comp>=5 & types>=2 -> 60.7% never-paid). This script:

1. Identifies the highest-degree resource nodes (the junk connectors) and their
   type -- confirms whether a few promiscuous values cause the mega-merge.
2. Re-builds the cumulative prior-only graph with a per-resource DEGREE CAP:
   a resource node shared by more than CAP distinct users is treated as a
   non-edge (too promiscuous to be a ring -- shared infra, not a cabal).
3. Re-measures net-new never-paid at each cap to find the edge definition the
   data supports.

Caveat: the cap uses each node's LIFETIME distinct-user degree as a static junk
filter (a device being shared infra is ~time-invariant). For a deployable
feature the cap must be applied as-of; here it is a discovery screen.

    uv run python -m projects.fraud_anomaly_detection.analysis.graph_giant_component
"""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

DATASET_ID = "v1_76d3ad45"
WARMUP_DAYS = 30
_MISSING = {"", "none", "nan", "null", "nat", "0", "0-0", "none-none"}
_D, _B, _P = 1, 2, 4
RESOURCE_COLS = {"device_id": _D, "bank_account_key": _B, "persistent_account_id": _P}
CAPS = [None, 50, 20, 10, 5]


class UnionFind:
    __slots__ = ("parent", "nu", "tm")

    def __init__(self) -> None:
        self.parent, self.nu, self.tm = {}, {}, {}

    def add_user(self, x):
        if x not in self.parent:
            self.parent[x] = x; self.nu[x] = 1; self.tm[x] = 0

    def add_res(self, x, tb):
        if x not in self.parent:
            self.parent[x] = x; self.nu[x] = 0; self.tm[x] = tb

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
            self.parent[rb] = ra; self.nu[ra] += self.nu[rb]; self.tm[ra] |= self.tm[rb]


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


def _cumulative(order, user_node, res_nodes, banned: set):
    """Cumulative prior-only component features, skipping banned resource nodes."""
    n = len(user_node)
    cu = np.ones(n, dtype=np.int32)
    ct = np.zeros(n, dtype=np.int32)
    uf = UnionFind()
    for idx in order:
        u = user_node[idx]
        rs = [(r, tb) for r, tb in res_nodes[idx] if r not in banned]
        roots = {uf.find(r) for r, _ in rs if r in uf.parent}
        pu = sum(uf.nu[rt] for rt in roots)
        tm = 0
        for rt in roots:
            tm |= uf.tm[rt]
        u_counted = u in uf.parent and uf.find(u) in roots
        own_tm = 0
        for _, tb in rs:
            own_tm |= tb
        cu[idx] = pu + (0 if u_counted else 1)
        ct[idx] = bin(tm | own_tm).count("1")
        uf.add_user(u)
        for r, tb in rs:
            uf.add_res(r, tb); uf.union(u, r)
    return cu, ct


def main():
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
    aid = df["advance_id"].astype(str).to_numpy()
    order = np.lexsort((aid, ts_int))

    res_m = residual_mask(df).to_numpy()
    mat = (df["label_mature_d45"].astype(float) == 1).to_numpy()
    dpd = (df["label_gross_dpd45"].astype(float) == 1).to_numpy()
    never = dpd & (df["label_repaid_current_snapshot"].astype(float) == 0).to_numpy()
    keep = res_m & mat & (day_idx >= WARMUP_DAYS)
    base = never[keep].mean()

    # nodes
    user_node = [f"U:{u}" for u in df["user_id"].astype(str)]
    res_nodes = []
    present = {c: df[c].to_numpy(object) for c in RESOURCE_COLS}
    node_users = defaultdict(set)   # resource node -> set of distinct users (lifetime)
    for i in range(len(df)):
        rs = []
        for c, tb in RESOURCE_COLS.items():
            cv = _clean(present[c][i])
            if cv is not None:
                node = f"{c[0].upper()}:{cv}"
                rs.append((node, tb))
                node_users[node].add(user_node[i])
        res_nodes.append(rs)
    node_deg = {k: len(v) for k, v in node_users.items()}

    # 1. top junk connectors
    print("==================  highest-degree resource nodes (junk connectors)  ==================")
    top = sorted(node_deg.items(), key=lambda kv: -kv[1])[:15]
    for node, deg in top:
        print(f"  {node[:60]:<62} distinct users (lifetime) = {deg:,}")
    typ = defaultdict(list)
    for node, deg in node_deg.items():
        typ[node[0]].append(deg)
    print("\n  degree distribution by resource type (D=device B=bank P=persistent):")
    for t, degs in typ.items():
        a = np.array(degs)
        print(f"    {t}: nodes={len(a):,}  shared(>=2 users)={int((a>=2).sum()):,}  "
              f"max={a.max()}  p99={np.percentile(a,99):.0f}  >50 users={int((a>50).sum())}")

    # caught-by-1hop for net-new
    def num(col):
        return pd.to_numeric(df[col], errors="coerce").to_numpy() if col in df else np.zeros(len(df))
    edges = ["users_on_device_id_7d", "users_on_bank_account_7d",
             "users_on_persistent_account_id_7d", "users_on_phone_7d",
             "users_on_email_7d", "users_on_address_7d", "users_on_device_id_72h",
             "users_on_bank_account_72h", "users_on_persistent_account_id_72h", "users_on_phone_72h"]
    caught = np.zeros(len(df), bool)
    for e in edges:
        caught |= num(e) >= 2
    c1 = caught[keep]
    nv = never[keep]

    # 2-3. cap sweep
    print("\n\n==================  net-new never-paid by degree cap  ==================")
    print(f"  residual+mature+warmup population {keep.sum():,} | base never-paid {base:.3%}")
    print("  (net-new = graph rule fires AND no single 1-hop edge >=2)\n")
    for cap in CAPS:
        banned = set() if cap is None else {k for k, d in node_deg.items() if d > cap}
        cu, ct = _cumulative(order, user_node, res_nodes, banned)
        cuk, ctk = cu[keep], ct[keep]
        caplbl = "none" if cap is None else f"<= {cap}"
        print(f"  --- CAP {caplbl}  (banned junk nodes: {len(banned):,}; max comp now {cu.max()}) ---")
        for lbl, m in [("comp>=3", cuk >= 3), ("comp>=5", cuk >= 5),
                       ("comp>=3 & types>=2", (cuk >= 3) & (ctk >= 2)),
                       ("comp>=5 & types>=2", (cuk >= 5) & (ctk >= 2)),
                       ("comp>=3 & types>=3", (cuk >= 3) & (ctk >= 3))]:
            mm = m & ~c1
            n = int(mm.sum())
            if n:
                p = nv[mm].mean()
                print(f"      [net-new] {lbl:<24} n={n:>5}  never={p:.1%} ({p/base:.1f}x)  n_never={int(nv[mm].sum())}")
            else:
                print(f"      [net-new] {lbl:<24} n=    0")
        print()


if __name__ == "__main__":
    main()
