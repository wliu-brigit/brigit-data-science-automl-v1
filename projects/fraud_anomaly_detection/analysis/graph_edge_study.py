"""Edge-definition study for the entity graph (dataset v1_76d3ad45).

graph_component_screen.py found that an advance-co-occurrence graph (link a user
to the resources it ADVANCED on) is far sparser than the 1-hop edges: 7d
comp>=3 was 17 rows vs 376 for users_on_device_id_7d>=3. This script asks the
data WHY, and whether ANY graph construction on the existing dataset finds
never-paid the existing edges/scenarios miss. Two questions:

Q1. EDGE EVENT.  The 1-hop `users_on_*` columns count IDENTITIES attached to a
    resource (anchored on identity_created_time, from the full identity table --
    including identities that never took an advance). Our graph can only link
    users that ACTUALLY ADVANCED. Measure the gap: for the same resource+window,
    advance-co-occurrence degree vs the SQL identity-attach count. If the advance
    graph sees a small fraction, the rich graph needs the upstream (user,
    resource, attach_time) edge list emitted -> a rebuild, not the current data.

Q2. NET-NEW DISCOVERY.  Build the MOST connected honest graph -- cumulative,
    prior-only (every advance strictly before t_A, no window) -- and ask: among
    residual + mature rows (already not caught by the locked scenarios), how much
    never-paid does multi-hop component membership catch that NO single 1-hop
    edge catches? That marginal, multi-hop, never-paid count is the only number
    that justifies the graph effort. If it is tiny, the direction is exhausted
    on this data and the lever is the richer edge list (rebuild) or new features.

All strictly as-of / prior-only. Read-only on the pinned snapshot.

    uv run python -m projects.fraud_anomaly_detection.analysis.graph_edge_study
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

DATASET_ID = "v1_76d3ad45"
WARMUP_DAYS = 30
SEVEN_D_NS = 7 * 86_400 * 1_000_000_000
_MISSING = {"", "none", "nan", "null", "nat", "0", "0-0", "none-none"}
_D, _B, _P = 1, 2, 4
RESOURCE_COLS = {"device_id": _D, "bank_account_key": _B, "persistent_account_id": _P}


class UnionFind:
    __slots__ = ("parent", "nu", "tm")

    def __init__(self) -> None:
        self.parent, self.nu, self.tm = {}, {}, {}

    def add_user(self, x: str) -> None:
        if x not in self.parent:
            self.parent[x] = x; self.nu[x] = 1; self.tm[x] = 0

    def add_res(self, x: str, tb: int) -> None:
        if x not in self.parent:
            self.parent[x] = x; self.nu[x] = 0; self.tm[x] = tb

    def find(self, x: str) -> str:
        p = self.parent; root = x
        while p[root] != root:
            root = p[root]
        while p[x] != root:
            p[x], x = root, p[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra; self.nu[ra] += self.nu[rb]; self.tm[ra] |= self.tm[rb]


def _load_env() -> None:
    p = Path(".env")
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _clean(val) -> str | None:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    s = str(val).strip()
    return None if s.lower() in _MISSING else s


def _cln_str(label, mask, never, base) -> str:
    n = int(mask.sum())
    if n == 0:
        return f"  {label:<48} n=    0"
    nv = never[mask].mean()
    return (f"  {label:<48} n={n:>6}  never={nv:.1%} ({nv / base:.1f}x)  "
            f"n_never={int(never[mask].sum())}")


# ── Q1: advance-co-occurrence degree vs the SQL identity-attach count ─────────
def advance_degree_7d(df: pd.DataFrame, col: str, ts_int: np.ndarray) -> np.ndarray:
    """Per advance: # DISTINCT OTHER users that ADVANCED on this row's `col`
    value within the prior 7d (strictly before t_A). Comparable to
    users_on_<col>_7d, but counting advancers instead of attached identities."""
    n = len(df)
    deg = np.zeros(n, dtype=np.int32)
    vals = np.array([_clean(v) for v in df[col].to_numpy(object)], dtype=object)
    valid = np.array([v is not None for v in vals])
    users = df["user_id"].astype(str).to_numpy()
    idx = np.where(valid)[0]
    # sort the valid rows by (resource value, ts)
    order = sorted(idx, key=lambda i: (vals[i], ts_int[i], i))
    from collections import deque
    cur_val = None
    win: deque = deque()          # (ts, user) within 7d, prior
    counts: dict[str, int] = {}   # user -> occurrences in window
    for i in order:
        v = vals[i]
        if v != cur_val:
            cur_val, win, counts = v, deque(), {}
        t = ts_int[i]
        while win and win[0][0] < t - SEVEN_D_NS:
            ot, ou = win.popleft()
            counts[ou] -= 1
            if counts[ou] == 0:
                del counts[ou]
        # distinct OTHER users in the prior window (exclude this row's own user)
        u = users[i]
        deg[i] = len(counts) - (1 if u in counts else 0)
        win.append((t, u)); counts[u] = counts.get(u, 0) + 1
    return deg


def main() -> None:
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

    res_m = residual_mask(df).to_numpy()
    mat = (df["label_mature_d45"].astype(float) == 1).to_numpy()
    dpd = (df["label_gross_dpd45"].astype(float) == 1).to_numpy()
    never = dpd & (df["label_repaid_current_snapshot"].astype(float) == 0).to_numpy()
    keep = res_m & mat & (day_idx >= WARMUP_DAYS)
    base = never[keep].mean()
    print(f"residual+mature+warmup population: {keep.sum():,} | base never-paid {base:.3%}\n")

    def num(col):
        return pd.to_numeric(df[col], errors="coerce").to_numpy() if col in df else np.zeros(len(df))

    # ===== Q1: edge-event gap (device is the largest ring; check all three) =====
    print("==================  Q1  advance-co-occurrence degree vs SQL identity-attach count  ==================")
    for col, sqlcol in [("device_id", "users_on_device_id_7d"),
                        ("bank_account_key", "users_on_bank_account_7d"),
                        ("persistent_account_id", "users_on_persistent_account_id_7d")]:
        if sqlcol not in df:
            continue
        adv = advance_degree_7d(df, col, ts_int)
        sqlc = num(sqlcol)
        # SQL count is "# identities incl. self"; subtract 1 to compare distinct-OTHERS
        sql_others = np.clip(sqlc - 1, 0, None)
        ring = sql_others >= 2          # SQL says >=3 identities on the resource (7d)
        seen = adv[ring]
        print(f"\n  {col}:  rows where SQL says >=3 identities/7d (others>=2): {ring.sum():,}")
        if ring.sum():
            print(f"    of those, advance-co-occurrence sees others: "
                  f"median={np.median(seen):.0f}  mean={seen.mean():.2f}  "
                  f">=2 advancers={int((seen >= 2).sum()):,} ({(seen >= 2).mean():.1%})  "
                  f"0 advancers={int((seen == 0).sum()):,} ({(seen == 0).mean():.1%})")

    # ===== Q2: cumulative prior-only component (max connectivity) =====
    print("\n\n==================  Q2  cumulative prior-only graph (max connectivity)  ==================")
    typebit = RESOURCE_COLS
    user_node = [f"U:{u}" for u in df["user_id"].astype(str)]
    res_nodes: list[list[tuple[str, int]]] = []
    present = {c: df[c].to_numpy(object) for c in RESOURCE_COLS}
    for i in range(len(df)):
        rs = []
        for c, tb in typebit.items():
            cv = _clean(present[c][i])
            if cv is not None:
                rs.append((f"{c[0].upper()}:{cv}", tb))
        res_nodes.append(rs)

    aid = df["advance_id"].astype(str).to_numpy()
    order = np.lexsort((aid, ts_int))   # strict global time order
    n = len(df)
    comp_n_users = np.ones(n, dtype=np.int32)
    comp_n_types = np.zeros(n, dtype=np.int32)
    uf = UnionFind()
    for idx in order:
        u = user_node[idx]; rs = res_nodes[idx]
        roots = {uf.find(r) for r, _ in rs if r in uf.parent}
        pu = sum(uf.nu[rt] for rt in roots)
        tm = 0
        for rt in roots:
            tm |= uf.tm[rt]
        u_counted = u in uf.parent and uf.find(u) in roots
        own_tm = 0
        for _, tb in rs:
            own_tm |= tb
        comp_n_users[idx] = pu + (0 if u_counted else 1)
        comp_n_types[idx] = bin(tm | own_tm).count("1")
        uf.add_user(u)
        for r, tb in rs:
            uf.add_res(r, tb); uf.union(u, r)

    print(f"  max cumulative component user-count = {comp_n_users.max()} "
          f"(rows with comp>=2: {(comp_n_users >= 2).sum():,}, comp>=3: {(comp_n_users >= 3).sum():,})")

    cu, ct = comp_n_users[keep], comp_n_types[keep]
    nv = never[keep]
    print("\n  --- gross precision on residual+mature (cumulative component) ---")
    for lbl, m in [("comp>=2", cu >= 2), ("comp>=3", cu >= 3), ("comp>=5", cu >= 5),
                   ("comp>=10", cu >= 10), ("comp>=3 & types>=2", (cu >= 3) & (ct >= 2)),
                   ("comp>=3 & types>=3", (cu >= 3) & (ct >= 3))]:
        print(_cln_str(lbl, m, nv, base))

    # net-new: residual+mature rows where NO single 1-hop edge fires (>=2 on anything)
    edges = ["users_on_device_id_7d", "users_on_bank_account_7d",
             "users_on_persistent_account_id_7d", "users_on_phone_7d",
             "users_on_email_7d", "users_on_address_7d",
             "users_on_device_id_72h", "users_on_bank_account_72h",
             "users_on_persistent_account_id_72h", "users_on_phone_72h"]
    caught_1hop = np.zeros(len(df), bool)
    for e in edges:
        caught_1hop |= num(e) >= 2
    c1 = caught_1hop[keep]
    print("\n  --- NET-NEW: multi-hop component fires AND no single 1-hop edge >=2 (the discovery payoff) ---")
    for lbl, m in [("comp>=2", cu >= 2), ("comp>=3", cu >= 3),
                   ("comp>=3 & types>=2", (cu >= 3) & (ct >= 2)),
                   ("comp>=5 & types>=2", (cu >= 5) & (ct >= 2))]:
        print(_cln_str(f"[net-new] {lbl}", m & ~c1, nv, base))
    print(f"\n  (residual+mature never-paid NOT caught by any 1-hop edge>=2: "
          f"{int((nv & ~c1).sum())} of {int(nv.sum())} total residual never-paid)")


if __name__ == "__main__":
    main()
