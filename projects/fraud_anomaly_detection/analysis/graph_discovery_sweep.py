"""Consolidated graph-discovery sweep on v1_76d3ad45 -- hunt for HIGH-PRECISION
(>=80%) NET-NEW never-paid pockets via network structure.

Synthesises everything learned across graph_component_screen / graph_edge_study /
graph_giant_component / graph_seed_proximity:
  - advance-co-occurrence edges (user <-> device/bank/persistent), degree-capped
    to drop shared-infra junk nodes;
  - cumulative prior-only graph (max connectivity), strictly as-of;
  - DPD45-matured seeds (activate at expected_dpd45_date), self-excluded so the
    signal is proximity to OTHER known-bad users (ring), not own credit history.

ONE as-of event pass computes per advance:
  cu        component user-count
  ct        # distinct resource types in component
  nb_comp   # OTHER DPD45-matured users in component (ring proximity count)
  nb_d1     # OTHER DPD45-matured users sharing a resource directly (dist-1)
  bad_rate  nb_comp / (cu-1)  -- fraction of the ring that is known-bad

Then a battery of rules (structural / proximity / bad-rate / behavioural
conjunctions), each scored on residual+mature+warmup with: n, never-paid %, lift,
net-new n (no single 1-hop edge >=2), and a binomial p-value vs base. The goal is
a defensible >=80% pocket that is a DISCOVERY (residual = scenario-missed).

    uv run python -m projects.fraud_anomaly_detection.analysis.graph_discovery_sweep
"""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

DATASET_ID = "v1_76d3ad45"
WARMUP_DAYS = 30
DEGREE_CAP = 20
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


def build_graph_features(df, ts_int, mat_dt, dpd, banned_thresh=DEGREE_CAP):
    """Single as-of event pass -> per-advance graph features (see module doc)."""
    n = len(df)
    user_node = [f"U:{u}" for u in df["user_id"].astype(str)]
    raw_nodes, node_users = [], defaultdict(set)
    present = {c: df[c].to_numpy(object) for c in RESOURCE_COLS}
    for i in range(n):
        rs = []
        for c, tb in RESOURCE_COLS.items():
            cv = _clean(present[c][i])
            if cv is not None:
                node = f"{c[0].upper()}:{cv}"; rs.append((node, tb)); node_users[node].add(user_node[i])
        raw_nodes.append(rs)
    banned = {k for k, v in node_users.items() if len(v) > banned_thresh}
    res_caps = [[(r, tb) for r, tb in rs if r not in banned] for rs in raw_nodes]

    # event stream: advances (kind 0, read-then-add) + DPD45 maturity (kind 1)
    events = [(ts_int[i], 0, i) for i in range(n)]
    for i in np.where(dpd & np.isfinite(mat_dt))[0]:
        events.append((mat_dt[i], 1, i))
    events.sort(key=lambda e: (e[0], e[1]))

    uf = UnionFind()
    seed_touched: dict[str, set] = {}
    seeded_users: set[str] = set()
    cu = np.ones(n, np.int32); ct = np.zeros(n, np.int32)
    nb_comp = np.zeros(n, np.int32); nb_d1 = np.zeros(n, np.int32)

    def add_advance(i):
        u = user_node[i]; uf.add_user(u)
        for r, tb in res_caps[i]:
            uf.add_res(r, tb); uf.union(u, r)

    for _, kind, i in events:
        u = user_node[i]; caps = res_caps[i]
        if kind == 0:
            roots = {uf.find(r) for r, _ in caps if r in uf.parent}
            if u in uf.parent:
                roots.add(uf.find(u))
            pu = sum(uf.nu[rt] for rt in roots)
            tot_seed = sum(uf.seed[rt] for rt in roots)
            self_in = 1 if (u in seeded_users and u in uf.parent and uf.find(u) in roots) else 0
            tm = 0
            for rt in roots:
                tm |= uf.tm[rt]
            own_tm = 0
            for _, tb in caps:
                own_tm |= tb
            cu[i] = pu + (0 if (u in uf.parent and uf.find(u) in roots) else 1)
            ct[i] = bin(tm | own_tm).count("1")
            nb_comp[i] = tot_seed - self_in
            d1set = set()
            for r, _ in caps:
                d1set |= seed_touched.get(r, set())
            d1set.discard(u)
            nb_d1[i] = len(d1set)
            add_advance(i)
        else:
            if u not in uf.parent:
                add_advance(i)
            for r, _ in caps:
                seed_touched.setdefault(r, set()).add(u)
            if u not in seeded_users:
                seeded_users.add(u); uf.seed[uf.find(u)] += 1

    other = np.maximum(cu - 1, 1)
    bad_rate = nb_comp / other
    return {"cu": cu, "ct": ct, "nb_comp": nb_comp, "nb_d1": nb_d1, "bad_rate": bad_rate}


def main():
    _load_env()
    from automl.data.registry import load_dataset_by_id
    from automl.project.session import use_project
    from projects.fraud_anomaly_detection.scenarios import residual_mask

    sess = use_project("fraud_anomaly_detection", dry_run=False)
    df = load_dataset_by_id(DATASET_ID, session=sess).df
    print(f"loaded {len(df):,} rows (cap={DEGREE_CAP})\n")
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

    def num(col):
        return pd.to_numeric(df[col], errors="coerce").to_numpy() if col in df else np.zeros(n)
    caught = np.zeros(n, bool)
    for e in ["users_on_device_id_7d", "users_on_bank_account_7d", "users_on_persistent_account_id_7d",
              "users_on_phone_7d", "users_on_email_7d", "users_on_address_7d", "users_on_device_id_72h",
              "users_on_bank_account_72h", "users_on_persistent_account_id_72h", "users_on_phone_72h"]:
        caught |= num(e) >= 2
    c1 = caught[keep]

    f = build_graph_features(df, ts_int, mat_dt, dpd)
    cu, ct, nbc, nbd, br = (f["cu"][keep], f["ct"][keep], f["nb_comp"][keep],
                            f["nb_d1"][keep], f["bad_rate"][keep])

    # behavioural features for conjunctions
    neob = num("is_neobank_high_risk_institution")[keep] == 1
    fresh = num("days_since_plaid_account_created")[keep] <= 37.5
    small = num("total_disbursed")[keep] <= 50

    rules = {
        # structural
        "comp>=5 & types>=2": (cu >= 5) & (ct >= 2),
        "comp>=3 & types>=3": (cu >= 3) & (ct >= 3),
        # ring proximity to OTHER known-bad
        "nb_comp>=1 (>=1 other bad in ring)": nbc >= 1,
        "nb_comp>=2": nbc >= 2,
        "nb_comp>=3": nbc >= 3,
        "nb_d1>=1 (dist-1 to other bad)": nbd >= 1,
        "nb_d1>=2": nbd >= 2,
        # bad-rate (fraction of ring known-bad)
        "bad_rate>=0.5 & cu>=3": (br >= 0.5) & (cu >= 3),
        "bad_rate>=0.5 & cu>=4": (br >= 0.5) & (cu >= 4),
        "bad_rate==1.0 & cu>=3 (all-other-bad)": (br >= 0.999) & (cu >= 3),
        # combine proximity with structure
        "nb_comp>=1 & types>=2": (nbc >= 1) & (ct >= 2),
        "nb_comp>=1 & types>=3": (nbc >= 1) & (ct >= 3),
        # SHARPEN toward >=80%: intersect the working levers
        "nb_comp>=2 & types>=3": (nbc >= 2) & (ct >= 3),
        "nb_comp>=2 & types>=2": (nbc >= 2) & (ct >= 2),
        "nb_d1>=2 & types>=2": (nbd >= 2) & (ct >= 2),
        "nb_d1>=1 & types>=3": (nbd >= 1) & (ct >= 3),
        "bad_rate>=0.5 & cu>=4 & types>=2": (br >= 0.5) & (cu >= 4) & (ct >= 2),
        "bad_rate>=0.66 & cu>=4": (br >= 0.66) & (cu >= 4),
        "nb_comp>=2 & bad_rate>=0.5": (nbc >= 2) & (br >= 0.5),
        # combine proximity with behaviour (lift the review cohort?)
        "nb_comp>=1 & neobank": (nbc >= 1) & neob,
        "nb_comp>=1 & fresh-acct": (nbc >= 1) & fresh,
        "nb_comp>=1 & fresh & small": (nbc >= 1) & fresh & small,
        # volume unions
        "comp>=5&types>=2 | nb_comp>=1 (union)": ((cu >= 5) & (ct >= 2)) | (nbc >= 1),
        "nb_comp>=1 | (comp>=3 & types>=3)": (nbc >= 1) | ((cu >= 3) & (ct >= 3)),
    }

    rows = []
    for lbl, m in rules.items():
        for tag, mm in [("all", m), ("net-new", m & ~c1)]:
            k = int(mm.sum())
            if k == 0:
                rows.append((lbl, tag, 0, np.nan, np.nan, 0, np.nan)); continue
            p = never[keep][mm].mean()
            kk = int(nv[mm].sum())
            pv = binomtest(kk, k, base, alternative="greater").pvalue
            rows.append((lbl, tag, k, p, p / base, kk, pv))

    print(f"residual+mature+warmup {keep.sum():,} | base never-paid {base:.3%}\n")
    print(f"{'rule':<42}{'set':<9}{'n':>6}{'never%':>9}{'lift':>7}{'n_np':>6}{'binom_p':>11}")
    print("-" * 90)
    for lbl, tag, k, p, lift, kk, pv in rows:
        if k == 0:
            print(f"{lbl:<42}{tag:<9}{k:>6}"); continue
        star = " *" if (p >= 0.80 and pv < 1e-3) else ""
        print(f"{lbl:<42}{tag:<9}{k:>6}{p:>8.1%}{lift:>6.1f}x{kk:>6}{pv:>11.1e}{star}")

    # ── cap sensitivity: does relaxing/tightening the junk cap grow volume? ──
    print("\n\n=== degree-cap sensitivity (key proximity rules) ===")
    print(f"{'cap':<6}{'rule':<26}{'n':>6}{'never%':>9}{'lift':>7}{'n_np':>6}")
    for cap in (10, 20, 50, 200):
        fc = build_graph_features(df, ts_int, mat_dt, dpd, banned_thresh=cap)
        cuk, ctk, nbck = fc["cu"][keep], fc["ct"][keep], fc["nb_comp"][keep]
        for lbl, m in [("nb_comp>=1", nbck >= 1), ("nb_comp>=1 & types>=3", (nbck >= 1) & (ctk >= 3)),
                       ("nb_comp>=2", nbck >= 2), ("nb_comp>=3", nbck >= 3)]:
            k = int(m.sum())
            if k:
                p = nv[m].mean()
                print(f"{cap:<6}{lbl:<26}{k:>6}{p:>8.1%}{p/base:>6.1f}x{int(nv[m].sum()):>6}")
            else:
                print(f"{cap:<6}{lbl:<26}{k:>6}")


if __name__ == "__main__":
    main()
