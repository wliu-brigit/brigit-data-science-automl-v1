"""Per-edge precision screen on the gated residual (dataset v1_76d3ad45).

The unsupervised lens (unsupervised_lens.py) showed the new scarce-resource
sharing edges are enriched 100-200x at the top of the anomaly cohort, but the
anomaly SCORE is the wrong deployment vehicle (aggregate AP stays ~1.3x base —
the edges are too rare to move it, and the cohort bulk is the same fast-cycling
pattern). Per the project stance, the deployment vehicle is a precise rule, so
this screen reports each candidate edge's OWN shape-stat — exactly what a
register entry needs: volume, never-paid / DPD45 precision, and lift vs base.

Computed on the full residual + mature population (rules are validated on the
whole pinned snapshot, not a held-out split — there is no model to overfit).

    uv run python -m projects.fraud_anomaly_detection.analysis.edge_precision_screen
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

DATASET_ID = "v1_76d3ad45"

EDGES = [
    ("users_on_device_id_72h", "device 72h"),
    ("users_on_device_id_7d", "device 7d"),
    ("users_on_device_id_30d", "device 30d"),
    ("users_on_persistent_account_id_72h", "persistent 72h"),
    ("users_on_persistent_account_id_7d", "persistent 7d"),
    ("users_on_persistent_account_id_30d", "persistent 30d"),
    ("users_on_address_72h", "address 72h"),
    ("users_on_address_7d", "address 7d"),
    ("users_on_address_30d", "address 30d"),
    ("users_on_phone_72h", "phone 72h"),
    ("users_on_phone_7d", "phone 7d"),
    ("users_on_phone_30d", "phone 30d"),
    ("users_on_email_72h", "email 72h"),
    ("users_on_email_7d", "email 7d"),
    ("users_on_email_30d", "email 30d"),
    ("users_on_bank_account_7d", "bank_account 7d (sub-ring)"),
]
THRESHOLDS = (2, 3)


def _load_env() -> None:
    p = Path(".env")
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _row(label: str, mask: np.ndarray, never: np.ndarray, dpd: np.ndarray,
         base_never: float, base_dpd: float) -> dict:
    n = int(mask.sum())
    if n == 0:
        return {"rule": label, "n": 0, "never%": float("nan"), "never_lift": float("nan"),
                "dpd45%": float("nan"), "dpd_lift": float("nan"), "n_never": 0}
    nv = never[mask].mean()
    dp = dpd[mask].mean()
    return {
        "rule": label, "n": n, "n_never": int(never[mask].sum()),
        "never%": nv, "never_lift": nv / base_never if base_never else float("nan"),
        "dpd45%": dp, "dpd_lift": dp / base_dpd if base_dpd else float("nan"),
    }


def main() -> None:
    _load_env()
    from automl.data.registry import load_dataset_by_id
    from automl.project.session import use_project
    from projects.fraud_anomaly_detection.scenarios import residual_mask

    sess = use_project("fraud_anomaly_detection", dry_run=False)
    df = load_dataset_by_id(DATASET_ID, session=sess).df

    res = residual_mask(df).to_numpy()
    mat = (df["label_mature_d45"].astype(float) == 1).to_numpy()
    dpd = (df["label_gross_dpd45"].astype(float) == 1).to_numpy()
    never = dpd & (df["label_repaid_current_snapshot"].astype(float) == 0).to_numpy()
    keep = res & mat

    sub = df[keep].reset_index(drop=True)
    never_k, dpd_k = never[keep], dpd[keep]
    base_never, base_dpd = never_k.mean(), dpd_k.mean()
    print(f"residual + mature population: {keep.sum():,} rows")
    print(f"base never-paid {base_never:.3%} | base DPD45 {base_dpd:.3%}\n")

    def col(c):
        return pd.to_numeric(sub[c], errors="coerce").to_numpy(float)

    rows = []
    for c, label in EDGES:
        if c not in sub.columns:
            continue
        v = col(c)
        for t in THRESHOLDS:
            rows.append(_row(f"{label} >= {t}", v >= t, never_k, dpd_k, base_never, base_dpd))

    # name-match (low Jaro-Winkler last name = synthetic/stolen tell)
    if "name_match_last" in sub.columns:
        nm = col("name_match_last")
        for thr in (70, 80, 90):
            rows.append(_row(f"name_match_last < {thr}", nm < thr, never_k, dpd_k, base_never, base_dpd))

    # neobank flag alone
    if "is_neobank_high_risk_institution" in sub.columns:
        rows.append(_row("is_neobank_high_risk == 1",
                         col("is_neobank_high_risk_institution") == 1, never_k, dpd_k, base_never, base_dpd))

    res_df = pd.DataFrame(rows)
    with pd.option_context("display.width", 200, "display.max_rows", None):
        print("=== per-edge precision (residual + mature) ===")
        print(res_df.to_string(index=False, formatters={
            "never%": "{:.1%}".format, "dpd45%": "{:.1%}".format,
            "never_lift": "{:.1f}x".format, "dpd_lift": "{:.1f}x".format,
        }))

    # --- union of the new identity-coherence edges (device/persistent/phone/email),
    # 7d >=2, plus address with a joint-account disqualifier ---
    def ge(c, t):
        return col(c) >= t if c in sub.columns else np.zeros(len(sub), bool)

    is_joint = (col("is_joint") == 1) if "is_joint" in sub.columns else np.zeros(len(sub), bool)
    union_core = ge("users_on_device_id_7d", 2) | ge("users_on_persistent_account_id_7d", 2) \
        | ge("users_on_phone_7d", 2) | ge("users_on_email_7d", 2)
    union_addr = ge("users_on_address_7d", 2) & ~is_joint
    union_all = union_core | union_addr

    print("\n=== unions (7d, >=2; address excludes joint accounts) ===")
    for label, m in [
        ("device|persistent|phone|email (core)", union_core),
        ("core | address(non-joint)", union_all),
        ("device|persistent only", ge("users_on_device_id_7d", 2) | ge("users_on_persistent_account_id_7d", 2)),
    ]:
        r = _row(label, m, never_k, dpd_k, base_never, base_dpd)
        print(f"  {r['rule']:<40} n={r['n']:>5}  never={r['never%']:.1%} ({r['never_lift']:.1f}x)  "
              f"dpd45={r['dpd45%']:.1%} ({r['dpd_lift']:.1f}x)  n_never={r['n_never']}")


if __name__ == "__main__":
    main()
