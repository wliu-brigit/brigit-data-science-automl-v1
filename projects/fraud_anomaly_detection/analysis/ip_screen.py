"""IP-signal screen on the gated residual (dataset v1_76d3ad45).

wendao asked where IP fits. Two distinct questions:

  (A) IP SHARING — is `ip_address` a scarce-resource edge like device/persistent?
      The base table did NOT materialize a users_on_ip_* as-of count (round-1
      flagged raw IP sharing as ~worthless: carrier NAT and shared household
      IPs). This screens it CHEAPLY and HONESTLY as a CURRENT-STATE count
      (group by ip_address over the whole snapshot, count distinct users) — that
      is LEAKY-OPTIMISTIC (it sees future co-users), so it is a CEILING: if even
      the leaky version is weak, a proper as-of SQL build is not worth it.
  (B) IP-DERIVED signals already present: `signup_ip_matches_latest_ip` (did the
      IP change between signup and advance) and `has_ip_address`.

What this CANNOT do (needs a new enrichment source, see the printed roadmap):
datacenter / hosting / VPN / proxy detection (ASN or IP-intel DB) and IP
geolocation vs stated address/area-code mismatch (GeoLite2-style). Those are the
fraud-shaped IP features; raw sharing is not.

    uv run python -m projects.fraud_anomaly_detection.analysis.ip_screen
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

DATASET_ID = "v1_76d3ad45"


def _load_env() -> None:
    p = Path(".env")
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _stat(mask, never, base):
    n = int(mask.sum())
    if n == 0:
        return n, 0, float("nan"), float("nan")
    nv = never[mask].mean()
    return n, int(never[mask].sum()), nv, (nv / base if base else float("nan"))


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
    never = (dpd & (df["label_repaid_current_snapshot"].astype(float) == 0).to_numpy())
    keep = res & mat
    sub = df[keep].reset_index(drop=True)
    never_k = never[keep]
    base = never_k.mean()
    print(f"residual + mature: {keep.sum():,} rows | base never-paid {base:.3%}")
    print(f"ip_address null: {sub['ip_address'].isna().mean():.1%} | "
          f"signup_ip null: {sub['signup_ip'].isna().mean():.1%} | "
          f"ip_address unique: {sub['ip_address'].nunique()/len(sub):.1%}\n")

    # (A) CURRENT-STATE IP sharing — leaky ceiling.
    for col in ("ip_address", "signup_ip"):
        users_per = sub.groupby(col)["user_id"].transform("nunique")
        print(f"=== {col}: distinct users per value (CURRENT-STATE / leaky ceiling) ===")
        print(f"  {'rule':<22} {'n':>7} {'n_never':>8} {'never%':>8} {'lift':>7}")
        valid = sub[col].notna().to_numpy()
        for t in (2, 3, 5, 10):
            m = valid & (users_per.to_numpy() >= t)
            n, nn, nv, lift = _stat(m, never_k, base)
            print(f"  {col+' users>='+str(t):<22} {n:>7} {nn:>8} {nv:>7.1%} {lift:>6.1f}x")
        print()

    # (B) IP-derived features already in the table.
    print("=== IP-derived features already present ===")
    for col in ("signup_ip_matches_latest_ip", "has_ip_address"):
        if col not in sub.columns:
            continue
        v = pd.to_numeric(sub[col], errors="coerce").to_numpy(float)
        for val in (0, 1):
            n, nn, nv, lift = _stat(v == val, never_k, base)
            print(f"  {col} == {val}: n={n:>7}  never-paid={nv:.2%} ({lift:.1f}x)")
    print()

    print("=== roadmap — fraud-shaped IP signals NOT derivable from the raw IP alone ===")
    print("""  These need an IP-intelligence enrichment (a new pull; Tier-3):
    - datacenter / hosting / VPN / proxy flag  — a real borrower does not advance
      from AWS/DigitalOcean/a VPN exit; needs an ASN/hosting-range or IP-intel DB
      (MaxMind GeoIP2-ISP/ASN, IPinfo, IPQualityScore).
    - geo mismatch — IP-geolocated state/metro vs the KYC address state/zip or the
      phone area code; a tell for account takeover / remote rings. Needs GeoLite2.
    Raw IP SHARING (above) is the weak one (NAT/households); the DERIVED signals
    are the fraud-shaped ones and are the right thing to pull.""")


if __name__ == "__main__":
    main()
