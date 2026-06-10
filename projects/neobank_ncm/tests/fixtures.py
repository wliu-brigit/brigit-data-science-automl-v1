"""Synthetic base-table fixture for offline (no-warehouse) end-to-end runs.

Generates a frame with the same shape the materialized snapshot has: known
and unknown rows across the train/test/oot windows, NULL targets +
synthetic scores on unknowns, the spine metadata columns, a representative
subset of the locked feature set (including renamed derived features,
plaid-pattern and netflow columns), and one excluded column to prove
exclude_cols works. Signal is planted so AUC is comfortably above chance.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

BANKS = ["CHIME", "VARO", "CURRENT", "DAVE", "GO2BANK", "SOFI", "RAREBANK"]
BANK_RISK = {
    "CHIME": 0.0, "VARO": 0.2, "CURRENT": 0.3, "DAVE": -0.1,
    "GO2BANK": 0.1, "SOFI": -0.3, "RAREBANK": 0.4,
}
PAY_FREQS = ["BIWEEKLY", "WEEKLY", "MONTHLY", "SEMI_MONTHLY", "IRREGULAR"]

# representative subset of the locked feature list (snapshot column names)
NUMERIC_FEATURES = [
    "balancesd", "dailyincomemean", "maxnegativebalpast30days", "balancemean",
    "inflowsum14d", "outflowsum14d", "balancemeanafterpayday0",
    "balancemeanafterpayday1", "highestpaydepositmean", "daystopayday",
    "odandnsffeesdaily", "dailyincomeregularmean", "recurrentcount",
    "individualcreditamountmean", "individualdebitamountsd",
    "highestpaydepositvoladj", "negbalancerate", "noactivityrate",
    "dayswithbrigit", "creditorsummarycreditthirtydayamount",
    "davesummarycreditninetydayamount", "earninsummarycreditninetydayamount",
    "othercompetitorsummarycreditninetydayamount",
    "plaidfeaturessummary_incomewages_lookbackwindow14d_inflow_sum",
    "plaidfeaturessummary_loanpaymentspersonalloanpayment_lookbackwindow7d_outflow_sum",
    "plaidfeaturessummary_transferoutsavings_lookbackwindow30d_outflow_count",
    "total_incount_14", "total_outcount_30",
]


def make_synthetic_base_table(
    n_known_train: int = 1200,
    n_unknown_train: int = 900,
    n_known_test: int = 400,
    n_known_oot: int = 400,
    n_unknown_oot: int = 200,
    seed: int = 7,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    blocks = [
        ("train_window", True, n_known_train),
        ("test_window", True, n_known_test),
        ("train_window", False, n_unknown_train),
        ("oot", True, n_known_oot),
        ("oot", False, n_unknown_oot),
    ]
    frames = []
    offset = 0
    for window, is_known, n in blocks:
        frames.append(_block(rng, window=window, is_known=is_known, n=n, offset=offset))
        offset += n
    df = pd.concat(frames, ignore_index=True)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    # the legacy server-side downsample rank, exactly as base_table.sql
    # materializes it: rank within the unknown-train group by a deterministic
    # hash of entity_id, NULL everywhere else
    unk_train = ~df["is_known"] & (df["split"] == "train")
    df["unknown_train_hash_rank"] = np.nan
    df.loc[unk_train, "unknown_train_hash_rank"] = _hash_rank(df.loc[unk_train, "entity_id"])
    return df


def write_fixture_csv(path: Path, **kwargs) -> Path:
    df = make_synthetic_base_table(**kwargs)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def _block(rng, *, window: str, is_known: bool, n: int, offset: int) -> pd.DataFrame:
    if window == "train_window":
        days = rng.integers(0, 304, n)  # Jan 1 – Oct 31 2025
        start = np.datetime64("2025-01-01")
    elif window == "test_window":
        days = rng.integers(0, 61, n)  # Nov 1 – Dec 31 2025
        start = np.datetime64("2025-11-01")
    else:  # oot
        days = rng.integers(0, 59, n)  # Jan 1 – Feb 28 2026
        start = np.datetime64("2026-01-01")
    origination = start + days.astype("timedelta64[D]")

    df = pd.DataFrame(
        {
            "entity_id": [f"e{offset + i:07d}" for i in range(n)],
            "user_id": [f"u{offset + i:07d}" for i in range(n)],
            "sa_id": [f"sa{offset + i:07d}" for i in range(n)],
            "split": "oot" if window == "oot" else "train",
            "is_known": is_known,
            "origination_date": pd.to_datetime(origination),
            "original_due_date": pd.to_datetime(origination) + pd.Timedelta(days=14),
            "amount": rng.uniform(50, 250, n).round(2),
            "valid_from": pd.NaT if is_known else pd.to_datetime(origination),
            "valid_to": pd.NaT,
            "bankinstitution": rng.choice(BANKS, n, p=[0.45, 0.18, 0.12, 0.1, 0.08, 0.05, 0.02]),
            "highestpayfrequency": rng.choice(PAY_FREQS + [None], n),
        }
    )

    for column in NUMERIC_FEATURES:
        df[column] = rng.gamma(2.0, 50.0, n).round(3)
    # payday family carries informative missingness
    df.loc[rng.random(n) < 0.15, "daystopayday"] = np.nan
    df["daystopayday"] = df["daystopayday"].clip(0, 21).round(0)

    # plant signal: risk rises with volatility/negative balances, falls with income
    risk = (
        0.8 * _z(df["balancesd"])
        + 0.7 * _z(df["maxnegativebalpast30days"])
        + 0.6 * _z(df["negbalancerate"])
        - 0.9 * _z(df["dailyincomemean"])
        - 0.5 * _z(df["plaidfeaturessummary_incomewages_lookbackwindow14d_inflow_sum"])
        + df["bankinstitution"].map(BANK_RISK).astype(float)
        + rng.normal(0, 1.0, n)
    )
    p_bad = 1.0 / (1.0 + np.exp(-(risk - 2.0)))

    if is_known:
        df["went_dpd45"] = (rng.random(n) < p_bad).astype(float)
        df["synthetic_score"] = np.nan
    else:
        df["went_dpd45"] = np.nan
        df["synthetic_score"] = np.clip(p_bad + rng.normal(0, 0.05, n), 0.001, 0.999)

    # the derived columns, exactly as base_table.sql computes them
    eps = 1e-6
    income = df["dailyincomemean"].abs().clip(lower=eps)
    df["balancesdtodailyincomemeanratio"] = df["balancesd"] / income
    df["maxnegbalance30dtodailyincomemeanratio"] = df["maxnegativebalpast30days"].abs() / income
    df["inflowsumtooutflowsumratio14d"] = (
        df["inflowsum14d"].abs() / df["outflowsum14d"].abs().clip(lower=eps)
    )
    df["netflowtodailyincomemeanratio14d"] = (
        df["inflowsum14d"].abs() - df["outflowsum14d"].abs()
    ) / (income * 14)
    df["balancedepletionrate1d"] = (
        df["balancemeanafterpayday1"] - df["balancemeanafterpayday0"]
    ) / df["highestpaydepositmean"].abs().clip(lower=eps)
    df["incomebuffertodaystopaydayratio"] = (df["balancemean"] / income) / df[
        "daystopayday"
    ].clip(lower=1)
    df["competitorborrowintensity"] = (
        df["davesummarycreditninetydayamount"].fillna(0)
        + df["earninsummarycreditninetydayamount"].fillna(0)
        + df["othercompetitorsummarycreditninetydayamount"].fillna(0)
    ) / (income * 90)
    df["balancemeantodailyincomemeanratio"] = df["balancemean"] / income
    df["odandnsffeesdailytodailyincomemeanratio"] = df["odandnsffeesdaily"] / income
    df["dailyincomeregularmeantodailyincomemeanratio"] = (
        df["dailyincomeregularmean"].abs() / income
    )
    df["istaxseason"] = pd.to_datetime(df["origination_date"]).dt.month.isin([2, 3, 4]).astype(int)

    # an excluded column — must be dropped by DataSpec.exclude_cols
    df["signupsourcetype"] = rng.choice(["ORGANIC", "PAID", "REFERRAL"], n)
    return df


def make_synthetic_daily_table(n_users: int = 400, seed: int = 11) -> pd.DataFrame:
    """User-day fixture mirroring the OOT new-links daily snapshot shape.

    One row per user per day (D1..k, k>=2), lowercase columns, with the
    metadata/outcome/policy inputs the analysis package consumes: outcomes
    on known users, synthetic + v2 scores, approval states, income/paycheck/
    activity gates, and the derived-feature inputs. Risk is planted so a
    stub model scoring on ``balancesd`` recovers signal.
    """
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(n_users):
        n_days = int(rng.integers(2, 31))
        risk = float(rng.normal(0, 1))
        is_known = bool(rng.random() < 0.6)
        first_report = np.datetime64("2026-01-01") + int(rng.integers(0, 59))
        days = np.arange(1, n_days + 1)
        p_bad = 1.0 / (1.0 + np.exp(-(risk - 1.0)))

        df = pd.DataFrame(
            {
                "user_id": f"du{i:05d}",
                "first_report_date": pd.to_datetime(first_report),
                "day_number": days,
                "snapshot_date": pd.to_datetime(first_report)
                + pd.to_timedelta(days - 1, unit="D"),
                "went_dpd45": float(rng.random() < p_bad) if is_known else np.nan,
                "is_known": is_known,
                "synthetic_score": float(np.clip(p_bad + rng.normal(0, 0.05), 0.001, 0.999)),
                "v2_score": float(np.clip(0.4 + 0.2 * risk + rng.normal(0, 0.1), 0.0, 1.0)),
                "loan_amount_max": float(rng.uniform(25, 200)),
                "bankinstitution": rng.choice(BANKS, n_days),
                "account_approval_state": rng.choice(
                    [None, "APPROVED", "NOT_APPROVED"], n_days, p=[0.6, 0.25, 0.15]
                ),
                "noactivityrate": rng.uniform(0, 1, n_days),
                "daystopayday": np.where(
                    rng.random(n_days) < 0.15, np.nan, rng.integers(0, 21, n_days)
                ),
            }
        )
        # daily feature noise around a per-user level; balancesd tracks risk
        for column in (
            "dailyincomemean",
            "balancesd",
            "maxnegativebalpast30days",
            "balancemean",
            "inflowsum14d",
            "outflowsum14d",
            "balancemeanafterpayday0",
            "balancemeanafterpayday1",
            "highestpaydepositmean",
            "davesummarycreditninetydayamount",
            "earninsummarycreditninetydayamount",
            "othercompetitorsummarycreditninetydayamount",
            "plaidfeaturessummary_incomewages_lookbackwindow30d_inflow_sum",
        ):
            base = rng.gamma(2.0, 50.0)
            df[column] = base + rng.normal(0, 5.0, n_days)
        df["balancesd"] = 100.0 + 40.0 * risk + rng.normal(0, 5.0, n_days)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def make_synthetic_ltv_table(user_ids, seed: int = 13) -> pd.DataFrame:
    """Per-user LTV fixture matching user_ltv.sql's output shape."""
    rng = np.random.default_rng(seed)
    user_ids = list(user_ids)
    covered = [u for u in user_ids if rng.random() < 0.7]
    n = len(covered)
    activated = rng.random(n) < 0.5
    max_days = rng.integers(10, 150, n)
    df = pd.DataFrame(
        {
            "user_id": covered,
            "loan_amount_max": np.where(rng.random(n) < 0.9, rng.uniform(25, 200, n), np.nan),
            "underwriting_strategy": rng.choice(
                ["UNDERWRITING_NEOBANK_STRATEGY_V3A", "OTHER_STRATEGY"], n, p=[0.7, 0.3]
            ),
            "first_activation_date": pd.Series(
                np.where(activated, np.datetime64("2026-01-15"), np.datetime64("NaT"))
            ),
            "max_linked_days": max_days,
        }
    )
    revenue_base = rng.gamma(2.0, 10.0, n) * activated
    for h in (30, 60, 90, 120):
        df[f"ltv_{h}_elig"] = max_days >= h
        df[f"total_revenue_{h}"] = revenue_base * (h / 30.0)
        df[f"total_ltv_lite_{h}"] = df[f"total_revenue_{h}"] - rng.normal(2.0, 4.0, n)
    df["is_activated"] = activated
    return df


class StubDailyModel:
    """Deterministic stand-in scorer: risk rises with balancesd."""

    def score_chunk(self, chunk: pd.DataFrame) -> np.ndarray:
        z = (chunk["balancesd"].astype(float) - 100.0) / 40.0
        return 1.0 / (1.0 + np.exp(-z))


def _hash_rank(values: pd.Series) -> pd.Series:
    keys = values.map(
        lambda v: int.from_bytes(hashlib.md5(str(v).encode()).digest()[:8], "big")
    )
    return keys.rank(method="first")


def _z(series: pd.Series) -> np.ndarray:
    values = series.to_numpy(dtype=float)
    std = values.std() or 1.0
    return (values - values.mean()) / std
