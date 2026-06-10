"""Financial impact: LTV merge, imputation, decomposition, sample size.

Faithful port of cells 19–28 of the legacy financial_impact_analysis.ipynb.
Frames are lowercase-named; user_scores comes from policy.collapse_to_users
and ltv_data from data.load_user_ltv (or the test fixture).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

HORIZONS = [30, 60, 90, 120]
N_BINS = 30  # synthetic-score quantile bins for the lookup table (cell 22)
V3A_STRATEGY = "UNDERWRITING_NEOBANK_STRATEGY_V3A"
N_MONTHS_OOT = 2  # Jan–Feb 2026 window (cell 20)

_LTV_COLS = (
    ["user_id", "loan_amount_max", "underwriting_strategy", "is_activated"]
    + [f"total_revenue_{h}" for h in HORIZONS]
    + [f"total_ltv_lite_{h}" for h in HORIZONS]
    + [f"ltv_{h}_elig" for h in HORIZONS]
)


# ── Merge (legacy cell 19) ───────────────────────────────────────────────────
def merge_ltv(user_scores: pd.DataFrame, ltv_data: pd.DataFrame) -> pd.DataFrame:
    """user_scores ⟕ ltv_data with the legacy fill rules + loss columns."""
    us = user_scores.merge(ltv_data[_LTV_COLS], on="user_id", how="left")
    us["is_activated"] = us["is_activated"].fillna(False).astype(bool)
    for h in HORIZONS:
        us[f"total_revenue_{h}"] = us[f"total_revenue_{h}"].fillna(0.0)
        us[f"total_ltv_lite_{h}"] = us[f"total_ltv_lite_{h}"].fillna(0.0)
        us[f"loss_{h}"] = us[f"total_revenue_{h}"] - us[f"total_ltv_lite_{h}"]
    return us


# ── Historical reference (legacy cell 20) ────────────────────────────────────
def historical_reference(us: pd.DataFrame) -> dict:
    """V3A per-activation metrics, ACT_RATE and monthly volume."""
    n = len(us)
    act_us = us[us["is_activated"] & (us["underwriting_strategy"] == V3A_STRATEGY)].copy()
    out: dict = {
        "n": n,
        "monthly_vol": n / N_MONTHS_OOT,
        "act_us": act_us,
    }
    for h in HORIZONS:
        elig = act_us[f"ltv_{h}_elig"].astype(bool)
        out[f"n_elig_{h}"] = int(elig.sum())
        out[f"mean_rev_{h}"] = act_us.loc[elig, f"total_revenue_{h}"].mean()
        out[f"mean_loss_{h}"] = act_us.loc[elig, f"loss_{h}"].mean()
        out[f"mean_ltv_{h}"] = act_us.loc[elig, f"total_ltv_lite_{h}"].mean()
        # LTV per link = mean LTV per activation x (n elig activated / N links)
        out[f"lpl_{h}"] = out[f"mean_ltv_{h}"] * out[f"n_elig_{h}"] / n
    approved = us[us["v3a_approved"] | us["cle_approved"]]
    out["act_rate"] = float(approved["is_activated"].mean())
    out["n_activations_approved"] = int(approved["is_activated"].sum())
    return out


# ── Lookup table (legacy cell 22) ────────────────────────────────────────────
def build_lookup(us: pd.DataFrame, n_bins: int = N_BINS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(us with score_bin column, per-bin lookup of rev / loss-rate / ltv).

    Bins are synthetic-score quantiles over the full population; the lookup
    values come from activated users with enough days at each horizon;
    sparse bins fall back to the global (activated) mean.
    """
    us = us.copy()
    score_valid = us["synthetic_score"].dropna().clip(0, 1)
    _, edges = pd.qcut(score_valid, q=n_bins, retbins=True, duplicates="drop")
    edges[0] = -np.inf
    edges[-1] = np.inf
    us["score_bin"] = pd.cut(
        us["synthetic_score"].clip(0, 1), bins=edges, labels=False, include_lowest=True
    )

    act = us[us["is_activated"]].copy()
    lam_den = act["loan_amount_max"].fillna(act["loan_amount_max"].median()).clip(lower=1)
    for h in HORIZONS:
        act[f"_lr_{h}"] = act[f"loss_{h}"] / lam_den  # loss rate per $ of limit

    parts = []
    for h in HORIZONS:
        elig = act[f"ltv_{h}_elig"].astype(bool)
        parts.append(
            act[elig]
            .groupby("score_bin")
            .agg(
                **{
                    f"rev_{h}": (f"total_revenue_{h}", "mean"),
                    f"lr_{h}": (f"_lr_{h}", "mean"),
                    f"ltv_{h}": (f"total_ltv_lite_{h}", "mean"),
                }
            )
        )
    lookup = pd.concat(parts, axis=1)
    for h in HORIZONS:  # sparse bins -> global mean over the activated population
        for kind in ("rev", "lr", "ltv"):
            col = f"{kind}_{h}"
            lookup[col] = lookup[col].fillna(lookup[col].mean())
    return us, lookup


def infer_financials(
    us_binned: pd.DataFrame,
    lookup: pd.DataFrame,
    approved_mask: pd.Series,
    lam: pd.Series,
) -> pd.DataFrame:
    """Per-approved-user revenue/loss/LTV (legacy cell 23 ``_infer``).

    Activated users with enough days at a horizon keep their actuals;
    everyone else gets the score-bin lookup, with loss scaled to the
    approval loan amount.
    """
    sub = us_binned.loc[approved_mask].copy()
    lam_values = lam[approved_mask].values
    activated = sub["is_activated"].values
    bins = sub["score_bin"]
    for h in HORIZONS:
        look_rev = bins.map(lookup[f"rev_{h}"]).fillna(lookup[f"rev_{h}"].mean()).values
        look_lr = bins.map(lookup[f"lr_{h}"]).fillna(lookup[f"lr_{h}"].mean()).values
        elig = sub[f"ltv_{h}_elig"].astype(bool).values
        use_actual = activated & elig
        sub[f"inf_rev_{h}"] = np.where(use_actual, sub[f"total_revenue_{h}"].values, look_rev)
        sub[f"inf_loss_{h}"] = np.where(use_actual, sub[f"loss_{h}"].values, look_lr * lam_values)
        sub[f"inf_ltv_{h}"] = sub[f"inf_rev_{h}"] - sub[f"inf_loss_{h}"]
    return sub


# ── Scenario monthly aggregate (legacy cell 23 ``_agg``) ─────────────────────
def monthly_aggregate(
    inf_df: pd.DataFrame,
    n: int,
    act_rate: float,
    monthly_links: float,
) -> dict[str, float]:
    """Approval/activation volumes and per-horizon monthly financials."""
    n_app = len(inf_df)
    ar = n_app / n
    mo_app = ar * monthly_links
    mo_act = mo_app * act_rate
    out: dict[str, float] = dict(
        n_app=n_app,
        n_act=int(round(n_app * act_rate)),
        ar=ar,
        mo_app=mo_app,
        mo_act=mo_act,
    )
    for h in HORIZONS:
        out[f"mo_rev_{h}"] = mo_act * inf_df[f"inf_rev_{h}"].mean()
        out[f"mo_loss_{h}"] = mo_act * inf_df[f"inf_loss_{h}"].mean()
        out[f"mo_ltv_{h}"] = mo_act * inf_df[f"inf_ltv_{h}"].mean()
        out[f"lpl_{h}"] = out[f"mo_ltv_{h}"] / monthly_links if monthly_links else float("nan")
        out[f"ltv_per_act_{h}"] = out[f"mo_ltv_{h}"] / mo_act if mo_act else float("nan")
    return out


# ── Revenue decomposition (legacy cell 24) ───────────────────────────────────
def revenue_decomposition(act_us: pd.DataFrame, v3_inf: pd.DataFrame) -> dict:
    """Reference-vs-proposed per-activation revenue, split into four groups.

    overlap (activated under both), newly-activated under V3, non-activated
    (lookup-imputed), and V3A users dropped by V3. Index alignment matters:
    both frames must share the merged user table's index.
    """
    act_idx = set(act_us.index)
    v3_act_idx = set(v3_inf[v3_inf["is_activated"]].index)
    v3_nonact_idx = set(v3_inf[~v3_inf["is_activated"]].index)

    groups = {
        "overlap": sorted(act_idx & v3_act_idx),
        "new": sorted(v3_act_idx - act_idx),
        "nonact": sorted(v3_nonact_idx),
        "dropped": sorted(act_idx - set(v3_inf.index)),
    }
    n_ref, n_v3 = len(act_us), len(v3_inf)
    counts = {name: len(idx) for name, idx in groups.items()}

    def _mean(idx: list, df: pd.DataFrame, col: str, elig_col: str | None) -> float:
        if not idx:
            return 0.0
        sub = df.loc[idx]
        if elig_col is not None:
            elig = sub[elig_col].astype(bool)
            return float(sub.loc[elig, col].mean()) if elig.any() else 0.0
        return float(sub[col].mean())

    per_group_rev = {}
    decomposition = {}
    for h in HORIZONS:
        ref_col, v3_col, elig_col = f"total_revenue_{h}", f"inf_rev_{h}", f"ltv_{h}_elig"
        rev = {
            "ref": _mean(list(act_idx), act_us, ref_col, elig_col),
            "v3": float(v3_inf[v3_col].mean()),
            "overlap": _mean(groups["overlap"], act_us, ref_col, elig_col),
            "new": _mean(groups["new"], v3_inf, v3_col, None),
            "nonact": _mean(groups["nonact"], v3_inf, v3_col, None),
            "dropped": _mean(groups["dropped"], act_us, ref_col, elig_col),
        }
        per_group_rev[h] = rev
        effects = {
            "overlap_effect": rev["overlap"] * (counts["overlap"] / n_v3 - counts["overlap"] / n_ref),
            "new_effect": rev["new"] * counts["new"] / n_v3,
            "nonact_effect": rev["nonact"] * counts["nonact"] / n_v3,
            "dropped_effect": -rev["dropped"] * counts["dropped"] / n_ref,
        }
        effects["total_delta"] = rev["v3"] - rev["ref"]
        effects["check"] = sum(
            v for k, v in effects.items() if k.endswith("_effect")
        )
        decomposition[h] = effects

    return {
        "n_ref": n_ref,
        "n_v3": n_v3,
        "counts": counts,
        "groups": groups,
        "per_group_revenue": pd.DataFrame(per_group_rev).T.rename_axis("horizon"),
        "decomposition": pd.DataFrame(decomposition).T.rename_axis("horizon"),
    }


# ── Sample size (legacy cell 28) ─────────────────────────────────────────────
def sample_size_analysis(
    us_binned: pd.DataFrame,
    v3a_inf: pd.DataFrame,
    v3_inf: pd.DataFrame,
    p_control: float,
    p_treatment: float,
    monthly_vol: float,
    *,
    alpha: float = 0.05,
    power: float = 0.80,
    ltv_horizons: tuple[int, ...] = (90, 120),
    sampling_rates: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20),
) -> dict:
    """Experiment sizing: AR z-test + LTV/link mean tests; binding = max."""
    n = len(us_binned)
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)

    def _ltv_arr(inf_df: pd.DataFrame, h: int) -> np.ndarray:
        arr = pd.Series(np.zeros(n), index=us_binned.index)
        arr.loc[inf_df.index] = inf_df[f"inf_ltv_{h}"].values
        return arr.values

    tests: dict[int, dict[str, float]] = {}
    for h in ltv_horizons:
        ctrl = _ltv_arr(v3a_inf, h)
        trt = _ltv_arr(v3_inf, h)
        tests[h] = dict(
            mu_ctrl=float(ctrl.mean()),
            mu_trt=float(trt.mean()),
            sigma2=float(max(np.var(ctrl, ddof=1), np.var(trt, ddof=1))),
            ci_ctrl=float(z_a * np.std(ctrl, ddof=1) / np.sqrt(n)),
            ci_trt=float(z_a * np.std(trt, ddof=1) / np.sqrt(n)),
            ci_diff=float(
                z_a * np.sqrt(np.var(ctrl, ddof=1) / n + np.var(trt, ddof=1) / n)
            ),
        )

    ar_trivial = abs(p_treatment - p_control) < 1e-4
    h_ar = (
        2 * (np.arcsin(np.sqrt(p_treatment)) - np.arcsin(np.sqrt(p_control)))
        if not ar_trivial
        else 0.0
    )
    n_per_test = {
        "(A) Approval rate": ((z_a + z_b) / h_ar) ** 2 if not ar_trivial else 0.0
    }
    for label_idx, h in enumerate(ltv_horizons):
        delta = tests[h]["mu_trt"] - tests[h]["mu_ctrl"]
        n_per_test[f"({'BC'[label_idx]}) LTV/link D{h}"] = (
            2 * tests[h]["sigma2"] * (z_a + z_b) ** 2 / delta**2
            if abs(delta) > 1e-4
            else 0.0
        )

    binding_label = max(n_per_test, key=n_per_test.get)
    n_total = 2 * n_per_test[binding_label]
    months = pd.DataFrame(
        [
            {
                "holdout_rate": rate,
                "monthly_sampled": monthly_vol * rate,
                "months_needed": n_total / (monthly_vol * rate)
                if monthly_vol * rate
                else float("inf"),
            }
            for rate in sampling_rates
        ]
    )
    return {
        "alpha": alpha,
        "power": power,
        "z_a": float(z_a),
        "z_b": float(z_b),
        "p_control": p_control,
        "p_treatment": p_treatment,
        "ltv_tests": tests,
        "n_per_test": n_per_test,
        "binding": binding_label,
        "n_binding": n_per_test[binding_label],
        "n_total": n_total,
        "months_table": months,
    }
