"""Feature due diligence for the v2 feature base (dataset v1_76d3ad45).

Phase-1 gate before the unsupervised discovery run. An anomaly model has no
label to "notice" a leak the way AP-vs-proxy did, so any outcome-adjacent or
near-label column in the active feature space silently contaminates the score.
This script enumerates the materialized columns, diffs against the old dataset
to isolate what the v2 build added, and prints the exact feature space the
unsupervised model would consume (numeric + bool, flag=feature) so the
exclude/metadata decisions can be made at config level (no 3.3TB rebuild).

    uv run python -m projects.fraud_anomaly_detection.analysis.feature_due_diligence
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

NEW_DATASET = "v1_76d3ad45"
OLD_DATASET = "v1_42baf0ba"

# Substrings that mark a column as outcome-derived / leakage-prone. Used only to
# FLAG for human review here, not to auto-exclude.
OUTCOME_HINTS = (
    "dpd", "mature", "charge_off", "days_past_due", "repaid", "loan_status",
    "expected_dpd", "gross_dpd", "heuristic_fraud",
)


def _load_env() -> None:
    p = Path(".env")
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _flag_of(registry, col: str) -> str:
    for flag in ("feature", "metadata", "excluded", "target"):
        try:
            if col in set(registry.get_by_flag(flag)):
                return flag
        except Exception:
            pass
    return "?"


def main() -> None:
    _load_env()
    from automl.data.registry import load_dataset_by_id
    from automl.project.session import use_project

    # The full v2 build (with the Tier-1 features) is registered in the
    # NON-dry-run scope; the dry-run scope still holds the old v1_42baf0ba.
    sess = use_project("fraud_anomaly_detection", dry_run=False)

    loaded = load_dataset_by_id(NEW_DATASET, session=sess)
    df, registry = loaded.df, loaded.registry
    print(f"=== {NEW_DATASET}: {df.shape[0]:,} rows x {df.shape[1]} cols ===\n")

    # Column set diff vs the old dataset (dry-run scope; best-effort).
    old_cols: set[str] = set()
    try:
        sess_old = use_project("fraud_anomaly_detection", dry_run=True)
        old_cols = set(load_dataset_by_id(OLD_DATASET, session=sess_old).df.columns)
    except Exception as e:  # pragma: no cover - diagnostic only
        print(f"(could not load {OLD_DATASET} for diff: {e})\n")
    new_cols = set(df.columns)
    if old_cols:
        added = sorted(new_cols - old_cols)
        removed = sorted(old_cols - new_cols)
        print(f"--- ADDED in {NEW_DATASET} ({len(added)}) ---")
        for c in added:
            print(f"  + {c}")
        print(f"\n--- REMOVED vs {OLD_DATASET} ({len(removed)}) ---")
        for c in removed:
            print(f"  - {c}")
        print()

    # What flag does the registry assign each column, and is it numeric/bool
    # (i.e. would an unsupervised num+bool model consume it)?
    n = len(df)
    rows = []
    for c in sorted(df.columns):
        s = df[c]
        nulls = float(s.isna().mean())
        try:
            nuniq = int(s.nunique(dropna=True))
        except TypeError:
            nuniq = -1
        flag = _flag_of(registry, c)
        hinted = any(h in c.lower() for h in OUTCOME_HINTS)
        rows.append({
            "column": c, "dtype": str(s.dtype), "flag": flag,
            "null_frac": nulls, "n_unique": nuniq,
            "outcome_hint": hinted,
        })
    inv = pd.DataFrame(rows)

    # The active feature space the unsupervised model would see.
    num_cols = set(registry.get_by_dtype("num", flag="feature"))
    bool_cols = set(registry.get_by_dtype("bool", flag="feature"))
    feature_space = sorted(num_cols | bool_cols)

    print(f"--- FULL INVENTORY ({len(inv)} cols) ---")
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(inv.to_string(index=False))

    print(f"\n--- ACTIVE UNSUPERVISED FEATURE SPACE (num+bool, flag=feature): {len(feature_space)} ---")
    for c in feature_space:
        hint = "  <-- OUTCOME HINT" if any(h in c.lower() for h in OUTCOME_HINTS) else ""
        print(f"  {c}{hint}")

    # Loud check: any feature-flagged column that smells like an outcome.
    leaky = inv[(inv["flag"] == "feature") & (inv["outcome_hint"])]
    print("\n--- FEATURE-FLAGGED COLUMNS MATCHING AN OUTCOME HINT (review!) ---")
    print(leaky.to_string(index=False) if len(leaky) else "  (none)")

    # Constant / near-constant features add noise to anomaly geometry.
    const = inv[(inv["flag"] == "feature") & (inv["n_unique"] <= 1)]
    print("\n--- CONSTANT FEATURE COLUMNS (n_unique<=1) ---")
    print(const.to_string(index=False) if len(const) else "  (none)")

    # name_match_official: the known noise column the materialized data still carries.
    print("\n--- name_match_official present? ---")
    print(f"  {'YES (in dataframe)' if 'name_match_official' in new_cols else 'no'}")


if __name__ == "__main__":
    main()
