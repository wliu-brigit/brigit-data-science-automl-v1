"""One-time materialization of the oot_new_links_with_ltv external EvalDataset.

Model-INDEPENDENT: the daily scoring frame + derived features + per-user LTV
(broadcast per daily row). Every trial's decision re-eval points at this one
dataset (see docs/to-do/native-decision-reeval-plan.md). Idempotent — re-running
returns the existing id unless --overwrite.

Run OFF-VPN if possible (writes ~GB to GCS; VPN throttles GCS). Cached frames are
reused from .cache/automl/fin/; no Snowflake/VPN needed for the data itself.

    uv run python projects/neobank_ncm/scripts/reeval/prepare_oot_new_links_dataset.py
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

CACHE = Path(".cache/automl/fin")


def _load_env() -> None:
    """Set repo .env (GCS/MLflow creds + REQUESTS_CA_BUNDLE/SSL_CERT_FILE for the
    VPN TLS intercept) into the process before any TLS client initializes."""
    envp = Path(".env")
    if not envp.exists():
        return
    for line in envp.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            if k.strip().isidentifier():
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    _load_env()

    import pandas as pd

    from automl.eval import prepare_eval_dataset
    from automl.project import use_project
    from projects.neobank_ncm.analysis import impact, scoring

    session = use_project("neobank_ncm")

    daily = pd.read_parquet(CACHE / "daily.parquet")
    daily.columns = [c.lower() for c in daily.columns]
    daily["is_known"] = daily["went_dpd45"].notna()
    scoring.add_daily_derived_features(daily)  # model expects these as inputs

    ltv = pd.read_parquet(CACHE / "user_ltv.parquet")
    ltv.columns = [c.lower() for c in ltv.columns]
    ltv_cols = (
        ["user_id", "loan_amount_max", "underwriting_strategy", "first_activation_date"]
        + [f"total_revenue_{h}" for h in impact.HORIZONS]
        + [f"total_ltv_lite_{h}" for h in impact.HORIZONS]
        + [f"ltv_{h}_elig" for h in impact.HORIZONS]
    )
    # ``loan_amount_max`` exists in BOTH frames (daily _EXTRA_COLS + ltv); the LTV
    # frame's value is the one impact.merge_ltv/build_lookup expect, so drop daily's
    # copy to avoid a _x/_y suffix collision that would hide the required column.
    daily = daily.drop(columns=["loan_amount_max"], errors="ignore")
    frame = daily.merge(ltv[ltv_cols], on="user_id", how="left")  # broadcast LTV per daily row

    ds, existed = prepare_eval_dataset(
        session=session,
        kind="external",
        frame=frame,
        target_col="went_dpd45",
        unique_key=("user_id", "day_number"),
        provenance={"population": "oot_new_links_with_ltv", "ltv_pull_date": "2026-06-11"},
        overwrite=args.overwrite,
    )
    print(f"eval_dataset_id={ds.id}  (existed={existed})  rows={len(frame)}  cols={frame.shape[1]}")


if __name__ == "__main__":
    main()
