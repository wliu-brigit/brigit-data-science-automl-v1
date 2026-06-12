"""Assemble the cross-trial decision comparison from each trial's report.json.

Reads eval/oot_new_links/report.json off each trial run (via the artifacts
seam), pulls the headline scenario-2 UW numbers + day2_known_auc, and writes
.cache/automl/fin/decision_comparison.parquet with the settled column names.
See docs/to-do/native-decision-reeval-plan.md (Task 5).
"""
from __future__ import annotations

import os
from pathlib import Path


def _load_env() -> None:
    envp = Path(".env")
    if not envp.exists():
        return
    for line in envp.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            if k.strip().isidentifier():
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


# trial -> run_id (from docs/HANDOFF.md; trial 7 excluded — not deployable)
TRIALS = {
    1: "51bd38d4bcb845cbbad52dcacd637e1e",
    3: "2f39e0ead13d4a588e4a385f272dc38f",
    11: "b3e5efdb9a924157b4ca521022ccf816",
    12: "9c6b2e176e9f45888d7489be3e38aedc",
    13: "e4ea3b7256924bdc83e942e79bb85715",
}


def main() -> None:
    _load_env()

    import pandas as pd

    from automl.mlflow.trial import artifacts
    from automl.project import use_project

    use_project("neobank_ncm")
    rows = []
    for trial, run_id in TRIALS.items():
        result = artifacts.load_eval(run_id, "oot_new_links")
        metrics = {m["name"]: m["value"] for m in result.metrics}
        rep = metrics["decision_report"]
        sc = rep["scenarios"]["2_income500_match_bad_rate"]
        uw = sc["tracks"]["uw"]
        rows.append({
            "trial": trial,
            "day2_known_auc": metrics["day2_known_auc"],
            "candidate_approval_rate": uw["candidate_approval_rate"],
            "approval_rate_delta": uw["approval_rate_delta"],
            "swap_in_bad_rate": uw["swap_in_bad_rate"],
            "ltv_per_link_d90": sc["ltv_per_link_d90"],
            "ltv_per_link_d120": sc["ltv_per_link_d120"],
        })
    df = pd.DataFrame(rows).sort_values("trial").reset_index(drop=True)
    out = Path(".cache/automl/fin/decision_comparison.parquet")
    df.to_parquet(out)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
