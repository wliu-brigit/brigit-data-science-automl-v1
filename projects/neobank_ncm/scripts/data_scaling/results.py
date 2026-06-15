"""Data-scaling study results: two scaling curves (synthetic axis, known axis).

Reads the run manifest (.cache/automl/fin/data_scaling_runs.json) plus the
legacy prod-replica anchor (run #1, the 20%-synthetic / full-known point), pulls
test AUC + the OOT new-links decision report (scenario-2 UW ΔAR, swap-in BR,
day2 AUC) for each, and renders the curves sorted by the axis variable (NOT by
ΔAR — this is a scaling study, not a model bake-off). Writes
.cache/automl/fin/data_scaling_results.parquet.

    uv run python projects/neobank_ncm/scripts/data_scaling/results.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

MANIFEST = Path(".cache/automl/fin/data_scaling_runs.json")
# run #1 anchor: legacy prod-replica XGBoost = full known, 20% synthetic (80/20)
ANCHOR_RUN_ID = "2f39e0ead13d4a588e4a385f272dc38f"
# Actual known rows in the TRAIN split (origination < 2025-11-01), measured.
# (The 282,642 in PROJECT_INSTRUCTIONS is the full-2025 Phase-4 retrain count,
# train+test windows — not this train-split count.) A KNOWN_SAMPLE_N at/above
# this is a no-op, so the 250k point == full.
KNOWN_FULL = 202_593


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


def _known_n(size_tok: str) -> int:
    # cap at the actual available known rows: a larger KNOWN_SAMPLE_N is a no-op
    if size_tok == "full":
        return KNOWN_FULL
    return min(int(size_tok.rstrip("k")) * 1000, KNOWN_FULL)


def main() -> None:
    _load_env()
    import pandas as pd

    from automl.mlflow import client
    from automl.mlflow.trial import artifacts
    from automl.project import use_project

    use_project("neobank_ncm")
    manifest = json.loads(MANIFEST.read_text())

    # slug -> (known_n, synth_pct); plus the anchor
    points = []
    for slug, rec in manifest.items():
        if rec.get("status") != "FINISHED" or rec.get("eval") != "ok":
            continue
        # data_scaling_known{SIZE}_synth{PCT}
        _, _, known_tok, synth_tok = slug.split("_")
        known_n = _known_n(known_tok.replace("known", ""))
        synth_pct = int(synth_tok.replace("synth", ""))
        points.append((slug, rec["run_id"], known_n, synth_pct))
    points.append(("run1_prod_replica (legacy)", ANCHOR_RUN_ID, KNOWN_FULL, 20))

    rows = []
    for label, run_id, known_n, synth_pct in points:
        run = client.raw().get_run(run_id)
        test_auc = run.data.metrics.get("eval.test.auc")
        m = {x["name"]: x["value"] for x in artifacts.load_eval(run_id, "oot_new_links").metrics}
        uw = m["decision_report"]["scenarios"]["2_income500_match_bad_rate"]["tracks"]["uw"]
        rows.append({
            "point": label,
            "known_n": known_n,
            "synth_pct": synth_pct,
            "test_auc": round(test_auc, 4) if test_auc is not None else None,
            "day2_auc": round(m["day2_known_auc"], 4),
            "approval_gain_pct": round(uw["approval_rate_delta"] * 100, 2),
            "swap_in_BR": round(uw["swap_in_bad_rate"], 4),
            "run_id": run_id,
        })
    df = pd.DataFrame(rows)
    out = Path(".cache/automl/fin/data_scaling_results.parquet")
    df.to_parquet(out)

    cols = ["known_n", "synth_pct", "test_auc", "day2_auc", "approval_gain_pct", "swap_in_BR"]

    def _render(title, sub):
        print(f"\n### {title}")
        print("| " + " | ".join(cols) + " |")
        print("|" + "|".join(["---"] * len(cols)) + "|")
        for _, r in sub.iterrows():
            print("| " + " | ".join(str(r[c]) for c in cols) + " |")

    # 250k == full (KNOWN_SAMPLE_N >= available known is a no-op): dedupe the
    # identical (known_n, synth_pct) configs for a clean table.
    df = df.drop_duplicates(["known_n", "synth_pct"]).reset_index(drop=True)
    # Study A — synthetic axis (known = full), sorted by synth%
    a = df[df["known_n"] == KNOWN_FULL].sort_values("synth_pct")
    _render(f"Study A — synthetic axis (known = full = {KNOWN_FULL:,})", a)
    # Study B — known axis (synth = 0%), sorted by known_n
    b = df[df["synth_pct"] == 0].sort_values("known_n")
    _render("Study B — known axis (synthetic = 0%)", b)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
