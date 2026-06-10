"""Refresh the scenario register's validation stats from the active snapshot.

Read-only against MLflow/GCS (loads the active materialized dataset, full
frame, no split slice); the only thing written is the second, machine-owned
YAML document at the bottom of register.yaml — the hand-written register
document above it (comments included) is never touched. Run it whenever a
scenario rule or a stat definition changes, so the register always carries
current evidence next to the definitions:

    uv run python -m projects.fraud_anomaly_detection.scenarios.validation

Per scenario: match count and share, never-paid-DPD45 rate on RESOLVED rows
(gross DPD45 and not repaid as of snapshot — the bust-out cut, same as the
eval metrics; denominator = repaid + never_paid, matching the
scenarios/backtest definition), plain gross-DPD45 rate on mature rows, and the
heuristic-band distribution of matched rows (how many the heuristic already
flagged vs called clean).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from projects.fraud_anomaly_detection.scenarios.engine import Scenario, evaluate
from projects.fraud_anomaly_detection.scenarios import REGISTER_PATH

_STATS_DOC_HEADER = (
    "---\n"
    "# ── validation stats — machine-owned; refresh with ──────────────────────────\n"
    "# uv run python -m projects.fraud_anomaly_detection.scenarios.validation\n"
)


def _rate(numerator: int, denominator: int) -> float | None:
    return (numerator / denominator) if denominator else None


def compute_stats(df: pd.DataFrame, scenarios: Sequence[Scenario]) -> dict[str, Any]:
    """Validation stats over the given frame: baseline, per-scenario, overall.

    Scenarios are matched independently (an entity may match several; overlap
    is a signal, not double-counting — SCENARIOS.md). Sequencing-free bounds
    per scenario: gross capture (n) is its best-case contribution, unique
    capture (unique_n — rows no other scenario matches) is its worst-case
    marginal contribution if every other rule fired first.
    """
    flags = evaluate(df, scenarios)
    mature = (df["label_mature_d45"].astype(float) == 1).to_numpy()
    dpd45 = (df["label_gross_dpd45"].astype(float) == 1).to_numpy()
    never_paid = dpd45 & (df["label_repaid_current_snapshot"].astype(float) == 0).to_numpy()
    repaid = (df["label_repaid_current_snapshot"].astype(float) == 1).to_numpy()
    # "resolved" = the advance reached a verdict — repaid, or matured & DPD45
    # (went bad); still-open advances excluded. never_paid_rate uses this
    # denominator to match the month-over-month backtest (scenarios/backtest);
    # on a mostly-matured snapshot it is ~identical to the matured denominator.
    resolved = repaid | (mature & dpd45)
    bands_col = df["heuristic_fraud_band"].astype(str).to_numpy()
    matched_by = {s.name: flags[f"scenario_{s.name}"].to_numpy() for s in scenarios}
    any_matched = flags["scenario_any"].to_numpy()
    match_counts = sum(matched_by.values()) if matched_by else any_matched.astype(int)

    def outcome_block(mask) -> dict[str, Any]:
        m_mature = mask & mature
        n_mature = int(m_mature.sum())
        n_never_paid = int((m_mature & never_paid).sum())
        n_resolved = int((mask & resolved).sum())
        return {
            "n": int(mask.sum()),
            "share": _rate(int(mask.sum()), len(df)),
            "n_mature": n_mature,
            "n_never_paid": n_never_paid,
            "n_resolved": n_resolved,
            # resolved denominator (repaid + never_paid), matching the backtest
            "never_paid_rate": _rate(n_never_paid, n_resolved),
        }

    # ── baseline: what everything gets compared against ──
    base_never_paid_rate = _rate(int((never_paid & mature).sum()), int(resolved.sum()))
    baseline = {
        "n_mature": int(mature.sum()),
        "never_paid_rate": base_never_paid_rate,
        "dpd45_rate": _rate(int(dpd45[mature].sum()), int(mature.sum())),
        "bands": {
            band: outcome_block(bands_col == band)
            for band in pd.unique(bands_col)
        },
    }

    # ── per scenario: gross + unique capture, lift vs base ──
    per_scenario: dict[str, Any] = {}
    for scenario in scenarios:
        matched = matched_by[scenario.name]
        matched_mature = matched & mature
        n_dpd45 = int((matched_mature & dpd45).sum())
        unique = matched & (match_counts == 1)
        unique_block = outcome_block(unique)
        block = outcome_block(matched)
        bands = pd.Series(bands_col[matched]).value_counts()
        per_scenario[scenario.name] = {
            "tier": scenario.tier,
            "status": scenario.status,
            "entry_date": scenario.entry_date,
            **block,
            "n_dpd45": n_dpd45,
            "dpd45_rate": _rate(n_dpd45, block["n_mature"]),
            "lift_vs_base": (
                block["never_paid_rate"] / base_never_paid_rate
                if block["never_paid_rate"] is not None and base_never_paid_rate
                else None
            ),
            "unique_n": unique_block["n"],
            "unique_never_paid_rate": unique_block["never_paid_rate"],
            "band_distribution": {band: int(count) for band, count in bands.items()},
        }

    # ── overall: union, overlap, residual (incl. per-band coverage) ──
    pair_overlap = {
        f"{a} & {b}": int((matched_by[a] & matched_by[b]).sum())
        for i, a in enumerate(matched_by)
        for b in list(matched_by)[i + 1 :]
        if int((matched_by[a] & matched_by[b]).sum())
    }
    residual_bands = {}
    for band in pd.unique(bands_col):
        in_band = bands_col == band
        captured = int((in_band & any_matched).sum())
        residual_bands[band] = {
            "n_left": int(in_band.sum()) - captured,
            "coverage": _rate(captured, int(in_band.sum())),
        }
    overall = {
        "union": outcome_block(any_matched),
        # the real win: captured rows the heuristic called clean (LOW band) —
        # matching the elevated bands is coverage; this is genuine discovery
        "discovery": outcome_block(any_matched & (bands_col == "LOW")),
        "overlap": {"n_multi_matched": int((match_counts > 1).sum()), "pairs": pair_overlap},
        "residual": {**outcome_block(~any_matched), "bands": residual_bands},
    }
    return {
        "n_rows": int(len(df)),
        "baseline": baseline,
        "scenarios": per_scenario,
        "overall": overall,
    }


def write_stats(stats: dict[str, Any], *, path: Path = REGISTER_PATH) -> None:
    """Rewrite the machine-owned stats document, leaving the register untouched.

    The register file is a two-document YAML stream: doc 1 is hand-written,
    doc 2 (everything from the first document separator on) is ours to
    regenerate wholesale.
    """
    text = path.read_text()
    register_doc = text.split("\n---\n", 1)[0]
    if not register_doc.endswith("\n"):
        register_doc += "\n"
    stats_doc = yaml.safe_dump(stats, sort_keys=False, default_flow_style=False)
    path.write_text(register_doc + _STATS_DOC_HEADER + stats_doc)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-dry-run",
        action="store_true",
        help="use the full-scale scope instead of the dry-run scope",
    )
    args = parser.parse_args()

    import automl
    import automl.data as data
    from projects.fraud_anomaly_detection.scenarios import SCENARIOS, SCENARIOS_VERSION

    active = automl.use_project("fraud_anomaly_detection", dry_run=not args.no_dry_run)
    loaded = data.load_dataset(session=active)  # full frame, no split slice
    stats = {
        "computed_at": dt.date.today().isoformat(),
        "dataset_id": loaded.dataset.id,
        "scenarios_version": SCENARIOS_VERSION,
        **compute_stats(loaded.df, SCENARIOS),
    }
    write_stats(stats)
    print(f"updated {REGISTER_PATH}")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
