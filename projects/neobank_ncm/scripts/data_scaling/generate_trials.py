"""Generate the data-scaling trial dirs (prunable one-shot setup).

Each trial is the faithful legacy XGBoost (model/baseline.py) with ONLY the
two data-size knobs changed: KNOWN_SAMPLE_N and UNKNOWN_FRAC. See
docs/superpowers/specs/2026-06-13-neobank-ncm-data-scaling-design.md.

    uv run python projects/neobank_ncm/scripts/data_scaling/generate_trials.py
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

EXP_DIR = (
    Path(__file__).resolve().parents[2]  # .../scripts/data_scaling/<file> -> project dir
    / "experiments" / "neobank_ncm" / "neobank_ncm_v3_replicate"
)

# (slug, KNOWN_SAMPLE_N, UNKNOWN_FRAC, hypothesis)
TRIALS = [
    # Study A — synthetic axis (known = full; 20% = run #1, not re-run)
    ("data_scaling_knownfull_synth00", None, 0.0,
     "Data-scaling synthetic axis: full known, 0% synthetic (known-only). Vs "
     "run #1 (20% synthetic) and the 10/30% points, does reject-inference "
     "synthetic data move scenario-2 UW ΔAR / swap-in BR under the new "
     "decision metrics (the old AUC-era finding was 'doesn't matter much')?"),
    ("data_scaling_knownfull_synth10", None, 0.10,
     "Data-scaling synthetic axis: full known, 10% synthetic share."),
    ("data_scaling_knownfull_synth30", None, 0.30,
     "Data-scaling synthetic axis: full known, 30% synthetic share."),
    # Study B — known axis (synthetic = 0%; full point shared with Study A)
    ("data_scaling_known050k_synth00", 50_000, 0.0,
     "Data-scaling known axis: 50K known, 0% synthetic. How does boosting "
     "scale with known ground-truth data on the decision metrics?"),
    ("data_scaling_known100k_synth00", 100_000, 0.0,
     "Data-scaling known axis: 100K known, 0% synthetic."),
    ("data_scaling_known150k_synth00", 150_000, 0.0,
     "Data-scaling known axis: 150K known, 0% synthetic."),
    ("data_scaling_known200k_synth00", 200_000, 0.0,
     "Data-scaling known axis: 200K known, 0% synthetic."),
    ("data_scaling_known250k_synth00", 250_000, 0.0,
     "Data-scaling known axis: 250K known, 0% synthetic."),
]

MODEL_TEMPLATE = '''\
"""Data-scaling trial: {slug}.

{hypothesis}

Faithful legacy XGBoost (model/baseline.py) with ONLY the data-size knobs
changed. Everything else — locked 162-feature set, WoE(bank), monotone
constraints, dual-record soft labels, locked XGBoost params, random_state=42 —
is run #1.
"""
from __future__ import annotations

from projects.neobank_ncm.model.baseline import NeobankNCMReplicationModel


class DataScalingModel(NeobankNCMReplicationModel):
    KNOWN_SAMPLE_N = {known_n}   # None = all known rows
    UNKNOWN_FRAC = {unknown_frac}   # unknown's share of the training mix; 0.0 = known-only


MODEL_CLASS = DataScalingModel
'''

RUN_PY = '''\
from __future__ import annotations

import sys
from pathlib import Path

from automl import runner


def _field(result, name, default=""):
    if isinstance(result, dict):
        return result.get(name, default)
    return getattr(result, name, default)


def _status_value(status):
    return getattr(status, "value", status)


if __name__ == "__main__":
    result = runner.run_trial(Path(__file__).parent)
    status = _status_value(_field(result, "status"))
    metrics = _field(result, "metrics", {}) or {}
    primary = next(iter(metrics.values()), "")
    error = _field(result, "error", "") or ""
    print(f"AUTOML_STATUS={status}", flush=True)
    print(f"AUTOML_TRIAL_ID={_field(result, 'trial_id')}", flush=True)
    print(f"AUTOML_RUN_ID={_field(result, 'run_id')}", flush=True)
    print(f"AUTOML_PRIMARY={primary}", flush=True)
    if error:
        print(f"AUTOML_ERROR={error}", flush=True)
    sys.exit(0 if status == "FINISHED" else 1)
'''


def main() -> None:
    now = datetime.now(UTC).isoformat()
    for slug, known_n, unknown_frac, hypothesis in TRIALS:
        d = EXP_DIR / slug
        d.mkdir(parents=True, exist_ok=True)
        (d / "model.py").write_text(
            MODEL_TEMPLATE.format(
                slug=slug, hypothesis=hypothesis,
                known_n=known_n, unknown_frac=unknown_frac,
            )
        )
        (d / "run.py").write_text(RUN_PY)
        (d / "metadata.json").write_text(json.dumps({
            "schema_version": 1,
            "slug": slug,
            "strategy": "data_scaling",
            "hypothesis": hypothesis,
            "training_origin": "human",
            "created_at": now,
            "project_name": "neobank_ncm",
            "project_package": "projects.neobank_ncm",
            "experiment_id": "neobank_ncm_v3_replicate",
            "seed": None,
        }, indent=2))
        print(f"wrote {d}  (KNOWN_SAMPLE_N={known_n}, UNKNOWN_FRAC={unknown_frac})")


if __name__ == "__main__":
    main()
