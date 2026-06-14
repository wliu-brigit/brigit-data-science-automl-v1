# Fraud Control Skeleton Handoff

Updated: 2026-06-14

Workspace:

- Isolated clone: `/Users/zhengisamazing/1.python_dir/brigit/brigit-data-science-automl-v1-fraud-control-skeleton`
- Branch: `feature/fraud-control-skeleton-build`
- Original checkout is not the active worktree for this effort.

## Current State

The control skeleton now has a repeatable discovery -> plug -> validation flow
over the local sample graph store:

1. Scenario discovery reads the canonical scenario register.
2. Graph methods are screened separately.
3. Scenario users are unioned and deduped by `user_id`.
4. Graph methods are added only when their marginal net-new population clears
   the configured support and DPD45-rate bar.
5. The final discovery union feeds plug derivation.
6. Plug validation reports `covered_discovery`, `uncovered_discovery`, and
   `outside_discovery` for State A and holdout.

The default control skeleton entry point remains:

```bash
uv run --group fraud python - <<'PY'
from pathlib import Path

from projects.fraud_anomaly_detection.codex_poc.control.config import ControlConfig
from projects.fraud_anomaly_detection.codex_poc.control.run import run_skeleton

report = run_skeleton(
    Path("projects/fraud_anomaly_detection/data/graph/fraud_graph.duckdb"),
    findings_db=Path("/tmp/fraud_control_findings.duckdb"),
    reports_db=Path("/tmp/fraud_control_reports.duckdb"),
    config=ControlConfig(),
)
print(report.keys())
PY
```

The scenario-by-scenario report that matches the latest review format is
repeatable with:

```bash
uv run --group fraud python -m \
  projects.fraud_anomaly_detection.codex_poc.control.selected_discovery_report \
  --store projects/fraud_anomaly_detection/data/graph/fraud_graph.duckdb \
  --out-dir projects/fraud_anomaly_detection/codex_poc/reports \
  --refresh-key selected_discovery_plug_report
```

Generated report files under
`projects/fraud_anomaly_detection/codex_poc/reports/` are intentionally ignored
by git. Regenerate them when inputs or thresholds change.

## Current Scenarios

Scenario register version: `2026-06-08.2`

The current canonical scenarios are:

- `ring_account_reuse`
- `ring_identity_burst`
- `ring_shared_persistent_account`
- `ring_device_burst`

The latest selected report over the sample produced this scenario readout:

| scenario | users found | DPD45 user rate | DPD45 advance rate |
| --- | ---: | ---: | ---: |
| ring_account_reuse | 554 | 96.6% | 93.7% |
| ring_identity_burst | 845 | 91.5% | 85.5% |
| ring_shared_persistent_account | 140 | 96.4% | 92.7% |
| ring_device_burst | 988 | 91.0% | 85.1% |
| scenario union, deduped | 1,024 | 90.4% | 84.8% |

## Current Graph Screen

The selected report screens graph methods after the scenario union. The current
selection rule is:

- marginal net-new users after dedupe >= 10
- marginal DPD45 user rate >= 50%

The latest sample run selected:

- `high_risk_entity_members_scenario_fraud_seed`

It excluded low-precision or duplicate graph methods, including:

- `residual_ring_members`
- `suspicion_queue_top200`
- `fraud_neighbours_hops2`
- `multi_witness_neighbors_scenario_fraud_seed`
- scenario-neighborhood variants whose marginal contribution did not clear the
  rule after dedupe.

This is intentional. Low-precision graph methods stay visible in the screened
table for auditability, but they do not feed the final discovery union or plug
derivation.

## Latest Selected Discovery Readout

Latest sample selected report:

- scenario union users: 1,024
- selected graph net-new users: 15
- final deduped discovery union users: 1,039
- final discovery DPD45 user rate: 90.4%
- final discovery DPD45 advance rate: 84.6%

Plug validation from the final discovery union:

- candidate keys from State A final discovery: 3,252
- qualified burned keys: 214

State A:

| bucket | users | DPD45 advance rate |
| --- | ---: | ---: |
| covered_discovery | 808 | 96.5% |
| uncovered_discovery | 108 | 32.0% |
| outside_discovery | 1 | 100.0% |

Holdout:

| bucket | users | DPD45 advance rate |
| --- | ---: | ---: |
| covered_discovery | 45 | 93.5% |
| uncovered_discovery | 79 | 79.4% |
| outside_discovery | 0 | 0.0% |

## Code Added Or Changed

Core skeleton:

- `control/outcomes.py` - advance-grain DPD45 outcome summaries.
- `control/discovery_report.py` - method, union, and scenario-vs-graph reports.
- `control/plug_report.py` - plug coverage and outside-discovery validation.
- `control/report_store.py` - DuckDB JSON run report persistence.
- `control/run.py` - holistic discovery, State A, holdout, and plug report.
- `control/selected_discovery_report.py` - repeatable report runner and CLI.

Audit fixes:

- `control/finding_store.py` now persists empty snapshots and hashes method
  versions correctly.
- `graph/discover.py` filters zero-score disconnected PPR queue rows.
- `analysis/subgroup_core.py` dedupes subgroup rules by actual test footprint,
  not aggregate stats.
- `pyproject.toml` includes `projects/*/codex_poc/tests` in default pytest
  collection.

Docs:

- `codex_poc/README.md` documents the control-loop workflow and repeatable
  selected report command.
- `codex_poc/docs/CONTROL_SYSTEM_DESIGN.md` records the implemented validation
  spine.

## Caveats

- The local sample is graph-thinned and fraud-enriched. Treat reported numbers
  as skeleton/process validation, not production calibration.
- Some archive-inspired graph ideas use hindsight-like seeds or entity-risk
  summaries. They are discovery/review evidence until rewritten as leak-free
  as-of features.
- The selected graph method is useful in this sample, but the graph catalog is
  still not a formal reviewed registry. Future graph methods should move toward
  a config-backed catalog like scenarios.
- Plug persistence is still a report-level burned-key table, not the full
  lifecycle-managed production registry from the design spec.
- Warehouse writes, GCS durability, and full v3 tuning remain future work.
- The `reports/` directory is generated output and ignored by git.

## Verification Commands

The last full verification before handoff should be:

```bash
uv run --group fraud pytest -q
uv run --group fraud ruff check \
  projects/fraud_anomaly_detection/codex_poc/control \
  projects/fraud_anomaly_detection/codex_poc/tests/control \
  projects/fraud_anomaly_detection/graph/discover.py \
  projects/fraud_anomaly_detection/analysis/subgroup_core.py
python - <<'PY'
import subprocess
import sys

result = subprocess.run(
    [
        "rg",
        "codex_poc\\\\.archived|from .*archived|import .*archived",
        "projects/fraud_anomaly_detection/codex_poc/control",
    ],
    capture_output=True,
    text=True,
)
if result.returncode == 0:
    print(result.stdout)
    sys.exit(1)
if result.returncode == 1:
    print("No archived imports found")
    sys.exit(0)
print(result.stderr)
sys.exit(result.returncode)
PY
```

## Next Steps

- Promote graph discovery methods into a reviewed graph-method catalog with
  explicit configs and evidence fields.
- Rewrite selected graph screens into leak-free as-of features before treating
  them as production-actionable.
- Add a sticky plug registry with lifecycle states: added, active, expired.
- Rebuild and rerun on v3/full warehouse data when VPN and source access are
  available.
- Decide where the production-facing burned-key list lives in Snowflake and
  what fields production needs for enforcement.
