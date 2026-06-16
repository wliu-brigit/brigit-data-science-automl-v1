# Fraud Control Skeleton Handoff

Updated: 2026-06-16

Workspace:

- Isolated clone: `/Users/zhengisamazing/1.python_dir/brigit/brigit-data-science-automl-v1-fraud-control-skeleton`
- Branch: `neo4j_codex`
- Original checkout is not the active worktree for this effort.

## Current State

The control skeleton now has one repeatable discovery -> plug -> validation
operator report over the local sample graph store plus a local Neo4j mirror:

1. Scenario discovery reads the canonical scenario register.
2. Graph methods run in Neo4j via Cypher/GDS and are screened separately.
3. Scenario users are unioned and deduped by `user_id`.
4. Graph methods are labeled with a status. Review-only graph pockets remain
   visible, while only promotion-safe graph methods can enter plug derivation.
5. The final discovery union feeds plug derivation.
6. Plug validation reports `covered_discovery`, `uncovered_discovery`, and
   `outside_discovery` for State A and holdout.

The supported entry point is:

```bash
NEO4J_PASSWORD=fraudpocpass uv run --with neo4j --group fraud python -m \
  projects.fraud_anomaly_detection.neo4j_codex.control.control_loop_report \
  --store projects/fraud_anomaly_detection/data/graph/fraud_graph.duckdb \
  --out-dir projects/fraud_anomaly_detection/neo4j_codex/reports \
  --refresh-key fraud_control_loop_report \
  --neo4j-uri bolt://localhost:7687 \
  --neo4j-user neo4j \
  --neo4j-database neo4j
```

Start the current local mirror first:

```bash
bash projects/fraud_anomaly_detection/neo4j_codex/archived/scripts/setup_neo4j.sh
```

Use `--include-status review_only` to show only graph review rows, or repeat
`--include-status` for multiple statuses. The filter changes displayed graph
rows only; all statuses are still counted in the JSON and Markdown.

Generated report files under
`projects/fraud_anomaly_detection/neo4j_codex/reports/` are intentionally ignored
by git. Regenerate them when inputs or thresholds change.

## Current Scenarios

Scenario register version: `2026-06-08.2`

The current canonical scenarios are:

- `ring_account_reuse`
- `ring_identity_burst`
- `ring_shared_persistent_account`
- `ring_device_burst`

The latest control-loop report over the sample produced this scenario readout:

| scenario | users found | DPD45 user rate | DPD45 advance rate |
| --- | ---: | ---: | ---: |
| ring_account_reuse | 554 | 96.6% | 93.7% |
| ring_identity_burst | 845 | 91.5% | 85.5% |
| ring_shared_persistent_account | 140 | 96.4% | 92.7% |
| ring_device_burst | 988 | 91.0% | 85.1% |
| scenario union, deduped | 1,024 | 90.4% | 84.8% |

## Current Graph Screen

The control-loop report screens Neo4j graph methods after the scenario union.
The current selection rule is:

- marginal net-new users after dedupe >= 10
- marginal DPD45 user rate >= 50%

The latest sample run promoted no graph methods into plug derivation because
the current graph screens are `snapshot_review` / `review_queue`. The active
implementation no longer uses the old Python/igraph graph backend; graph
discovery runs through `neo4j_codex/control/graph/`. It still surfaces
review-only graph pockets, including:

- `residual_ring_members`
- `suspicion_queue_top200`
- `fraud_neighbours_hops2`
- `multi_witness_neighbors_scenario_fraud_seed`
- scenario-neighborhood variants, including
  `graph:scenario_neighborhood:ring_account_reuse` with 13 net-new users at
  84.6% DPD45 in the sample.

This is intentional. Review-only graph methods stay visible in the screened
table for auditability, but they do not feed the final discovery union or plug
derivation until their method metadata becomes promotion-safe.

## Latest Selected Discovery Readout

Latest live Neo4j-backed sample control-loop report:

- scenario union users: 1,024
- review-only graph net-new users: 235 at 14.0% DPD45 user rate
- selected graph net-new users: 0
- final deduped discovery union users: 1,024
- final discovery DPD45 user rate: 90.4%
- final discovery DPD45 advance rate: 84.8%

Plug validation from the final discovery union:

- candidate keys from State A final discovery: 3,211
- qualified burned keys: 210

State A:

| bucket | users | DPD45 advance rate |
| --- | ---: | ---: |
| covered_discovery | 796 | 96.6% |
| uncovered_discovery | 107 | 33.2% |
| outside_discovery | 9 | 100.0% |

Holdout:

| bucket | users | DPD45 advance rate |
| --- | ---: | ---: |
| covered_discovery | 44 | 93.3% |
| uncovered_discovery | 78 | 79.2% |
| outside_discovery | 1 | 100.0% |

## Code Added Or Changed

Core skeleton:

- `control/outcomes.py` - advance-grain DPD45 outcome summaries.
- `control/discovery_report.py` - method, union, and scenario-vs-graph reports.
- `control/graph/` - Neo4j client boundary and Cypher/GDS graph discovery
  methods.
- `control/plug_report.py` - plug coverage and outside-discovery validation.
- `control/report_store.py` - DuckDB JSON run report persistence.
- `control/control_loop_report.py` - repeatable scenario, graph, plug,
  State A, and holdout report runner and CLI.

Audit fixes:

- `control/finding_store.py` now persists empty snapshots and hashes method
  versions correctly.
- `graph/discover.py` filters zero-score disconnected PPR queue rows.
- `analysis/subgroup_core.py` dedupes subgroup rules by actual test footprint,
  not aggregate stats.
- `pyproject.toml` includes `projects/*/neo4j_codex/tests` in default pytest
  collection.

Docs:

- `neo4j_codex/README.md` documents the control-loop workflow and repeatable
  report command.
- `neo4j_codex/docs/CONTROL_SYSTEM_DESIGN.md` records the implemented validation
  spine.

## Caveats

- The local sample is graph-thinned and fraud-enriched. Treat reported numbers
  as skeleton/process validation, not production calibration.
- Current graph methods are Neo4j-backed but still snapshot-review evidence.
  They are discovery/review leads until each promoted method has leak-free
  as-of or production-safe Neo4j semantics.
- The active report needs a running Neo4j mirror with GDS. Unit tests use a fake
  Neo4j runner to verify orchestration without requiring a local database.
- Manual/human-assigned plug lifecycle is not implemented yet. Automated plugs
  are derived and evaluated; human-assigned plugs should be added as a separate
  registry path and evaluated separately.
- Plug persistence is still a report-level burned-key table, not the full
  lifecycle-managed production registry from the design spec.
- Warehouse writes, GCS durability, and full v3 tuning remain future work.
- The `reports/` directory is generated output and ignored by git.

## Verification Commands

The last full verification before handoff should be:

```bash
uv run --group fraud pytest -q
uv run --group fraud ruff check \
  projects/fraud_anomaly_detection/neo4j_codex/control \
  projects/fraud_anomaly_detection/neo4j_codex/tests/control \
  projects/fraud_anomaly_detection/graph/discover.py \
  projects/fraud_anomaly_detection/analysis/subgroup_core.py
python - <<'PY'
import subprocess
import sys

result = subprocess.run(
    [
        "rg",
        "neo4j_codex\\\\.archived|from .*archived|import .*archived",
        "projects/fraud_anomaly_detection/neo4j_codex/control",
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
