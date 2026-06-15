# Fraud Control Registry And Selection Implementation Record

This plan has been superseded by the forward-only `neo4j_codex` control-loop
implementation.

## Current State

- The active package is
  `projects/fraud_anomaly_detection/neo4j_codex`.
- The supported operator entry point is
  `projects.fraud_anomaly_detection.neo4j_codex.control.control_loop_report`.
- Scenario discovery comes from the canonical `scenarios/register.yaml`.
- Graph discovery screens are registered in
  `control/discovery/graph_screen_catalog.py`.
- Graph rows carry explicit statuses:
  `promoted_to_plug_derivation`, `review_only`,
  `below_min_marginal_users`, and `below_min_marginal_dpd45_user_rate`.
- `--include-status` filters displayed graph rows without changing underlying
  evaluation or plug derivation.

## Deliberate Deletions

- The old representative runner is removed.
- The one-off representative graph adapter is removed.
- The old small default-method catalog is removed.

Those pieces were useful while proving the walking skeleton, but they are now
misleading because the full control-loop report is the system surface.

## V3 Follow-Up

- Promote graph screen registration into the durable method-registry shape once
  v3 data confirms which methods deserve lifecycle ownership.
- Split graph query implementations out of `control_loop_report.py` when the
  method set grows beyond the current sample-report scope.
- Add sticky plug lifecycle state and warehouse-facing plug export.
- Turn the State A / holdout report path into a reusable monthly backtest
  runner.
