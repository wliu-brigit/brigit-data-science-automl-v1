"""Month-over-month scenario backtest (standalone ops tooling).

Runs the registered scenario predicates against the full warehouse history,
bucketed by advance month, to watch the caught-volume and dpd45 quality trend
over time. Lives outside the AutoML harness — same status as the upstream
feature DDL. See docs/execution_parallel/month-over-month-backtest/README.md.
"""
