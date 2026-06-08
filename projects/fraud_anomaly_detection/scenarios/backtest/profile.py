"""Targeted query profiling — stop guessing where the time goes.

Two ways in:

  run_and_profile(sql)      execute a SELECT, capture its query_id, return the
                            result frame + the per-operator time breakdown.
  profile_operators(qid)    profile an already-run query by id (e.g. one still
                            visible in QUERY_HISTORY).
  explain(sql)              the plan WITHOUT running it (free) — join types and
                            estimated partition pruning.

The per-operator breakdown (GET_QUERY_OPERATOR_STATS) is the useful one: it
attributes the wall time to each scan/join/aggregate and flags spilling to
local/remote storage (spilling = the warehouse ran out of memory, the usual
cause of a query that is slow out of proportion to the data).

    uv run python -m projects.fraud_anomaly_detection.scenarios.backtest.profile
"""

from __future__ import annotations

import pandas as pd

from automl.utils.io import snowflake


def _load_env() -> None:
    from dotenv import load_dotenv

    from automl.project.config import find_repo_root

    load_dotenv(find_repo_root() / ".env", override=False)


def profile_operators(query_id: str) -> pd.DataFrame:
    """Per-operator time %, row counts, and spill for a completed query."""
    sql = f"""
        SELECT
            operator_id,
            operator_type,
            ROUND(execution_time_breakdown:overall_percentage::float, 2) AS time_pct,
            operator_statistics:input_rows::number   AS input_rows,
            operator_statistics:output_rows::number  AS output_rows,
            operator_statistics:spilling:bytes_spilled_local::number  AS spill_local_bytes,
            operator_statistics:spilling:bytes_spilled_remote::number AS spill_remote_bytes
        FROM TABLE(GET_QUERY_OPERATOR_STATS('{query_id}'))
        ORDER BY time_pct DESC NULLS LAST
    """
    return snowflake.fetch_df(sql)


def run_and_profile(sql: str) -> tuple[pd.DataFrame, str, pd.DataFrame]:
    """Run a SELECT, return (result_df, query_id, operator_profile_df).

    Uses a single connection so we can read cursor.sfqid for the exact query —
    no guessing which row in QUERY_HISTORY was ours.
    """
    with snowflake.connect() as conn:
        cur = conn.cursor()
        try:
            cur.execute(sql)
            result = snowflake.coerce_decimal_columns(cur.fetch_pandas_all())
            query_id = cur.sfqid
        finally:
            cur.close()
    return result, query_id, profile_operators(query_id)


def explain(sql: str) -> pd.DataFrame:
    """Query plan without executing (free, instant)."""
    return snowflake.fetch_df(f"EXPLAIN USING TABULAR {sql}")


def main() -> int:
    import sys

    _load_env()
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", None)

    if len(sys.argv) > 1:  # profile an existing query id
        qid = sys.argv[1]
        print(f"=== operator profile for {qid} ===")
        print(profile_operators(qid).to_string(index=False))
        return 0

    # default: profile the backtest query end to end
    from projects.fraud_anomaly_detection.scenarios.backtest.monthly_backtest import build_sql

    result, qid, ops = run_and_profile(build_sql())
    print(f"query_id: {qid}")
    print(result.to_string(index=False))
    print("\n=== operator profile (time_pct desc) ===")
    print(ops.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
