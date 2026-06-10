"""Run the heuristic-band-by-month comparison in-session (no console paste).

Executes heuristic_band_by_month.sql against the pre-built snapshot table via
the same warehouse seam the backtest uses, prints the result, and saves it to
results/heuristic/ so it sits apart from the scenario backtest CSVs. Needs VPN.

    uv run python -m projects.fraud_anomaly_detection.scenarios.backtest.heuristic_band
"""

from __future__ import annotations

from pathlib import Path

SQL_FILE = Path(__file__).parent / "heuristic_band_by_month.sql"


def build_sql() -> str:
    """The first statement of the .sql file.

    Strips ``--`` line comments first (a comment sentence may contain a ``;``,
    which would otherwise truncate the statement), then takes up to the first
    real semicolon.
    """
    import re

    no_comments = re.sub(r"--.*", "", SQL_FILE.read_text())
    return no_comments.split(";", 1)[0].strip() + ";"


def run() -> "tuple[object, str]":
    from automl.utils.io import snowflake

    with snowflake.connect() as conn:
        cur = conn.cursor()
        try:
            cur.execute(build_sql())
            return snowflake.coerce_decimal_columns(cur.fetch_pandas_all()), cur.sfqid
        finally:
            cur.close()


def main() -> int:
    import pandas as pd
    from dotenv import load_dotenv

    from automl.project.config import find_repo_root

    load_dotenv(find_repo_root() / ".env", override=False)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", None)

    df, qid = run()
    print(f"query_id: {qid}")
    print(df.to_string(index=False))

    out_dir = Path(__file__).parent / "results" / "heuristic"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "heuristic_band_by_month.csv"
    df.to_csv(out_path, index=False)
    print(f"\nsaved -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
