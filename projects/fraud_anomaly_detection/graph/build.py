"""Build the persisted entity-graph store: one lossless, self-contained DuckDB file.

The build is a pure SQL transformation of the base table (kept
warehouse-portable: plain SELECT/UNION ALL — the same shape can run in
Snowflake). No judgment calls here: edges are UNCAPPED, every entity type is
stored (ip/email included; analysis views exclude them by default), and the
full base table is snapshotted in (`advances`) so the file alone carries all
metadata and labels. The only build-time cleaning is sentinel screening —
non-values like '' / 'none' / '0-0' are not identities — with screened counts
logged to `meta`.

Refresh model: full rebuild only (idempotent — the file is replaced).
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

# entity_type -> base-table column. THE canonical map; load.py reuses it.
ENTITY_COLS: dict[str, str] = {
    "device": "device_id",
    "bank": "bank_account_key",
    "persistent": "persistent_account_id",
    "phone": "phone_key",
    "address": "address_key",
    "email": "email_key",
    "ip": "ip_address",
}

# Non-values seen in the wild (prior graph effort + v2 due diligence).
SENTINELS: tuple[str, ...] = ("", "none", "nan", "null", "nat", "0", "0-0", "none-none")

ADVANCE_ID, USER_ID, TS = "advance_id", "user_id", "feature_as_of_ts"


def _sentinel_list() -> str:
    return ", ".join(f"'{s}'" for s in SENTINELS)


def _normalized(col: str) -> str:
    """The screen and its complement count MUST use the same expression."""
    return f"lower(trim(CAST({col} AS VARCHAR)))"


def _edge_select(etype: str, col: str) -> str:
    value = _normalized(col)
    return f"""
        SELECT DISTINCT {ADVANCE_ID} AS advance_id, {USER_ID} AS user_id,
               '{etype}' AS entity_type,
               CAST({col} AS VARCHAR) AS entity_value, {TS} AS ts
        FROM advances
        WHERE {col} IS NOT NULL AND {value} NOT IN ({_sentinel_list()})
    """


def build_store(
    source: Path | str | pd.DataFrame,
    out_path: Path | str,
    source_label: str = "",
) -> dict[str, int]:
    """Derive the store from the base table; returns the count summary.

    `source` is a DataFrame or a parquet path. The output file is replaced
    (full-rebuild refresh model). Summary counts are also persisted to `meta`.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.unlink(missing_ok=True)
    Path(f"{out}.wal").unlink(missing_ok=True)

    con = duckdb.connect(str(out))
    try:
        if isinstance(source, pd.DataFrame):
            con.register("src", source)
            con.execute("CREATE TABLE advances AS SELECT * FROM src")
        else:
            con.execute(
                "CREATE TABLE advances AS SELECT * FROM read_parquet(?)", [str(source)]
            )

        edge_union = "\nUNION ALL\n".join(
            _edge_select(etype, col) for etype, col in ENTITY_COLS.items()
        )
        con.execute(f"CREATE TABLE edges AS {edge_union}")

        con.execute(f"""
            CREATE TABLE users AS
            SELECT {USER_ID} AS user_id, count(DISTINCT {ADVANCE_ID}) AS n_advances,
                   min({TS}) AS first_seen_ts, max({TS}) AS last_seen_ts
            FROM advances GROUP BY 1
        """)
        con.execute("""
            CREATE TABLE entities AS
            SELECT entity_type, entity_value,
                   count(DISTINCT user_id) AS n_users,
                   count(DISTINCT advance_id) AS n_advances,
                   min(ts) AS first_seen_ts, max(ts) AS last_seen_ts
            FROM edges GROUP BY 1, 2
        """)

        summary: dict[str, int] = {}
        summary["n_advances"] = con.execute("SELECT count(*) FROM advances").fetchone()[0]
        summary["n_users"] = con.execute("SELECT count(*) FROM users").fetchone()[0]
        summary["n_edges"] = con.execute("SELECT count(*) FROM edges").fetchone()[0]
        for etype, col in ENTITY_COLS.items():
            summary[f"edges_{etype}"] = con.execute(
                "SELECT count(*) FROM edges WHERE entity_type = ?", [etype]
            ).fetchone()[0]
            summary[f"screened_{etype}"] = con.execute(
                f"SELECT count(*) FROM advances WHERE {col} IS NOT NULL "
                f"AND {_normalized(col)} IN ({_sentinel_list()})"
            ).fetchone()[0]

        con.execute("CREATE TABLE meta (key VARCHAR, value VARCHAR)")
        con.execute("INSERT INTO meta VALUES ('built_at', CAST(current_timestamp AS VARCHAR))")
        if not source_label:
            source_label = (
                f"<in-memory DataFrame, {len(source)} rows>"
                if isinstance(source, pd.DataFrame) else str(source)
            )
        con.execute("INSERT INTO meta VALUES ('source', ?)", [source_label])
        for key, val in summary.items():
            con.execute("INSERT INTO meta VALUES (?, ?)", [key, str(val)])
        return summary
    finally:
        con.close()
