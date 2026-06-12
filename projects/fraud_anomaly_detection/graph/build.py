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
               CAST({col} AS VARCHAR) AS entity_value, {TS} AS ts,
               'advance' AS source
        FROM advances
        WHERE {col} IS NOT NULL AND {value} NOT IN ({_sentinel_list()})
    """


ICT = "identity_created_time"  # per-user; carried into `users` when available


def build_store(
    source: Path | str | pd.DataFrame,
    out_path: Path | str,
    source_label: str = "",
    links: Path | str | pd.DataFrame | None = None,
) -> dict[str, int]:
    """Derive the store from the base table; returns the count summary.

    `source` is a DataFrame or a parquet path. The output file is replaced
    (full-rebuild refresh model). Summary counts are also persisted to `meta`.

    `links` is the OPTIONAL link-grain source (DataFrame or parquet): one row
    per user<->entity link with columns user_id, entity_type, entity_value,
    ts (when the link formed), and optionally identity_created_time. It may
    include users with no advances — that is its purpose (the advance-grain
    blind spot). Link rows get the same sentinel screen and land in `edges`
    with source='link' (advance edges carry source='advance').
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

        if links is not None:
            if isinstance(links, pd.DataFrame):
                con.register("link_src", links)
            else:
                con.execute(
                    "CREATE TEMP VIEW link_src AS SELECT * FROM read_parquet(?)",
                    [str(links)],
                )
            unknown = [
                t for (t,) in con.execute(
                    "SELECT DISTINCT entity_type FROM link_src").fetchall()
                if t not in ENTITY_COLS
            ]
            if unknown:
                raise ValueError(
                    f"unknown link entity_type(s) {sorted(unknown)};"
                    f" expected {sorted(ENTITY_COLS)}"
                )
            link_cols = {r[0] for r in con.execute("DESCRIBE link_src").fetchall()}
            link_ict = (
                f"min({ICT})" if ICT in link_cols else "CAST(NULL AS TIMESTAMP)"
            )
            con.execute(f"""
                INSERT INTO edges (advance_id, user_id, entity_type, entity_value, ts, source)
                SELECT DISTINCT NULL, CAST(user_id AS VARCHAR), entity_type,
                       CAST(entity_value AS VARCHAR), ts, 'link'
                FROM link_src
                WHERE entity_value IS NOT NULL
                  AND {_normalized("entity_value")} NOT IN ({_sentinel_list()})
            """)

        base_cols = {r[0] for r in con.execute("DESCRIBE advances").fetchall()}
        base_ict = f"min({ICT})" if ICT in base_cols else "CAST(NULL AS TIMESTAMP)"
        con.execute(f"""
            CREATE TABLE users AS
            SELECT {USER_ID} AS user_id, count(DISTINCT {ADVANCE_ID}) AS n_advances,
                   min({TS}) AS first_seen_ts, max({TS}) AS last_seen_ts,
                   {base_ict} AS identity_created_time
            FROM advances GROUP BY 1
        """)
        if links is not None:
            # link-only users: real nodes with zero advances (the ring members
            # the advance-grain view cannot see)
            con.execute(f"""
                INSERT INTO users
                SELECT e.user_id, 0, min(e.ts), max(e.ts), {link_ict.replace(ICT, f"l.{ICT}")}
                FROM edges e
                LEFT JOIN link_src l ON CAST(l.user_id AS VARCHAR) = e.user_id
                WHERE e.source = 'link'
                  AND e.user_id NOT IN (SELECT user_id FROM users)
                GROUP BY 1
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
        summary["n_link_edges"] = con.execute(
            "SELECT count(*) FROM edges WHERE source = 'link'"
        ).fetchone()[0]
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
