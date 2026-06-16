# To-do (tiny): a Snowflake IO helper for metadata commands

**Status:** noted, not started. Small, self-contained — no design call needed.

## What

Add a helper to `automl/utils/io/snowflake.py` that can run metadata /
non-SELECT-result commands — `DESCRIBE TABLE`, `SHOW COLUMNS`, `SHOW TABLES`,
etc. — and return rows.

## Why

`fetch_df()` runs `cursor.fetch_pandas_all()`, an Arrow-backed fetch. Arrow
result sets only exist for real query results; metadata commands like
`DESCRIBE TABLE` raise `snowflake.connector.errors.NotSupportedError: Unknown
error` from `fetch_pandas_all`. `fetch_one()` works (plain `fetchone`) but only
returns the first row.

Hit on neobank_ncm VPN day (2026-06-09): the runbook's `DESCRIBE TABLE` checks
failed and had to be rewritten against `INFORMATION_SCHEMA.COLUMNS` (which *is*
a real SELECT, so `fetch_df` works). That workaround is fine but the helper
should exist so the next person doesn't rediscover it.

## Shape (suggested)

A `fetch_rows(sql) -> list[tuple]` (or `-> pd.DataFrame` built from
`cursor.description` for column names) that uses `cursor.fetchall()` instead of
the Arrow path — the same connect/cursor/close pattern as `fetch_one`, just
returning all rows. Then DESCRIBE/SHOW work directly.

```python
def fetch_rows(sql: str) -> list[tuple]:
    with connect() as connection:
        cursor = connection.cursor()
        try:
            cursor.execute(sql)
            return cursor.fetchall()
        finally:
            cursor.close()
```

## Don't forget

- It's IO under `automl/utils` — a leaf; nothing imports up, no contract churn.
- Add a unit test only if it's cheap to do offline; otherwise an e2e-gated one.
- Delete this file once landed.
