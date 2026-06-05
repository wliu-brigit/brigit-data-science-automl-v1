-- The SELECT that defines your base data: joins, CTEs, filters, feature SQL.
-- The harness wraps it in CREATE OR REPLACE TABLE and injects SPLIT_PCT from
-- split_group_key — do not emit SPLIT_PCT yourself.
SELECT *
FROM {database}.{schema}.<TBD_SOURCE_TABLE>
