-- Payment-routing training-data starter: pulls training rows from the base table.
-- SPLIT_PCT flows through from the base table; keep it in the projection.
-- Requirements:
--   1. Emit the configured target column.
--   2. Exclude fields only known after the routing decision from features.
SELECT
    *
FROM {database}.{schema}.{base_table}
