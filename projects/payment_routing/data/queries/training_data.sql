-- Payment-routing training-data starter.
-- Requirements:
--   1. Emit the configured target column.
--   2. Emit SPLITID as a deterministic integer 0-99.
--   3. Exclude fields only known after the routing decision from features.

SELECT
    *,
    MOD(ABS(HASH(payment_id)), 100) AS SPLITID
FROM {database}.{schema}.{base_table};
