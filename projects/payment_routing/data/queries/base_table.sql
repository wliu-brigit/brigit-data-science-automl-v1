-- Payment-routing base starter: the SELECT that defines the base data.
-- The harness wraps it in CREATE OR REPLACE TABLE and injects SPLIT_PCT from
-- split_group_key — do not emit SPLIT_PCT yourself.
-- Replace source table names, routing outcome fields, and date filters for
-- the specific routing project.
SELECT
    *
FROM {database}.{schema}.payment_routing_source_events
WHERE created_at >= DATEADD(month, -6, CURRENT_DATE)
