-- Payment-routing base-data starter.
-- Build or reference the base table used by training_data.sql.
-- Replace source table names, routing outcome fields, and date filters for
-- the specific routing project.

CREATE OR REPLACE TABLE {database}.{schema}.{base_table} AS
SELECT
    *
FROM {database}.{schema}.payment_routing_source_events
WHERE created_at >= DATEADD(month, -6, CURRENT_DATE);
