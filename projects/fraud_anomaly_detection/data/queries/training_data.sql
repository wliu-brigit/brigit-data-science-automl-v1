-- The SELECT that pulls training rows from the base table.
-- SPLIT_PCT flows through; keep it in the projection.
SELECT *
FROM {database}.{schema}.{base_table}
