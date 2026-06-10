-- The SELECT that pulls training rows from the materialized snapshot.
-- SPLIT_PCT flows through; keep it in the projection. No filtering here:
-- the recipe's named splits (train / test / oot) carve the rows.
SELECT *
FROM {database}.{schema}.{base_table}
