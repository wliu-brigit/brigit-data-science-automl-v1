-- Snowflake training-data starter.
SELECT
    *,
    MOD(ABS(HASH(<TBD_HASH_KEY_COLUMN>)), 100) AS SPLIT_PCT
FROM {database}.{schema}.{base_table};
