-- The SELECT that defines the base data. The upstream feature snapshot
-- (fraud_advance_feature_base) is built and owned outside the harness — its
-- DDL is kept for reference at data/queries/upstream_fraud_advance_feature_base.sql.
-- The harness wraps this SELECT in CREATE OR REPLACE TABLE {base_table} and
-- injects SPLIT_PCT from split_group_key — do not emit SPLIT_PCT yourself.
SELECT *
FROM {database}.{schema}.fraud_advance_feature_base
