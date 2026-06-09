-- Diagnostics for the Tier-1 rebuild — RUN MANUALLY on the warehouse.
-- Not used by the harness. Two jobs:
--   (A) Verify the new source columns actually exist (before materializing).
--   (B) Find high-fan-out dummy values to extend the sentinel screens in
--       base_table.sql (identity_attribute_links) and to spot real-but-huge
--       shared values worth capping. This is the "group-by, count, find the
--       bad ones, then exclude" loop — performance tuning, do it AFTER the SQL
--       is confirmed correct.

-- ─────────────────────────────────────────────────────────────────────────
-- (A) COLUMN-EXISTENCE CHECKS  (run first — these gate the materialize)
-- ─────────────────────────────────────────────────────────────────────────

-- official_name / is_joint on the as-of plaid table (we read the historical
-- base_prod__plaid_accounts, NOT *_current_state which the team's queries use):
SELECT column_name, data_type
FROM brigit_snowflake.information_schema.columns
WHERE table_schema = 'DBT_ANALYTICS'
  AND table_name = 'BASE_PROD__PLAID_ACCOUNTS'
  AND column_name IN ('OFFICIAL_NAME', 'IS_JOINT')
ORDER BY column_name;

-- is_neobank_high_risk_institution on fct_loans:
SELECT column_name, data_type
FROM brigit_snowflake.information_schema.columns
WHERE table_schema = 'DBT_ANALYTICS'
  AND table_name = 'FCT_LOANS'
  AND column_name = 'IS_NEOBANK_HIGH_RISK_INSTITUTION';

-- ─────────────────────────────────────────────────────────────────────────
-- (B) FAN-OUT SCREENS  — top shared values per attribute, after current screen.
--     If a value at the top is a dummy/placeholder, add it to the matching
--     CASE in identity_attribute_links. If it's a real value shared by an
--     absurd number of users, consider a giant-group cap (parked refinement).
-- ─────────────────────────────────────────────────────────────────────────

-- Top emails by distinct users (mirror the base_table screen so we see what
-- survives it):
WITH norm AS (
    SELECT
        i.user_id,
        CASE
            WHEN i.email IS NULL OR TRIM(i.email) = '' THEN NULL
            WHEN LOWER(TRIM(i.email)) LIKE 'test@%'
              OR LOWER(TRIM(i.email)) LIKE 'noreply@%'
              OR LOWER(TRIM(i.email)) LIKE 'no-reply@%'
              OR LOWER(TRIM(i.email)) LIKE '%@example.com'
              OR LOWER(TRIM(i.email)) LIKE '%@test.com'
            THEN NULL
            ELSE LOWER(TRIM(i.email))
        END AS email_norm
    FROM brigit_snowflake.dbt_analytics.base_prod__identities i
    WHERE i.is_deleted = FALSE
)
SELECT email_norm, COUNT(DISTINCT user_id) AS n_users
FROM norm
WHERE email_norm IS NOT NULL
GROUP BY email_norm
ORDER BY n_users DESC
LIMIT 100;

-- Top phones by distinct users (digits-only, last 10, after screen):
WITH norm AS (
    SELECT
        i.user_id,
        CASE
            WHEN i.phone_number IS NULL THEN NULL
            WHEN LENGTH(REGEXP_REPLACE(i.phone_number, '[^0-9]', '')) < 10 THEN NULL
            WHEN RIGHT(REGEXP_REPLACE(i.phone_number, '[^0-9]', ''), 10) IN (
                '0000000000','1111111111','2222222222','3333333333','4444444444',
                '5555555555','6666666666','7777777777','8888888888','9999999999',
                '1234567890','0123456789') THEN NULL
            ELSE RIGHT(REGEXP_REPLACE(i.phone_number, '[^0-9]', ''), 10)
        END AS phone_norm
    FROM brigit_snowflake.dbt_analytics.base_prod__identities i
    WHERE i.is_deleted = FALSE
)
SELECT phone_norm, COUNT(DISTINCT user_id) AS n_users
FROM norm
WHERE phone_norm IS NOT NULL
GROUP BY phone_norm
ORDER BY n_users DESC
LIMIT 100;

-- Top addresses by distinct users:
WITH norm AS (
    SELECT
        i.user_id,
        CASE
            WHEN i.matched_street_address IS NULL OR TRIM(i.matched_street_address) = ''
              OR i.matched_zip_code IS NULL OR TRIM(i.matched_zip_code) = ''
            THEN NULL
            ELSE UPPER(TRIM(i.matched_street_address)) || '|' || TRIM(i.matched_zip_code)
        END AS address_key
    FROM brigit_snowflake.dbt_analytics.base_prod__identities i
    WHERE i.is_deleted = FALSE
)
SELECT address_key, COUNT(DISTINCT user_id) AS n_users
FROM norm
WHERE address_key IS NOT NULL
GROUP BY address_key
ORDER BY n_users DESC
LIMIT 100;

-- Top device_ids by distinct users (no screen needed — device is ~unique;
-- this just confirms the fan-out is bounded):
SELECT cm.device_id, COUNT(DISTINCT cm.identifying_id) AS n_users
FROM pc_fivetran_db.wal_brigit_production_public.user_client_metadata cm
WHERE cm._fivetran_deleted = FALSE
  AND cm.device_id IS NOT NULL
GROUP BY cm.device_id
ORDER BY n_users DESC
LIMIT 100;
