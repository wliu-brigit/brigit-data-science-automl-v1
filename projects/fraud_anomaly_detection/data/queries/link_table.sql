-- Link-grain extraction for the graph store (TODO "LINK-GRAIN EDGES", 2026-06-10).
--
-- One row per user<->entity LINK, independent of advances -- this is the fix for
-- the advance-grain blind spot: users who registered and linked a shared
-- device/account but never drew an advance are exactly the ring evidence the
-- warehouse users_on_*_72h columns count and the advance-built graph cannot see.
--
-- Output columns: user_id, entity_type, entity_value, ts (when the link formed,
-- the as-of bound), identity_created_time (per user, for freshness windows).
-- entity_value normalization MUST stay byte-identical to base_table.sql so link
-- nodes collide with advance nodes: raw device_id, routing-account key, raw
-- persistent_account_id, SHA-256 of the normalized phone/email/address. No raw
-- PII is emitted. Sentinel screens are copied from base_table.sql verbatim --
-- a screened value is NULL here and never forms an edge (build.py screens
-- again, which is harmless).
--
-- NOT a harness dataset: this feeds graph_store_build --links-parquet. Export
-- however is convenient on the prod side, for example
--   COPY INTO @<stage>/fraud_link_table.parquet FROM (<this SELECT>)
--   FILE_FORMAT = (TYPE = PARQUET) HEADER = TRUE
-- then download and pass the file path. Volume estimate: device links collapse
-- to ~16M rows (the SCD dedup below), plaid accounts a few M, identity keys one
-- row per user per populated attribute -- tens of millions of rows total.
--
-- STATUS: written 2026-06-10, NOT yet validated on the warehouse (needs the
-- usual EXPLAIN + fan-out value scan before first use -- same checklist as the
-- v3 build; see TODO).
--
-- No lower time floor on links: a long-standing device or account link is real
-- as-of any later advance (the warehouse sharing counts are lifetime-bounded
-- the same way). History depth is governed by the source tables themselves.

WITH identities_one_per_user AS (
    SELECT
        i.user_id::VARCHAR AS user_id,
        i.created_time::TIMESTAMP_NTZ AS identity_created_time,
        i.email,
        i.phone_number,
        i.matched_street_address,
        i.matched_zip_code
    FROM brigit_snowflake.dbt_analytics.base_prod__identities i
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY i.user_id
        ORDER BY i.is_deleted ASC, i.created_time DESC
    ) = 1
),

-- one row per (user, device), earliest association = the as-of bound
-- (user_client_metadata is SCD, ~11 rows per user-device -- collapse first)
device_links AS (
    SELECT
        cm.identifying_id::VARCHAR AS user_id,
        cm.device_id,
        MIN(cm.valid_from::TIMESTAMP_NTZ) AS link_ts
    FROM pc_fivetran_db.wal_brigit_production_public.user_client_metadata cm
    WHERE cm._fivetran_deleted = FALSE
      AND cm.device_id IS NOT NULL
    GROUP BY 1, 2
),

-- one row per (user, routing, account); earliest created_at as the as-of bound
-- (slightly more conservative than base_table, which keys off the latest row)
bank_links AS (
    SELECT
        pa.user_id::VARCHAR AS user_id,
        CONCAT(pa.routing_number, '-', pa.account_number) AS bank_account_key,
        MAX(pa.persistent_account_id) AS persistent_account_id,
        MIN(pa.created_at::TIMESTAMP_NTZ) AS link_ts
    FROM brigit_snowflake.dbt_analytics.base_prod__plaid_accounts pa
    WHERE pa.routing_number IS NOT NULL
      AND pa.account_number IS NOT NULL
    GROUP BY 1, 2
),

-- normalized identity attributes, sentinel-screened -- byte-identical to
-- base_table.sql identity_attribute_links so the hashes collide
identity_attribute_links AS (
    SELECT
        i.user_id,
        i.identity_created_time,

        CASE
            WHEN i.email IS NULL OR TRIM(i.email) = '' THEN NULL
            WHEN LOWER(TRIM(i.email)) LIKE 'test@%'
              OR LOWER(TRIM(i.email)) LIKE 'noreply@%'
              OR LOWER(TRIM(i.email)) LIKE 'no-reply@%'
              OR LOWER(TRIM(i.email)) LIKE '%@example.com'
              OR LOWER(TRIM(i.email)) LIKE '%@test.com'
            THEN NULL
            ELSE LOWER(TRIM(i.email))
        END AS email_norm,

        CASE
            WHEN i.phone_number IS NULL THEN NULL
            WHEN LENGTH(REGEXP_REPLACE(i.phone_number, '[^0-9]', '')) < 10 THEN NULL
            WHEN RIGHT(REGEXP_REPLACE(i.phone_number, '[^0-9]', ''), 10) IN (
                '0000000000', '1111111111', '2222222222', '3333333333',
                '4444444444', '5555555555', '6666666666', '7777777777',
                '8888888888', '9999999999', '1234567890', '0123456789'
            ) THEN NULL
            ELSE RIGHT(REGEXP_REPLACE(i.phone_number, '[^0-9]', ''), 10)
        END AS phone_norm,

        CASE
            WHEN i.matched_street_address IS NULL OR TRIM(i.matched_street_address) = ''
              OR i.matched_zip_code IS NULL OR TRIM(i.matched_zip_code) = ''
            THEN NULL
            WHEN TRIM(i.matched_zip_code) IN ('00000', '00003') THEN NULL
            WHEN UPPER(TRIM(i.matched_street_address)) IN ('MAIL RETURN', 'PLEASE UPDATE') THEN NULL
            ELSE UPPER(TRIM(i.matched_street_address)) || '|' || TRIM(i.matched_zip_code)
        END AS address_key

    FROM identities_one_per_user i
)

SELECT d.user_id,
       'device' AS entity_type,
       CAST(d.device_id AS VARCHAR) AS entity_value,
       d.link_ts AS ts,
       i.identity_created_time
FROM device_links d
JOIN identities_one_per_user i ON d.user_id = i.user_id

UNION ALL

SELECT b.user_id, 'bank', b.bank_account_key, b.link_ts, i.identity_created_time
FROM bank_links b
JOIN identities_one_per_user i ON b.user_id = i.user_id

UNION ALL

SELECT b.user_id, 'persistent', CAST(b.persistent_account_id AS VARCHAR),
       b.link_ts, i.identity_created_time
FROM bank_links b
JOIN identities_one_per_user i ON b.user_id = i.user_id
WHERE b.persistent_account_id IS NOT NULL

UNION ALL

-- identity attributes form at identity creation -- that is their link ts
SELECT user_id, 'phone', SHA2(phone_norm, 256), identity_created_time, identity_created_time
FROM identity_attribute_links WHERE phone_norm IS NOT NULL

UNION ALL

SELECT user_id, 'email', SHA2(email_norm, 256), identity_created_time, identity_created_time
FROM identity_attribute_links WHERE email_norm IS NOT NULL

UNION ALL

SELECT user_id, 'address', SHA2(address_key, 256), identity_created_time, identity_created_time
FROM identity_attribute_links WHERE address_key IS NOT NULL
