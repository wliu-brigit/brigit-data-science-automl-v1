-- The SELECT that defines the base data.
--
-- INLINED (2026-06-08): this file now contains the full feature-base logic that
-- used to live in the out-of-band `upstream_fraud_advance_feature_base.sql`
-- (archived under data/queries/archive/). The harness wraps this SELECT in
-- CREATE OR REPLACE TABLE {base_table} and injects SPLIT_PCT from
-- split_group_key -- do NOT emit SPLIT_PCT here, and do NOT add a CREATE wrapper.
--
-- It reads the raw dbt/production tables directly, so the harness Snowflake role
-- needs read grants on fct_loans, base_prod__*, user_client_metadata, etc.
--
-- Everything from the original upstream is preserved byte-for-byte (in
-- particular heuristic_fraud_score / heuristic_fraud_band are UNCHANGED, to keep
-- the is_fraud proxy label and trial comparability stable). New Tier-1 features
-- are added in clearly-marked `-- NEW (Tier 1, 2026-06-08)` sections. See the
-- project TODO.md "CONSOLIDATED FEATURE-ADD PLAN" for rationale.

WITH params AS (
    SELECT
        '2025-11-01'::TIMESTAMP_NTZ AS history_start_ts,
        '2025-12-01'::TIMESTAMP_NTZ AS output_start_ts
),

/* ---------------------------------------------------------------------
   1. All advances used for historical context.
--------------------------------------------------------------------- */
all_advances AS (
    SELECT
        l.id::VARCHAR AS advance_id,
        l.user_id::VARCHAR AS user_id,

        l.plaid_routing_number,
        l.plaid_account_number,

        l.loan_amount,
        COALESCE(l.express_transfer_fee, 0) AS express_transfer_fee,
        l.loan_amount + COALESCE(l.express_transfer_fee, 0) AS total_disbursed,

        l.origination_timestamp::TIMESTAMP_NTZ AS feature_as_of_ts,
        l.origination_date,
        l.loan_status,

        -- NEW (Tier 1, 2026-06-08): high-risk-neobank flag, as-of safe (a property
        -- of the bank at origination). Adopted from scoring_model_20260429.
        IFF(l.is_neobank_high_risk_institution, 1, 0) AS is_neobank_high_risk_institution,

        /* Outcome / label fields. These are NOT model-time features. */
        IFF(l.loan_status = 'REPAID', 1, 0) AS label_repaid_current_snapshot,
        l.expected_dpd45_date,
        l.expected_dpd45_month,
        l.days_past_due,
        l.charge_off_timestamp,
        l.pre_charge_off_timestamp,
        l.gross_dpd45_amount,

        IFF(l.is_gross_dpd7, 1, 0) AS label_gross_dpd7,
        IFF(l.is_gross_dpd14, 1, 0) AS label_gross_dpd14,
        IFF(l.is_gross_dpd21, 1, 0) AS label_gross_dpd21,
        IFF(l.is_gross_dpd28, 1, 0) AS label_gross_dpd28,
        IFF(l.is_gross_dpd35, 1, 0) AS label_gross_dpd35,
        IFF(l.is_gross_dpd45, 1, 0) AS label_gross_dpd45,

        IFF(l.is_mature_d7, 1, 0) AS label_mature_d7,
        IFF(l.is_mature_d14, 1, 0) AS label_mature_d14,
        IFF(l.is_mature_d21, 1, 0) AS label_mature_d21,
        IFF(l.is_mature_d28, 1, 0) AS label_mature_d28,
        IFF(l.is_mature_d35, 1, 0) AS label_mature_d35,
        IFF(l.is_mature_d45, 1, 0) AS label_mature_d45

    FROM brigit_snowflake.dbt_analytics.fct_loans l
    CROSS JOIN params p
    WHERE l.origination_timestamp >= p.history_start_ts
),

/* ---------------------------------------------------------------------
   2. Anchor advances that will appear in the final base table.
--------------------------------------------------------------------- */
anchor_advances AS (
    SELECT a.*
    FROM all_advances a
    CROSS JOIN params p
    WHERE a.feature_as_of_ts >= p.output_start_ts
),

/* ---------------------------------------------------------------------
   3. Identity table: one row per user.
--------------------------------------------------------------------- */
identities_one_per_user AS (
    SELECT
        i.user_id::VARCHAR AS user_id,
        i.created_time::TIMESTAMP_NTZ AS identity_created_time,
        i.is_deleted,
        i.matched_first_name,
        i.matched_last_name,
        i.first_name,
        i.last_name,
        i.email,
        i.phone_number,
        i.matched_street_address,
        i.matched_city,
        i.matched_zip_code,
        i.matched_state
    FROM brigit_snowflake.dbt_analytics.base_prod__identities i
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY i.user_id
        ORDER BY i.is_deleted ASC, i.created_time DESC
    ) = 1
),

/* ---------------------------------------------------------------------
   4. User-bank-account links.
--------------------------------------------------------------------- */
bank_account_links AS (
    SELECT
        pa.user_id::VARCHAR AS user_id,
        pa.routing_number,
        pa.account_number,

        /* Canonical bank account key. Use this convention everywhere. */
        CONCAT(pa.routing_number, '-', pa.account_number) AS bank_account_key,

        pa.plaid_account_id,

        MAX(pa.persistent_account_id) OVER (
            PARTITION BY pa.user_id, pa.routing_number, pa.account_number
        ) AS persistent_account_id,

        pa.created_at::TIMESTAMP_NTZ AS plaid_account_created_at,
        pa.institution_id,
        m.name AS institution_name,

        -- NEW (Tier 1, 2026-06-08): Plaid account-holder name + joint flag.
        -- official_name -> holder-name-vs-identity-name match (synthetic/stolen
        -- account tell). is_joint -> false-positive suppressor on sharing rules.
        -- VERIFY: confirm base_prod__plaid_accounts carries these columns the
        -- teams queries read them from base_prod__plaid_accounts_current_state.
        pa.official_name,
        pa.is_joint,

        i.identity_created_time,
        i.is_deleted,
        i.matched_first_name,
        i.matched_last_name,
        i.first_name,
        i.last_name,
        i.email,
        i.phone_number,
        i.matched_street_address,
        i.matched_city,
        i.matched_zip_code,
        i.matched_state

    FROM brigit_snowflake.dbt_analytics.base_prod__plaid_accounts pa
    JOIN identities_one_per_user i
        ON pa.user_id::VARCHAR = i.user_id
    LEFT JOIN brigit_snowflake.dbt_analytics.base_prod__mapping_plaid_institutions m
        ON pa.institution_id = m.institution_id
    WHERE pa.routing_number IS NOT NULL
      AND pa.account_number IS NOT NULL
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY pa.user_id, pa.routing_number, pa.account_number
        ORDER BY pa.incrementing_id DESC
    ) = 1
),

/* ---------------------------------------------------------------------
   5. Anchor advances mapped to candidate bank accounts.
--------------------------------------------------------------------- */
anchor_advance_account_candidates AS (
    SELECT
        a.*,

        ba.routing_number,
        ba.account_number,
        ba.bank_account_key,
        ba.plaid_account_id,
        ba.persistent_account_id,
        ba.plaid_account_created_at,
        ba.institution_id,
        ba.institution_name,

        -- NEW (Tier 1, 2026-06-08)
        ba.official_name,
        ba.is_joint,

        ba.identity_created_time,
        ba.matched_first_name,
        ba.matched_last_name,
        ba.first_name,
        ba.last_name,
        ba.email,
        ba.phone_number,
        ba.matched_street_address,
        ba.matched_city,
        ba.matched_zip_code,
        ba.matched_state

    FROM anchor_advances a
    JOIN bank_account_links ba
        ON a.user_id = ba.user_id
       AND ba.plaid_account_created_at <= a.feature_as_of_ts
       AND (
            (
                a.plaid_routing_number IS NOT NULL
                AND a.plaid_account_number IS NOT NULL
                AND a.plaid_routing_number = ba.routing_number
                AND a.plaid_account_number = ba.account_number
            )
            OR (
                a.plaid_routing_number IS NULL
                AND a.plaid_account_number IS NULL
            )
       )
),

/* ---------------------------------------------------------------------
   6. All historical advances mapped to bank accounts (for velocity).
--------------------------------------------------------------------- */
all_advance_account_candidates AS (
    SELECT
        a.advance_id,
        a.user_id,
        a.feature_as_of_ts,
        a.origination_date,
        a.loan_amount,
        a.total_disbursed,
        a.loan_status,
        a.label_repaid_current_snapshot,

        ba.routing_number,
        ba.account_number,
        ba.bank_account_key

    FROM all_advances a
    JOIN bank_account_links ba
        ON a.user_id = ba.user_id
       AND ba.plaid_account_created_at <= a.feature_as_of_ts
       AND (
            (
                a.plaid_routing_number IS NOT NULL
                AND a.plaid_account_number IS NOT NULL
                AND a.plaid_routing_number = ba.routing_number
                AND a.plaid_account_number = ba.account_number
            )
            OR (
                a.plaid_routing_number IS NULL
                AND a.plaid_account_number IS NULL
            )
       )
),

/* ---------------------------------------------------------------------
   7. User account count as of advance timestamp.
--------------------------------------------------------------------- */
user_account_counts_asof AS (
    SELECT
        a.advance_id,
        COUNT(DISTINCT other.bank_account_key) AS bank_accounts_per_user_asof
    FROM anchor_advance_account_candidates a
    LEFT JOIN bank_account_links other
        ON a.user_id = other.user_id
       AND other.plaid_account_created_at <= a.feature_as_of_ts
    GROUP BY 1
),

/* ---------------------------------------------------------------------
   8. Shared bank-account identity velocity (as-of, prior-only).
--------------------------------------------------------------------- */
bank_account_user_features AS (
    SELECT
        a.advance_id,
        a.bank_account_key,

        COUNT(DISTINCT other.user_id) AS users_on_bank_account_lifetime_asof,

        COUNT(DISTINCT IFF(
            other.identity_created_time >= DATEADD(hour, -72, a.feature_as_of_ts)
            AND other.identity_created_time <= a.feature_as_of_ts,
            other.user_id,
            NULL
        )) AS users_on_bank_account_72h,

        COUNT(DISTINCT IFF(
            other.identity_created_time >= DATEADD(day, -7, a.feature_as_of_ts)
            AND other.identity_created_time <= a.feature_as_of_ts,
            other.user_id,
            NULL
        )) AS users_on_bank_account_7d,

        COUNT(DISTINCT IFF(
            other.identity_created_time >= DATEADD(day, -30, a.feature_as_of_ts)
            AND other.identity_created_time <= a.feature_as_of_ts,
            other.user_id,
            NULL
        )) AS users_on_bank_account_30d,

        COUNT(DISTINCT IFF(
            other.identity_created_time >= DATEADD(day, -90, a.feature_as_of_ts)
            AND other.identity_created_time <= a.feature_as_of_ts,
            other.user_id,
            NULL
        )) AS users_on_bank_account_90d,

        MIN(other.identity_created_time) AS first_identity_created_on_account_asof,
        MAX(other.identity_created_time) AS latest_identity_created_on_account_asof,

        DATEDIFF(
            day,
            MIN(other.identity_created_time),
            MAX(other.identity_created_time)
        ) AS user_creation_days_span_asof

    FROM anchor_advance_account_candidates a
    LEFT JOIN bank_account_links other
        ON a.routing_number = other.routing_number
       AND a.account_number = other.account_number
       AND other.plaid_account_created_at <= a.feature_as_of_ts
       AND other.identity_created_time <= a.feature_as_of_ts
    GROUP BY 1, 2
),

bank_account_user_features_final AS (
    SELECT
        f.*,

        IFF(f.users_on_bank_account_72h >= 3, 1, 0) AS flag_3_users_on_bank_account_72h,
        IFF(f.users_on_bank_account_lifetime_asof >= 5, 1, 0) AS flag_5_users_on_bank_account_ever_asof,

        -- NEW (Tier 1, 2026-06-08): the clawback lists 10+ threshold, as-of.
        -- Completes the teams core detection set (3-in-72h, 5-ever, 10-ever).
        IFF(f.users_on_bank_account_lifetime_asof >= 10, 1, 0) AS flag_10_users_on_bank_account_ever_asof,

        ROUND(
            f.users_on_bank_account_lifetime_asof::FLOAT
            / NULLIF(f.user_creation_days_span_asof, 0),
            4
        ) AS avg_users_created_per_day_asof,

        ROUND(
            f.users_on_bank_account_lifetime_asof::FLOAT
            / NULLIF(f.user_creation_days_span_asof / 30.0, 0),
            2
        ) AS avg_users_created_per_month_asof

    FROM bank_account_user_features f
),

/* ---------------------------------------------------------------------
   9. Prior advance velocity on the same bank account (prior-only).
--------------------------------------------------------------------- */
bank_account_advance_features AS (
    SELECT
        a.advance_id,
        a.bank_account_key,

        COUNT(DISTINCT prior.advance_id) AS prior_advances_on_bank_account_lifetime,

        COUNT(DISTINCT IFF(
            prior.feature_as_of_ts >= DATEADD(hour, -24, a.feature_as_of_ts),
            prior.advance_id,
            NULL
        )) AS prior_advances_on_bank_account_24h,

        COUNT(DISTINCT IFF(
            prior.feature_as_of_ts >= DATEADD(hour, -72, a.feature_as_of_ts),
            prior.advance_id,
            NULL
        )) AS prior_advances_on_bank_account_72h,

        COUNT(DISTINCT IFF(
            prior.feature_as_of_ts >= DATEADD(day, -7, a.feature_as_of_ts),
            prior.advance_id,
            NULL
        )) AS prior_advances_on_bank_account_7d,

        COUNT(DISTINCT IFF(
            prior.feature_as_of_ts >= DATEADD(day, -30, a.feature_as_of_ts),
            prior.advance_id,
            NULL
        )) AS prior_advances_on_bank_account_30d,

        SUM(IFF(
            prior.feature_as_of_ts >= DATEADD(day, -30, a.feature_as_of_ts),
            prior.loan_amount,
            0
        )) AS prior_loan_amount_sum_30d,

        AVG(IFF(
            prior.feature_as_of_ts >= DATEADD(day, -30, a.feature_as_of_ts),
            prior.loan_amount,
            NULL
        )) AS prior_loan_amount_avg_30d,

        SUM(IFF(
            prior.feature_as_of_ts >= DATEADD(day, -30, a.feature_as_of_ts),
            prior.total_disbursed,
            0
        )) AS prior_total_disbursed_sum_30d,

        MAX(prior.feature_as_of_ts) AS previous_advance_on_account_ts,

        DATEDIFF(
            hour,
            MAX(prior.feature_as_of_ts),
            a.feature_as_of_ts
        ) AS hours_since_previous_advance_on_account,

        DATEDIFF(
            day,
            MIN(prior.feature_as_of_ts),
            MAX(prior.feature_as_of_ts)
        ) AS prior_advance_days_span,

        ROUND(
            COUNT(DISTINCT prior.advance_id)::FLOAT
            / NULLIF(DATEDIFF(day, MIN(prior.feature_as_of_ts), MAX(prior.feature_as_of_ts)), 0),
            4
        ) AS avg_prior_advances_per_day,

        ROUND(
            COUNT(DISTINCT prior.advance_id)::FLOAT
            / NULLIF(DATEDIFF(day, MIN(prior.feature_as_of_ts), MAX(prior.feature_as_of_ts)) / 30.0, 0),
            2
        ) AS avg_prior_advances_per_month

    FROM anchor_advance_account_candidates a
    LEFT JOIN all_advance_account_candidates prior
        ON a.bank_account_key = prior.bank_account_key
       AND prior.feature_as_of_ts < a.feature_as_of_ts
    GROUP BY 1, 2, a.feature_as_of_ts
),

/* =====================================================================
   NEW (Tier 1, 2026-06-08): prior-only minimum gap between advances on
   the bank account. Adapted from scoring_model_20260429s min_advance_distance
   (which is global/hindsight) -> made prior-only here.
   Step A: consecutive gap for every advance on the account (LAG).
   Step B: per anchor, MIN over gaps whose advance is at/<= the anchors time.
   ===================================================================== */
advance_gaps AS (
    SELECT
        advance_id,
        bank_account_key,
        feature_as_of_ts,
        ABS(TIMESTAMPDIFF(
            'hour',
            feature_as_of_ts,
            LAG(feature_as_of_ts) OVER (
                PARTITION BY bank_account_key
                ORDER BY feature_as_of_ts, advance_id
            )
        )) AS hours_to_prev
    FROM all_advance_account_candidates
),

prior_min_advance_gap AS (
    SELECT
        a.advance_id,
        a.bank_account_key,
        MIN(g.hours_to_prev) AS prior_min_hours_between_advances_on_account
    FROM anchor_advance_account_candidates a
    LEFT JOIN advance_gaps g
        ON a.bank_account_key = g.bank_account_key
       AND g.feature_as_of_ts <= a.feature_as_of_ts
       AND g.hours_to_prev IS NOT NULL
    GROUP BY 1, 2
),

/* =====================================================================
   NEW (Tier 1, 2026-06-08): persistent_account_id sharing (as-of, prior-only).
   Mirrors bank_account_user_features but keyed on persistent_account_id (the
   stable Plaid account id). Same grain (advance_id, bank_account_key) so it
   flows through the final per-advance dedup.
   ===================================================================== */
persistent_account_user_features AS (
    SELECT
        a.advance_id,
        a.bank_account_key,

        COUNT(DISTINCT IFF(
            other.identity_created_time >= DATEADD(hour, -72, a.feature_as_of_ts)
            AND other.identity_created_time <= a.feature_as_of_ts,
            other.user_id,
            NULL
        )) AS users_on_persistent_account_id_72h,

        COUNT(DISTINCT IFF(
            other.identity_created_time >= DATEADD(day, -7, a.feature_as_of_ts)
            AND other.identity_created_time <= a.feature_as_of_ts,
            other.user_id,
            NULL
        )) AS users_on_persistent_account_id_7d,

        COUNT(DISTINCT IFF(
            other.identity_created_time >= DATEADD(day, -30, a.feature_as_of_ts)
            AND other.identity_created_time <= a.feature_as_of_ts,
            other.user_id,
            NULL
        )) AS users_on_persistent_account_id_30d

    FROM anchor_advance_account_candidates a
    LEFT JOIN bank_account_links other
        ON a.persistent_account_id = other.persistent_account_id
       AND a.persistent_account_id IS NOT NULL
       AND other.plaid_account_created_at <= a.feature_as_of_ts
       AND other.identity_created_time <= a.feature_as_of_ts
    GROUP BY 1, 2
),

/* =====================================================================
   NEW (Tier 1, 2026-06-08): normalized identity attributes per user, with a
   first-pass sentinel screen. The dummy lists are a STARTING POINT refine via
   data/queries/diagnostics_sentinel_screen.sql (group-by-count the high-fan-out
   values, then extend the screens). NULL never matches, so a screened value is
   silently excluded from every sharing count.
   ===================================================================== */
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
            -- invalid-zip / data-entry placeholders seen in the fan-out scan
            WHEN TRIM(i.matched_zip_code) IN ('00000', '00003') THEN NULL
            WHEN UPPER(TRIM(i.matched_street_address)) IN ('MAIL RETURN', 'PLEASE UPDATE') THEN NULL
            ELSE UPPER(TRIM(i.matched_street_address)) || '|' || TRIM(i.matched_zip_code)
        END AS address_key

    FROM identities_one_per_user i
),

-- The anchor advances own normalized attributes (one row per advance).
anchor_identity AS (
    SELECT
        a.advance_id,
        a.user_id,
        a.feature_as_of_ts,
        l.email_norm,
        l.phone_norm,
        l.address_key
    FROM anchor_advances a
    JOIN identity_attribute_links l
        ON a.user_id = l.user_id
),

-- Email sharing (as-of, prior-only windowed on the OTHER users identity age).
email_user_features AS (
    SELECT
        a.advance_id,
        COUNT(DISTINCT IFF(
            other.identity_created_time >= DATEADD(hour, -72, a.feature_as_of_ts)
            AND other.identity_created_time <= a.feature_as_of_ts,
            other.user_id, NULL)) AS users_on_email_72h,
        COUNT(DISTINCT IFF(
            other.identity_created_time >= DATEADD(day, -7, a.feature_as_of_ts)
            AND other.identity_created_time <= a.feature_as_of_ts,
            other.user_id, NULL)) AS users_on_email_7d,
        COUNT(DISTINCT IFF(
            other.identity_created_time >= DATEADD(day, -30, a.feature_as_of_ts)
            AND other.identity_created_time <= a.feature_as_of_ts,
            other.user_id, NULL)) AS users_on_email_30d
    FROM anchor_identity a
    JOIN identity_attribute_links other
        ON a.email_norm = other.email_norm
       AND other.identity_created_time <= a.feature_as_of_ts
    WHERE a.email_norm IS NOT NULL
    GROUP BY 1
),

-- Phone sharing.
phone_user_features AS (
    SELECT
        a.advance_id,
        COUNT(DISTINCT IFF(
            other.identity_created_time >= DATEADD(hour, -72, a.feature_as_of_ts)
            AND other.identity_created_time <= a.feature_as_of_ts,
            other.user_id, NULL)) AS users_on_phone_72h,
        COUNT(DISTINCT IFF(
            other.identity_created_time >= DATEADD(day, -7, a.feature_as_of_ts)
            AND other.identity_created_time <= a.feature_as_of_ts,
            other.user_id, NULL)) AS users_on_phone_7d,
        COUNT(DISTINCT IFF(
            other.identity_created_time >= DATEADD(day, -30, a.feature_as_of_ts)
            AND other.identity_created_time <= a.feature_as_of_ts,
            other.user_id, NULL)) AS users_on_phone_30d
    FROM anchor_identity a
    JOIN identity_attribute_links other
        ON a.phone_norm = other.phone_norm
       AND other.identity_created_time <= a.feature_as_of_ts
    WHERE a.phone_norm IS NOT NULL
    GROUP BY 1
),

-- Address sharing. NOTE: has an innocent version (families/roommates) -> a
-- scenario built on this needs a disqualifier the raw count is still a feature.
address_user_features AS (
    SELECT
        a.advance_id,
        COUNT(DISTINCT IFF(
            other.identity_created_time >= DATEADD(hour, -72, a.feature_as_of_ts)
            AND other.identity_created_time <= a.feature_as_of_ts,
            other.user_id, NULL)) AS users_on_address_72h,
        COUNT(DISTINCT IFF(
            other.identity_created_time >= DATEADD(day, -7, a.feature_as_of_ts)
            AND other.identity_created_time <= a.feature_as_of_ts,
            other.user_id, NULL)) AS users_on_address_7d,
        COUNT(DISTINCT IFF(
            other.identity_created_time >= DATEADD(day, -30, a.feature_as_of_ts)
            AND other.identity_created_time <= a.feature_as_of_ts,
            other.user_id, NULL)) AS users_on_address_30d
    FROM anchor_identity a
    JOIN identity_attribute_links other
        ON a.address_key = other.address_key
       AND other.identity_created_time <= a.feature_as_of_ts
    WHERE a.address_key IS NOT NULL
    GROUP BY 1
),

/* =====================================================================
   NEW (Tier 1, 2026-06-08): device_id sharing (as-of, prior-only).
   client_metadata is SCD (valid_from) the anchors device is the latest one
   valid at/<= the advance. The links carry each users identity_created_time
   (for the freshness window) and the devices valid_from (for as-of).
   ===================================================================== */
client_metadata AS (
    SELECT
        cm.identifying_id::VARCHAR AS user_id,
        cm.valid_from::TIMESTAMP_NTZ AS valid_from,
        cm.device_id,
        cm.device_platform,
        cm.ip_address
    FROM pc_fivetran_db.wal_brigit_production_public.user_client_metadata cm
    WHERE cm._fivetran_deleted = FALSE
),

device_links AS (
    -- One row per (user, device). user_client_metadata is SCD and averages ~11
    -- rows per user-device (176M raw -> ~16M) collapsing here kills the device
    -- self-join fan-out. Use the EARLIEST valid_from (when the device first
    -- associated with the user) as the as-of bound. identity_created_time is
    -- one-per-user, so MAX() just picks that single value.
    SELECT
        cm.user_id,
        cm.device_id,
        MIN(cm.valid_from) AS device_valid_from,
        MAX(i.identity_created_time) AS identity_created_time
    FROM client_metadata cm
    JOIN identities_one_per_user i
        ON cm.user_id = i.user_id
    WHERE cm.device_id IS NOT NULL
    GROUP BY cm.user_id, cm.device_id
),

anchor_device AS (
    SELECT advance_id, user_id, feature_as_of_ts, device_id
    FROM (
        SELECT
            a.advance_id,
            a.user_id,
            a.feature_as_of_ts,
            cm.device_id,
            ROW_NUMBER() OVER (
                PARTITION BY a.advance_id
                ORDER BY cm.valid_from DESC NULLS LAST
            ) AS rn
        FROM anchor_advances a
        LEFT JOIN client_metadata cm
            ON a.user_id = cm.user_id
           AND cm.valid_from <= a.feature_as_of_ts
           AND cm.device_id IS NOT NULL
    )
    WHERE rn = 1
),

device_user_features AS (
    SELECT
        a.advance_id,
        COUNT(DISTINCT IFF(
            other.identity_created_time >= DATEADD(hour, -72, a.feature_as_of_ts)
            AND other.identity_created_time <= a.feature_as_of_ts,
            other.user_id, NULL)) AS users_on_device_id_72h,
        COUNT(DISTINCT IFF(
            other.identity_created_time >= DATEADD(day, -7, a.feature_as_of_ts)
            AND other.identity_created_time <= a.feature_as_of_ts,
            other.user_id, NULL)) AS users_on_device_id_7d,
        COUNT(DISTINCT IFF(
            other.identity_created_time >= DATEADD(day, -30, a.feature_as_of_ts)
            AND other.identity_created_time <= a.feature_as_of_ts,
            other.user_id, NULL)) AS users_on_device_id_30d
    FROM anchor_device a
    JOIN device_links other
        ON a.device_id = other.device_id
       AND other.device_valid_from <= a.feature_as_of_ts
       AND other.identity_created_time <= a.feature_as_of_ts
    WHERE a.device_id IS NOT NULL
    GROUP BY 1
),

/* ---------------------------------------------------------------------
   10. Lightweight bank-account network score (UNCHANGED).
--------------------------------------------------------------------- */
network_features AS (
    SELECT
        a.advance_id,
        a.bank_account_key,

        COALESCE(baf.users_on_bank_account_lifetime_asof, 0) AS network_user_count_asof,
        COALESCE(uac.bank_accounts_per_user_asof, 0) AS network_account_count_asof,

        COALESCE(baf.users_on_bank_account_lifetime_asof, 0)
        * GREATEST(COALESCE(uac.bank_accounts_per_user_asof, 0), 1) AS network_score_asof,

        CASE
            WHEN COALESCE(baf.users_on_bank_account_lifetime_asof, 0)
                 * GREATEST(COALESCE(uac.bank_accounts_per_user_asof, 0), 1) >= 15
            THEN
                CHR(65 + MOD(ABS(HASH(a.bank_account_key)),                 26))
                || CHR(65 + MOD(FLOOR(ABS(HASH(a.bank_account_key)) /  26), 26))
                || CHR(65 + MOD(FLOOR(ABS(HASH(a.bank_account_key)) / 676), 26))
            ELSE 'QQQ'
        END AS network_label

    FROM anchor_advance_account_candidates a
    LEFT JOIN bank_account_user_features_final baf
        ON a.advance_id = baf.advance_id
       AND a.bank_account_key = baf.bank_account_key
    LEFT JOIN user_account_counts_asof uac
        ON a.advance_id = uac.advance_id
),

/* ---------------------------------------------------------------------
   11. KYC source table (UNCHANGED).
--------------------------------------------------------------------- */
socure_kyc AS (
    SELECT
        s.id AS socure_id,
        s.user_id::VARCHAR AS user_id,
        s.created_at::TIMESTAMP_NTZ AS socure_created_at,
        s.decision AS socure_decision,
        s.status AS socure_status
    FROM brigit_snowflake.dbt_analytics.base_prod__socure_identities s
    WHERE s._fivetran_deleted = FALSE
),

/* ---------------------------------------------------------------------
   12A. Assemble feature snapshot before KYC / client metadata.
--------------------------------------------------------------------- */
feature_snapshot_pre_enrichment AS (
    SELECT
        a.advance_id,
        a.user_id,
        a.feature_as_of_ts,
        a.origination_date,

        /* Advance base features */
        a.loan_amount,
        a.express_transfer_fee,
        a.total_disbursed,
        DATE_PART(hour, a.feature_as_of_ts) AS origination_hour,
        DAYOFWEEKISO(a.feature_as_of_ts) AS origination_day_of_week,
        IFF(DAYOFWEEKISO(a.feature_as_of_ts) IN (6, 7), 1, 0) AS is_weekend_origination,

        -- NEW (Tier 1, 2026-06-08)
        a.is_neobank_high_risk_institution,

        /* Bank account identifiers */
        a.routing_number,
        a.account_number,
        a.bank_account_key,
        a.plaid_account_id,
        a.persistent_account_id,
        IFF(a.persistent_account_id IS NOT NULL, 1, 0) AS has_persistent_account_id,
        a.institution_id,
        a.institution_name,

        -- NEW (Tier 1, 2026-06-08): joint-account flag (false-positive suppressor).
        IFF(a.is_joint, 1, 0) AS is_joint,

        /* User/account relationship */
        a.identity_created_time,
        a.plaid_account_created_at,
        DATEDIFF(day, a.identity_created_time, a.feature_as_of_ts) AS days_since_identity_created,
        DATEDIFF(day, a.plaid_account_created_at, a.feature_as_of_ts) AS days_since_plaid_account_created,
        DATEDIFF(day, a.identity_created_time, a.plaid_account_created_at) AS days_between_identity_and_bank_account_creation,
        -- NEW (Tier 1, 2026-06-08): identity->advance speed in HOURS (round-1s
        -- top blind-spot factor: E_L median ~12 minutes). Finer than the days field.
        DATEDIFF(hour, a.identity_created_time, a.feature_as_of_ts) AS hours_since_identity_created,
        uac.bank_accounts_per_user_asof,

        /* Shared bank-account identity velocity */
        baf.users_on_bank_account_lifetime_asof,
        baf.users_on_bank_account_72h,
        baf.users_on_bank_account_7d,
        baf.users_on_bank_account_30d,
        baf.users_on_bank_account_90d,
        baf.flag_3_users_on_bank_account_72h,
        baf.flag_5_users_on_bank_account_ever_asof,
        baf.flag_10_users_on_bank_account_ever_asof,   -- NEW (Tier 1, 2026-06-08)
        baf.first_identity_created_on_account_asof,
        baf.latest_identity_created_on_account_asof,
        baf.user_creation_days_span_asof,
        baf.avg_users_created_per_day_asof,
        baf.avg_users_created_per_month_asof,

        -- NEW (Tier 1, 2026-06-08): other scarce-resource sharing edges, as-of.
        COALESCE(pauf.users_on_persistent_account_id_72h, 0) AS users_on_persistent_account_id_72h,
        COALESCE(pauf.users_on_persistent_account_id_7d, 0)  AS users_on_persistent_account_id_7d,
        COALESCE(pauf.users_on_persistent_account_id_30d, 0) AS users_on_persistent_account_id_30d,
        COALESCE(euf.users_on_email_72h, 0) AS users_on_email_72h,
        COALESCE(euf.users_on_email_7d, 0)  AS users_on_email_7d,
        COALESCE(euf.users_on_email_30d, 0) AS users_on_email_30d,
        COALESCE(puf.users_on_phone_72h, 0) AS users_on_phone_72h,
        COALESCE(puf.users_on_phone_7d, 0)  AS users_on_phone_7d,
        COALESCE(puf.users_on_phone_30d, 0) AS users_on_phone_30d,
        COALESCE(auf.users_on_address_72h, 0) AS users_on_address_72h,
        COALESCE(auf.users_on_address_7d, 0)  AS users_on_address_7d,
        COALESCE(auf.users_on_address_30d, 0) AS users_on_address_30d,
        COALESCE(duf.users_on_device_id_72h, 0) AS users_on_device_id_72h,
        COALESCE(duf.users_on_device_id_7d, 0)  AS users_on_device_id_7d,
        COALESCE(duf.users_on_device_id_30d, 0) AS users_on_device_id_30d,

        -- NEW (Tier 1, 2026-06-08): name-matchiness (Jaro-Winkler, 0-100). Higher
        -- = more similar. Low entered-vs-matched similarity is a synthetic/stolen
        -- identity tell (validate on the residual ruler, not the heuristic band).
        JAROWINKLER_SIMILARITY(UPPER(TRIM(a.first_name)), UPPER(TRIM(a.matched_first_name))) AS name_match_first,
        JAROWINKLER_SIMILARITY(UPPER(TRIM(a.last_name)),  UPPER(TRIM(a.matched_last_name)))  AS name_match_last,
        -- NOTE: name_match_official was DROPPED 2026-06-08 after a live data check
        -- showed base_prod__plaid_accounts.official_name is the account PRODUCT type
        -- (Checking / Varo Checking / Individual Account / ...), NOT the holder name.
        -- Holder-name match needs the Plaid identity/owner source (Tier-3 pull).

        /* Prior advance velocity */
        aaf.prior_advances_on_bank_account_lifetime,
        aaf.prior_advances_on_bank_account_24h,
        aaf.prior_advances_on_bank_account_72h,
        aaf.prior_advances_on_bank_account_7d,
        aaf.prior_advances_on_bank_account_30d,
        aaf.prior_loan_amount_sum_30d,
        aaf.prior_loan_amount_avg_30d,
        aaf.prior_total_disbursed_sum_30d,
        aaf.previous_advance_on_account_ts,
        aaf.hours_since_previous_advance_on_account,
        aaf.prior_advance_days_span,
        aaf.avg_prior_advances_per_day,
        aaf.avg_prior_advances_per_month,
        pmg.prior_min_hours_between_advances_on_account,  -- NEW (Tier 1, 2026-06-08)

        /* Lightweight network / ring-style fields */
        nf.network_user_count_asof,
        nf.network_account_count_asof,
        nf.network_score_asof,
        nf.network_label,

        /* Channel / platform */
        ma.new_sub_channel_design_appsflyer_source AS acquisition_channel,
        usr.signup_source AS platform,
        usr.signup_ip,

        /* Heuristic fraud score for review / investigation (UNCHANGED) */
        LEAST(
            100,

            IFF(COALESCE(baf.users_on_bank_account_72h, 0) >= 4, 25,
                IFF(COALESCE(baf.users_on_bank_account_72h, 0) >= 3, 15, 0)
            )

            + IFF(COALESCE(baf.users_on_bank_account_lifetime_asof, 0) >= 10, 25,
                IFF(COALESCE(baf.users_on_bank_account_lifetime_asof, 0) >= 5, 15,
                    IFF(COALESCE(baf.users_on_bank_account_lifetime_asof, 0) >= 3, 5, 0)
                )
            )

            + IFF(COALESCE(baf.avg_users_created_per_month_asof, 0) >= 10, 15,
                IFF(COALESCE(baf.avg_users_created_per_month_asof, 0) >= 3, 8, 0)
            )

            + IFF(COALESCE(aaf.prior_advances_on_bank_account_72h, 0) >= 2, 20,
                IFF(COALESCE(aaf.prior_advances_on_bank_account_7d, 0) >= 2, 10, 0)
            )

            + IFF(COALESCE(nf.network_score_asof, 0) >= 15, 15, 0)
        ) AS heuristic_fraud_score,

        CASE
            WHEN LEAST(
                100,
                IFF(COALESCE(baf.users_on_bank_account_72h, 0) >= 4, 25,
                    IFF(COALESCE(baf.users_on_bank_account_72h, 0) >= 3, 15, 0)
                )
                + IFF(COALESCE(baf.users_on_bank_account_lifetime_asof, 0) >= 10, 25,
                    IFF(COALESCE(baf.users_on_bank_account_lifetime_asof, 0) >= 5, 15,
                        IFF(COALESCE(baf.users_on_bank_account_lifetime_asof, 0) >= 3, 5, 0)
                    )
                )
                + IFF(COALESCE(baf.avg_users_created_per_month_asof, 0) >= 10, 15,
                    IFF(COALESCE(baf.avg_users_created_per_month_asof, 0) >= 3, 8, 0)
                )
                + IFF(COALESCE(aaf.prior_advances_on_bank_account_72h, 0) >= 2, 20,
                    IFF(COALESCE(aaf.prior_advances_on_bank_account_7d, 0) >= 2, 10, 0)
                )
                + IFF(COALESCE(nf.network_score_asof, 0) >= 15, 15, 0)
            ) <= 15 THEN 'LOW'

            WHEN LEAST(
                100,
                IFF(COALESCE(baf.users_on_bank_account_72h, 0) >= 4, 25,
                    IFF(COALESCE(baf.users_on_bank_account_72h, 0) >= 3, 15, 0)
                )
                + IFF(COALESCE(baf.users_on_bank_account_lifetime_asof, 0) >= 10, 25,
                    IFF(COALESCE(baf.users_on_bank_account_lifetime_asof, 0) >= 5, 15,
                        IFF(COALESCE(baf.users_on_bank_account_lifetime_asof, 0) >= 3, 5, 0)
                    )
                )
                + IFF(COALESCE(baf.avg_users_created_per_month_asof, 0) >= 10, 15,
                    IFF(COALESCE(baf.avg_users_created_per_month_asof, 0) >= 3, 8, 0)
                )
                + IFF(COALESCE(aaf.prior_advances_on_bank_account_72h, 0) >= 2, 20,
                    IFF(COALESCE(aaf.prior_advances_on_bank_account_7d, 0) >= 2, 10, 0)
                )
                + IFF(COALESCE(nf.network_score_asof, 0) >= 15, 15, 0)
            ) <= 35 THEN 'POSSIBLE'

            WHEN LEAST(
                100,
                IFF(COALESCE(baf.users_on_bank_account_72h, 0) >= 4, 25,
                    IFF(COALESCE(baf.users_on_bank_account_72h, 0) >= 3, 15, 0)
                )
                + IFF(COALESCE(baf.users_on_bank_account_lifetime_asof, 0) >= 10, 25,
                    IFF(COALESCE(baf.users_on_bank_account_lifetime_asof, 0) >= 5, 15,
                        IFF(COALESCE(baf.users_on_bank_account_lifetime_asof, 0) >= 3, 5, 0)
                    )
                )
                + IFF(COALESCE(baf.avg_users_created_per_month_asof, 0) >= 10, 15,
                    IFF(COALESCE(baf.avg_users_created_per_month_asof, 0) >= 3, 8, 0)
                )
                + IFF(COALESCE(aaf.prior_advances_on_bank_account_72h, 0) >= 2, 20,
                    IFF(COALESCE(aaf.prior_advances_on_bank_account_7d, 0) >= 2, 10, 0)
                )
                + IFF(COALESCE(nf.network_score_asof, 0) >= 15, 15, 0)
            ) <= 60 THEN 'LIKELY'

            ELSE 'EXTREMELY_LIKELY'
        END AS heuristic_fraud_band,

        /* Labels / outcomes */
        a.label_repaid_current_snapshot,
        a.expected_dpd45_date,
        a.expected_dpd45_month,
        a.days_past_due,
        a.charge_off_timestamp,
        a.pre_charge_off_timestamp,
        a.gross_dpd45_amount,
        a.label_gross_dpd7,
        a.label_gross_dpd14,
        a.label_gross_dpd21,
        a.label_gross_dpd28,
        a.label_gross_dpd35,
        a.label_gross_dpd45,
        a.label_mature_d7,
        a.label_mature_d14,
        a.label_mature_d21,
        a.label_mature_d28,
        a.label_mature_d35,
        a.label_mature_d45

    FROM anchor_advance_account_candidates a

    LEFT JOIN user_account_counts_asof uac
        ON a.advance_id = uac.advance_id

    LEFT JOIN bank_account_user_features_final baf
        ON a.advance_id = baf.advance_id
       AND a.bank_account_key = baf.bank_account_key

    LEFT JOIN bank_account_advance_features aaf
        ON a.advance_id = aaf.advance_id
       AND a.bank_account_key = aaf.bank_account_key

    -- NEW (Tier 1, 2026-06-08): (advance_id, bank_account_key)-grain joins.
    LEFT JOIN persistent_account_user_features pauf
        ON a.advance_id = pauf.advance_id
       AND a.bank_account_key = pauf.bank_account_key

    LEFT JOIN prior_min_advance_gap pmg
        ON a.advance_id = pmg.advance_id
       AND a.bank_account_key = pmg.bank_account_key

    -- NEW (Tier 1, 2026-06-08): advance_id-grain joins (one row per advance
    -- constant across an advances candidate accounts, so no fan-out).
    LEFT JOIN email_user_features euf   ON a.advance_id = euf.advance_id
    LEFT JOIN phone_user_features puf   ON a.advance_id = puf.advance_id
    LEFT JOIN address_user_features auf ON a.advance_id = auf.advance_id
    LEFT JOIN device_user_features duf  ON a.advance_id = duf.advance_id

    LEFT JOIN network_features nf
        ON a.advance_id = nf.advance_id
       AND a.bank_account_key = nf.bank_account_key

    LEFT JOIN brigit_snowflake.dbt_analytics.fct_marketing_attribution ma
        ON a.user_id = ma.user_id::VARCHAR

    LEFT JOIN brigit_snowflake.dbt_analytics.base_prod__users usr
        ON a.user_id = usr.user_id::VARCHAR
),

/* ---------------------------------------------------------------------
   12B. Add latest Socure/KYC row as of the advance timestamp (UNCHANGED).
--------------------------------------------------------------------- */
feature_snapshot_with_kyc AS (
    SELECT
        b.*,

        sk.socure_id,
        sk.socure_created_at,
        sk.socure_decision,
        sk.socure_status,
        IFF(sk.socure_id IS NOT NULL, 1, 0) AS has_kyc,
        DATEDIFF(hour, sk.socure_created_at, b.feature_as_of_ts) AS hours_since_socure_created

    FROM feature_snapshot_pre_enrichment b
    LEFT JOIN socure_kyc sk
        ON b.user_id = sk.user_id
       AND sk.socure_created_at <= b.feature_as_of_ts

    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY b.advance_id, b.bank_account_key
        ORDER BY
            sk.socure_created_at DESC NULLS LAST,
            sk.socure_id DESC NULLS LAST
    ) = 1
),

/* ---------------------------------------------------------------------
   12C. Add latest client metadata row as of the advance timestamp (UNCHANGED).
--------------------------------------------------------------------- */
feature_snapshot_raw AS (
    SELECT
        k.*,

        cm.device_id,
        cm.device_platform,
        cm.ip_address,
        IFF(cm.device_id IS NOT NULL, 1, 0) AS has_device_id,
        IFF(cm.ip_address IS NOT NULL, 1, 0) AS has_ip_address,

        IFF(
            k.signup_ip IS NOT NULL
            AND cm.ip_address IS NOT NULL
            AND k.signup_ip = cm.ip_address,
            1,
            0
        ) AS signup_ip_matches_latest_ip

    FROM feature_snapshot_with_kyc k
    LEFT JOIN client_metadata cm
        ON k.user_id = cm.user_id
       AND cm.valid_from <= k.feature_as_of_ts

    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY k.advance_id, k.bank_account_key
        ORDER BY
            cm.valid_from DESC NULLS LAST,
            cm.device_id DESC NULLS LAST,
            cm.ip_address DESC NULLS LAST
    ) = 1
)

/* ---------------------------------------------------------------------
   13. Final output. Deduping, not sampling. (UNCHANGED dedup ordering.)
--------------------------------------------------------------------- */
SELECT *
FROM feature_snapshot_raw
QUALIFY ROW_NUMBER() OVER (
    PARTITION BY advance_id
    ORDER BY
        users_on_bank_account_lifetime_asof DESC NULLS LAST,
        users_on_bank_account_72h DESC NULLS LAST,
        prior_advances_on_bank_account_30d DESC NULLS LAST,
        heuristic_fraud_score DESC NULLS LAST,
        days_since_plaid_account_created ASC NULLS LAST,
        plaid_account_created_at DESC NULLS LAST
) = 1
