-- Reference only: DDL for the upstream feature snapshot, built and run
-- OUTSIDE the harness (table already exists in brigit_data_science.SANDBOX_WLIU).
-- The harness never executes this file; base_table.sql selects over the
-- resulting table. Kept so the project records how its source data was built.

CREATE OR REPLACE TABLE brigit_data_science.SANDBOX_WLIU.fraud_advance_feature_base AS

WITH params AS (
    SELECT
        '2025-11-01'::TIMESTAMP_NTZ AS history_start_ts,
        '2025-12-01'::TIMESTAMP_NTZ AS output_start_ts
),

/* ---------------------------------------------------------------------
   1. All advances used for historical context.
   This is wider than the final output window because prior-advance features
   need lookback history before the review period.
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
   This is the output population. No sampling happens here.
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
   This is NOT filtered to the anchor advances only, because shared-account
   features need to count other users on the same bank account.
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
   Prefer exact routing/account match from fct_loans when populated.
   If fct_loans does not have routing/account, fall back to user's active
   bank accounts as of origination timestamp.
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
   6. All historical advances mapped to bank accounts.
   Used only for prior-advance velocity.
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
   8. Shared bank-account identity velocity.
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
   9. Prior advance velocity on the same bank account.
   These are prior-only relative to each advance.
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

/* ---------------------------------------------------------------------
   10. Lightweight bank-account network score.
   This is not the full exact fingerprint-ring query. It is a production-
   friendly network score using:
      shared users on the bank account * user's bank-account count
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
   11. KYC and client metadata source tables.
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
/* ---------------------------------------------------------------------
   12A. Assemble feature snapshot before KYC / client metadata.
   We avoid LEFT JOIN LATERAL because Snowflake can throw:
   "Unsupported subquery type cannot be evaluated"
   when using correlated ORDER BY / LIMIT subqueries.
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

        /* Bank account identifiers */
        a.routing_number,
        a.account_number,
        a.bank_account_key,
        a.plaid_account_id,
        a.persistent_account_id,
        IFF(a.persistent_account_id IS NOT NULL, 1, 0) AS has_persistent_account_id,
        a.institution_id,
        a.institution_name,

        /* User/account relationship */
        a.identity_created_time,
        a.plaid_account_created_at,
        DATEDIFF(day, a.identity_created_time, a.feature_as_of_ts) AS days_since_identity_created,
        DATEDIFF(day, a.plaid_account_created_at, a.feature_as_of_ts) AS days_since_plaid_account_created,
        DATEDIFF(day, a.identity_created_time, a.plaid_account_created_at) AS days_between_identity_and_bank_account_creation,
        uac.bank_accounts_per_user_asof,

        /* Shared bank-account identity velocity */
        baf.users_on_bank_account_lifetime_asof,
        baf.users_on_bank_account_72h,
        baf.users_on_bank_account_7d,
        baf.users_on_bank_account_30d,
        baf.users_on_bank_account_90d,
        baf.flag_3_users_on_bank_account_72h,
        baf.flag_5_users_on_bank_account_ever_asof,
        baf.first_identity_created_on_account_asof,
        baf.latest_identity_created_on_account_asof,
        baf.user_creation_days_span_asof,
        baf.avg_users_created_per_day_asof,
        baf.avg_users_created_per_month_asof,

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

        /* Lightweight network / ring-style fields */
        nf.network_user_count_asof,
        nf.network_account_count_asof,
        nf.network_score_asof,
        nf.network_label,

        /* Channel / platform */
        ma.new_sub_channel_design_appsflyer_source AS acquisition_channel,
        usr.signup_source AS platform,
        usr.signup_ip,

        /* Heuristic fraud score for review / investigation */
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

    LEFT JOIN network_features nf
        ON a.advance_id = nf.advance_id
       AND a.bank_account_key = nf.bank_account_key

    LEFT JOIN brigit_snowflake.dbt_analytics.fct_marketing_attribution ma
        ON a.user_id = ma.user_id::VARCHAR

    LEFT JOIN brigit_snowflake.dbt_analytics.base_prod__users usr
        ON a.user_id = usr.user_id::VARCHAR
),

/* ---------------------------------------------------------------------
   12B. Add latest Socure/KYC row as of the advance timestamp.
   This replaces the unsupported LEFT JOIN LATERAL block.
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
   12C. Add latest client metadata row as of the advance timestamp.
   This replaces the unsupported LEFT JOIN LATERAL block.
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
   13. Final output.
   This is deduping, not sampling.
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
