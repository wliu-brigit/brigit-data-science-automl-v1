-- Layer 1: master entity list with one row per booked loan (known) or theoretical loan (unknown).
-- Creates: brigit_data_science.sandbox_hyong.neobank_ncm_v3_spine
--
-- Run this first. risk_features.sql and balance_snapshots.sql both join to this table.
-- OOT split is defined here — do not redefine downstream.
--
-- Covers origination Jan 2025 – Feb 2026 (train: 2025, oot: Jan–Feb 2026).
--
-- Unknown group approach:
--   Anchor on LAST_UNDERWRITING_REPORT_FIRST_DAY (one row per user).
--   Features come from uw.score_attributes_id — exact SA snapshot at assessment time.
--   Link to the first theoretical loan where valid_from is within 24h after report_time.

CREATE OR REPLACE TABLE brigit_data_science.sandbox_hyong.neobank_ncm_v3_spine AS (

WITH known_loans AS (
    -- Booked NCM neobank loans with mature outcomes.
    -- Neobank filter uses underwriting_strategy at origination time (not today's prod list).
    SELECT
        l.loan_id::STRING                                                   AS entity_id,
        l.user_id,
        sa.id                                                               AS sa_id,
        'LOAN'                                                              AS entity_type,
        DATE(l.origination_timestamp)                                       AS origination_date,
        l.original_due_date,
        l.amount,
        sa.bankinstitution::STRING                                          AS bankinstitution,
        TRUE                                                                AS is_known,
        CASE
            WHEN DATE(l.origination_timestamp) >= '2026-01-01' THEN 'oot'
            ELSE 'train'
        END                                                                 AS split,
        CASE
            WHEN DATEDIFF('day', l.original_due_date,
                          COALESCE(l.removed_timestamp::DATE, CURRENT_DATE)) > 45
            THEN 1 ELSE 0
        END                                                                 AS went_dpd45,
        NULL::TIMESTAMP_NTZ                                                 AS valid_from,
        NULL::TIMESTAMP_NTZ                                                 AS valid_to
    FROM brigit_snowflake.dbt_analytics.base_prod__loans l
    JOIN brigit_snowflake.dbt_analytics.sa_flatten_wide_incremental sa
        ON sa.object_id = l.loan_id
        AND sa.object_type = 'LOAN'
    WHERE sa.brigitloansrepaid = 0                                          -- NCM filter
      AND (l.underwriting_strategy ILIKE '%CHIME_VARO%'
           OR l.underwriting_strategy ILIKE '%NEOBANK%')                    -- neobank filter at origination time
      AND l.status != 'UNDELIVERABLE'
      AND DATE(l.origination_timestamp) BETWEEN '2025-01-01' AND '2026-02-28'
      AND l.original_due_date + INTERVAL '45 days' < CURRENT_DATE          -- maturity filter
),

linked_uw_ncm AS (
    -- Linked neobank NCM users who triggered a UW within the study window.
    -- LAST_UNDERWRITING_REPORT_FIRST_DAY has one row per user.
    -- Join SA to enforce NCM filter and pull bankinstitution for the spine.
    -- No user deletion filter — UW event is point-in-time; subsequent account deletion is irrelevant.
    SELECT
        uw.user_id,
        uw.score_attributes_id,
        uw.report_time,
        uw.loan_amount_max,
        sa.bankinstitution::STRING                                          AS bankinstitution
    FROM brigit_snowflake.dbt_analytics.LAST_UNDERWRITING_REPORT_FIRST_DAY uw
    JOIN brigit_snowflake.dbt_analytics.sa_flatten_wide_incremental sa
        ON sa.id = uw.score_attributes_id
    WHERE uw.report_date BETWEEN '2025-01-01' AND '2026-02-28'
      AND (uw.underwriting_strategy ILIKE '%CHIME_VARO%'
           OR uw.underwriting_strategy ILIKE '%NEOBANK%')                   -- neobank filter at UW time    
      AND sa.brigitloansrepaid = 0                                          -- NCM filter
      AND uw.user_id NOT IN (
              SELECT DISTINCT user_id
              FROM brigit_snowflake.dbt_analytics.base_prod__loans
              WHERE status != 'UNDELIVERABLE'
          )
),

first_theoretical_loans AS (
    -- First theoretical loan per user where valid_from falls within 24h after report_time.
    -- Anchors the entity's theoretical loan window to the actual UW assessment event.
    SELECT
        uw.user_id,
        uw.score_attributes_id,
        uw.report_time,
        uw.loan_amount_max,
        uw.bankinstitution,
        tl.theoretical_loan_id                                             AS entity_id,
        tl.due_date                                                        AS original_due_date,
        tl.valid_from,
        tl.valid_to
    FROM linked_uw_ncm uw
    JOIN brigit_snowflake.dbt_analytics.base_prod__theoretical_loan tl
        ON tl.user_id = uw.user_id
        AND tl.valid_from >= uw.report_time
        AND tl.valid_from < uw.report_time + INTERVAL '24 hours'
    QUALIFY ROW_NUMBER() OVER (PARTITION BY uw.user_id ORDER BY tl.valid_from ASC) = 1
),

unknown_loans AS (
    SELECT
        ftl.entity_id,
        ftl.user_id,
        ftl.score_attributes_id                                            AS sa_id,
        'THEORETICAL_LOAN'                                                 AS entity_type,
        DATE(ftl.report_time)                                              AS origination_date,
        ftl.original_due_date,
        ftl.loan_amount_max::FLOAT                                         AS amount,
        ftl.bankinstitution,
        FALSE                                                              AS is_known,
        CASE
            WHEN DATE(ftl.report_time) >= '2026-01-01' THEN 'oot'
            ELSE 'train'
        END                                                                AS split,
        NULL::INT                                                          AS went_dpd45,
        ftl.valid_from,
        ftl.valid_to
    FROM first_theoretical_loans ftl
)

SELECT * FROM known_loans
UNION ALL
SELECT * FROM unknown_loans

);

-- Sanity check: run after CREATE to verify population and bad rates.
SELECT
    is_known,
    split,
    COUNT(*)                           AS n,
    SUM(went_dpd45)                    AS n_bad,
    ROUND(AVG(went_dpd45) * 100, 1)    AS bad_rate_pct
FROM brigit_data_science.sandbox_hyong.neobank_ncm_v3_spine
GROUP BY 1, 2
ORDER BY 1, 2;
