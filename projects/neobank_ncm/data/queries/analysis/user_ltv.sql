-- Per-user revenue / LTV aggregation for the financial impact analysis —
-- verbatim from the legacy financial_impact_analysis.ipynb cell 18.
--
-- THE ONE LIVE PULL in this analysis: FCT_DAILY_USER_LTV and
-- LAST_UNDERWRITING_REPORT_FIRST_DAY are production dbt tables, read at
-- analysis time (the sample_users CTE anchors them to the frozen daily
-- snapshot's population). Values reflect the warehouse on the day this runs.
WITH sample_users AS (
    SELECT DISTINCT nl.user_id, uw.loan_amount_max::FLOAT AS loan_amount_max, uw.underwriting_strategy
    FROM brigit_data_science.sandbox_hyong.neobank_ncm_v3_oot_new_links_daily nl
    JOIN brigit_snowflake.dbt_analytics.LAST_UNDERWRITING_REPORT_FIRST_DAY uw
        ON uw.user_id = nl.user_id
)
SELECT
    ltv.user_id,
    su.loan_amount_max,
    su.underwriting_strategy,
    ltv.first_activation_date,
    date_trunc(month, first_linked_date)                                        AS ltv_cohort,
    MAX(ltv.days_since_linked)                                                  AS max_linked_days,
    max_linked_days >= 30                                                       AS ltv_30_elig,
    max_linked_days >= 60                                                       AS ltv_60_elig,
    max_linked_days >= 90                                                       AS ltv_90_elig,
    max_linked_days >= 120                                                      AS ltv_120_elig,
    SUM(IFF(ltv.days_since_linked <= 30, ltv.user_day_ltv_lite, 0))             AS total_ltv_lite_30,
    SUM(IFF(ltv.days_since_linked <= 60, ltv.user_day_ltv_lite, 0))             AS total_ltv_lite_60,
    SUM(IFF(ltv.days_since_linked <= 90, ltv.user_day_ltv_lite, 0))             AS total_ltv_lite_90,
    SUM(IFF(ltv.days_since_linked <= 120, ltv.user_day_ltv_lite, 0))             AS total_ltv_lite_120,
    SUM(IFF(ltv.days_since_linked <= 30, ltv.user_day_transfer_fee_revenue, 0))
        + SUM(IFF(ltv.days_since_linked <= 30, ltv.user_day_subscription_revenue, 0)) AS total_revenue_30,
    SUM(IFF(ltv.days_since_linked <= 60, ltv.user_day_transfer_fee_revenue, 0))
        + SUM(IFF(ltv.days_since_linked <= 60, ltv.user_day_subscription_revenue, 0)) AS total_revenue_60,
    SUM(IFF(ltv.days_since_linked <= 90, ltv.user_day_transfer_fee_revenue, 0))
        + SUM(IFF(ltv.days_since_linked <= 90, ltv.user_day_subscription_revenue, 0)) AS total_revenue_90,
    SUM(IFF(ltv.days_since_linked <= 120, ltv.user_day_transfer_fee_revenue, 0))
        + SUM(IFF(ltv.days_since_linked <= 120, ltv.user_day_subscription_revenue, 0)) AS total_revenue_120
FROM BRIGIT_SNOWFLAKE.DBT_ANALYTICS.FCT_DAILY_USER_LTV AS ltv
JOIN sample_users AS su ON ltv.user_id = su.user_id
WHERE ltv.days_since_linked >= 0
GROUP BY ltv.user_id, su.loan_amount_max, su.underwriting_strategy, ltv.first_activation_date, ltv_cohort
