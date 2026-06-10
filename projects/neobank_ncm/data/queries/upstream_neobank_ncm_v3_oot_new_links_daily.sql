-- OOT new links: D1–D30 daily LSA snapshots for financial impact analysis.
-- Creates: brigit_data_science.sandbox_hyong.neobank_ncm_v3_oot_new_links_daily
--
-- Population: Jan–Feb 2026 neobank NCM users (both known and unknown).
-- For each user, pulls 30 daily LSA snapshots starting from first uw report date.
-- Plaid features are queried as VARIANT columns from LSA — unwrap JSON via util query.
--
-- Outcomes:
--   Known:   went_dpd45 from first_loans (mature + first NCM neobank loan only; NULL if not yet mature)
--   Unknown: synthetic_score from neobank_ncm_v3_synthetic_scores_final
--
-- V1, V2 score are pulled latest SA on and before LSA run_date.

CREATE OR REPLACE TABLE brigit_data_science.sandbox_hyong.neobank_ncm_v3_oot_new_links_daily AS (

WITH new_links AS (
    SELECT
        uw.user_id,
        uw.report_time,
        uw.report_date,
        uw.loan_amount_max,
        uw.account_approval_state
    FROM brigit_snowflake.dbt_analytics.LAST_UNDERWRITING_REPORT_FIRST_DAY uw
    WHERE uw.report_date BETWEEN '2026-01-01' AND '2026-02-28'
      AND (uw.underwriting_strategy ILIKE '%CHIME_VARO%'
           OR uw.underwriting_strategy ILIKE '%NEOBANK%')
      AND uw.brigitloansrepaid = 0
),

first_loans AS (
    SELECT
        l.user_id,
        CASE
            WHEN DATEDIFF('day', l.original_due_date,
                          COALESCE(l.removed_timestamp::DATE, CURRENT_DATE)) > 45
            THEN 1 ELSE 0
        END                                                                 AS went_dpd45
    FROM brigit_snowflake.dbt_analytics.base_prod__loans l
    JOIN new_links nl
        ON nl.user_id = l.user_id
    WHERE l.status != 'UNDELIVERABLE'
      AND l.original_due_date + INTERVAL '45 days' < CURRENT_DATE
    QUALIFY ROW_NUMBER() OVER (PARTITION BY l.user_id
                               ORDER BY l.origination_timestamp ASC) = 1
),

sa_with_scores AS (
    SELECT
        sa.user_id,
        sa.creation_timestamp,
        MAX(CASE WHEN ms.model_name = 'NEOBANK_NCM_XGBOOST_MODEL_V2'
                 THEN ms.model_score::FLOAT END)                            AS v2_score,
        MAX(CASE WHEN ms.model_name = 'NEO_BANK_XGBOOST_MODEL_V1'
                 THEN ms.model_score::FLOAT END)                            AS v1_score
    FROM new_links nl
    JOIN brigit_snowflake.dbt_analytics.sa_flatten_wide_incremental sa
        ON sa.user_id = nl.user_id
        AND sa.object_type IN ('LOAN', 'UNDERWRITING_REPORT')
        AND sa.creation_timestamp < nl.report_date + INTERVAL '30 days'
    JOIN brigit_data_science.model_scores.underwriting_model_scores ms
        ON ms.sa_id = sa.id
        AND ms.model_name IN ('NEOBANK_NCM_XGBOOST_MODEL_V2', 'NEO_BANK_XGBOOST_MODEL_V1')
    GROUP BY sa.user_id, sa.creation_timestamp
    HAVING v1_score IS NOT NULL AND v2_score IS NOT NULL
),

synthetic_scores AS (
    SELECT user_id, synthetic_score
    FROM brigit_data_science.sandbox_hyong.NEOBANK_NCM_V3_OOT_NEW_LINKS_RI_SCORES
)

SELECT
    nl.user_id,
    nl.report_date                                                         AS first_report_date,
    nl.loan_amount_max,
    nl.account_approval_state,
    lsa.run_date                                                           AS snapshot_date,
    DATEDIFF('day', nl.report_date, lsa.run_date) + 1                      AS day_number,

    fl.went_dpd45,
    syn.synthetic_score,

    sws.v1_score,
    sws.v2_score,

    lsa.bankinstitution::VARCHAR                                            AS bankinstitution,

    lsa.highestpayfrequency::VARCHAR                                        AS highestpayfrequency,

    lsa.accthistorydays::INT                                                AS accthistorydays,
    lsa.availablebalance::FLOAT                                             AS availablebalance,
    lsa.balance::FLOAT                                                      AS balance,
    lsa.balanceafterlastpayday2::FLOAT                                      AS balanceafterlastpayday2,
    lsa.balancemean::FLOAT                                                  AS balancemean,
    lsa.balancemeanafterpayday0::FLOAT                                      AS balancemeanafterpayday0,
    lsa.balancemeanafterpayday1::FLOAT                                      AS balancemeanafterpayday1,
    lsa.balancesd::FLOAT                                                    AS balancesd,
    lsa.balanceslope::FLOAT                                                 AS balanceslope,
    lsa.balanceslopesterr::FLOAT                                            AS balanceslopesterr,
    lsa.balpredictionnday::FLOAT                                            AS balpredictionnday,
    lsa.creditorsummarycreditninetydayamount::FLOAT                         AS creditorsummarycreditninetydayamount,
    lsa.creditorsummarycreditsevendayamount::FLOAT                          AS creditorsummarycreditsevendayamount,
    lsa.creditorsummarycreditthirtydayamount::FLOAT                         AS creditorsummarycreditthirtydayamount,
    lsa.creditorsummarydebitninetydayamount::FLOAT                          AS creditorsummarydebitninetydayamount,
    lsa.creditorsummarydebitninetydaycount::INT                             AS creditorsummarydebitninetydaycount,
    lsa.creditorsummarydebitsevendayamount::FLOAT                           AS creditorsummarydebitsevendayamount,
    lsa.dailydebitcountsd::FLOAT                                            AS dailydebitcountsd,
    lsa.dailyincomemean::FLOAT                                              AS dailyincomemean,
    lsa.dailyincomeregularmean::FLOAT                                       AS dailyincomeregularmean,
    lsa.davesummarycreditninetydayamount::FLOAT                             AS davesummarycreditninetydayamount,
    lsa.davesummarycreditsevendayamount::FLOAT                              AS davesummarycreditsevendayamount,
    lsa.davesummarycreditsevendaycount::INT                                 AS davesummarycreditsevendaycount,
    lsa.davesummarycreditthirtydaycount::INT                                AS davesummarycreditthirtydaycount,
    lsa.davesummarydebitninetydaycount::INT                                 AS davesummarydebitninetydaycount,
    lsa.davesummarydebitsevendayamount::FLOAT                               AS davesummarydebitsevendayamount,
    lsa.davesummarydebitthirtydayamount::FLOAT                              AS davesummarydebitthirtydayamount,
    lsa.dayssinceregularpayday::INT                                         AS dayssinceregularpayday,
    lsa.daystoregularpayday::INT                                            AS daystoregularpayday,
    lsa.dayswithbrigit::INT                                                 AS dayswithbrigit,
    lsa.debitamountz::FLOAT                                                 AS debitamountz,
    lsa.debitcountz::FLOAT                                                  AS debitcountz,
    lsa.highestincometotalobservedpaydays::INT                              AS highestincometotalobservedpaydays,
    lsa.highestpaydepositvoladj::FLOAT                                      AS highestpaydepositvoladj,
    lsa.incomesourcescount::INT                                             AS incomesourcescount,
    lsa.individualcreditamountmean::FLOAT                                   AS individualcreditamountmean,
    lsa.individualdebitamountsd::FLOAT                                      AS individualdebitamountsd,
    lsa.inflowsum14d::FLOAT                                                 AS inflowsum14d,
    lsa.lowest3monthbalancehighwatermark::FLOAT                             AS lowest3monthbalancehighwatermark,
    lsa.maxendofdaybalance14d::FLOAT                                        AS maxendofdaybalance14d,
    lsa.maxendofdaybalance28d::FLOAT                                        AS maxendofdaybalance28d,
    lsa.maxendofdaybalance7d::FLOAT                                         AS maxendofdaybalance7d,
    lsa.monthinflowsd::FLOAT                                                AS monthinflowsd,
    lsa.negbalancerate::FLOAT                                               AS negbalancerate,
    lsa.negbalepisodelengthmean::FLOAT                                      AS negbalepisodelengthmean,
    lsa.negbalepisodelengthsd::FLOAT                                        AS negbalepisodelengthsd,
    lsa.noactivityrate::FLOAT                                               AS noactivityrate,
    lsa.noactivityrate30d::FLOAT                                            AS noactivityrate30d,
    lsa.noactivityrate7d::FLOAT                                             AS noactivityrate7d,
    lsa.numofdepositoryaccounts::INT                                        AS numofdepositoryaccounts,
    lsa.othercompetitorsummarycreditninetydayamount::FLOAT                  AS othercompetitorsummarycreditninetydayamount,
    lsa.othercompetitorsummarycreditninetydaycount::INT                     AS othercompetitorsummarycreditninetydaycount,
    lsa.probzerobalancetopayday::FLOAT                                      AS probzerobalancetopayday,
    lsa.recurrentamountsum::FLOAT                                           AS recurrentamountsum,
    lsa.recurrentcount::INT                                                 AS recurrentcount,
    lsa.recurringrate::FLOAT                                                AS recurringrate,

    lsa.daystopayday::INT                                                   AS daystopayday,
    lsa.earninsummarycreditninetydayamount::FLOAT                           AS earninsummarycreditninetydayamount,
    lsa.highestpaydepositmean::FLOAT                                        AS highestpaydepositmean,
    lsa.maxnegativebalpast30days::FLOAT                                     AS maxnegativebalpast30days,
    lsa.outflowsum14d::FLOAT                                                AS outflowsum14d,

    lsa.* ILIKE 'plaidfeaturessummary%'

FROM new_links nl
JOIN brigit_snowflake.dbt_analytics.lsa_flatten_wide_incremental lsa
    ON lsa.user_id = nl.user_id
    AND lsa.run_date BETWEEN nl.report_date
        AND nl.report_date + INTERVAL '29 days'
LEFT JOIN first_loans fl
    ON fl.user_id = nl.user_id
LEFT JOIN synthetic_scores syn
    ON syn.user_id = nl.user_id
LEFT JOIN sa_with_scores sws
    ON sws.user_id = nl.user_id
    AND sws.creation_timestamp <= lsa.run_timestamp
QUALIFY ROW_NUMBER() OVER (
    PARTITION BY nl.user_id, lsa.run_date
    ORDER BY sws.creation_timestamp DESC NULLS LAST
) = 1

);

-- Sanity check: run after CREATE to verify population, day coverage, and outcome fill.
SELECT
    COUNT(DISTINCT user_id)                                                 AS n_users,
    COUNT(*)                                                                AS n_rows,
    ROUND(COUNT(*) * 1.0 / COUNT(DISTINCT user_id), 1)                     AS avg_days_per_user,
    SUM(CASE WHEN went_dpd45 IS NOT NULL THEN 1 ELSE 0 END)                AS rows_with_outcome,
    SUM(CASE WHEN synthetic_score IS NOT NULL THEN 1 ELSE 0 END)           AS rows_with_synthetic
FROM brigit_data_science.sandbox_hyong.neobank_ncm_v3_oot_new_links_daily;