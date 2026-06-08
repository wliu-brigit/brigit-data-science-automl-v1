-- Quick heuristic-band comparison, month over month.
--
-- Companion to the scenario backtest (monthly_backtest.py). Reads the EXISTING
-- materialized snapshot table -- no rebuild, single scan, runs in seconds.
-- The snapshot spans Dec 2025 -> the build date (~10.7M advances), so "by
-- month" yields several months.
--
-- The heuristic band (LOW / POSSIBLE / LIKELY / EXTREMELY_LIKELY) is the
-- production rule-of-thumb score already in the snapshot. We do NOT recompute
-- it in the scenario backtest because it depends on the lifetime / network
-- features the backtest drops for speed -- hence this separate read.
--
-- Outcome metrics mirror the scenario output for apples-to-apples comparison.
-- IMPORTANT (validated on this table): charge_off is essentially never
-- populated here (399 of 10.7M rows), so "loss" is NOT charge-off. The bad
-- outcome is delinquency: never_paid = matured AND gross-DPD45 AND not repaid.
--   dpd45_rate       n_dpd45 / n_matured                     (matured only)
--   never_paid_rate  never_paid / (repaid + never_paid)  -- RESOLVED bad-rate:
--                    of advances that reached a verdict, the share that went
--                    bad. "lower is better", same direction as dpd45_rate;
--                    still-open advances excluded. Fraud -> ~1.
--   loss_disbursed   $ disbursed to advances that never paid (loss exposure)

SELECT
    DATE_TRUNC('month', feature_as_of_ts)                    AS advance_month,
    heuristic_fraud_band                                     AS band,

    COUNT(*)                                                 AS n_advances,
    SUM(loan_amount)                                         AS total_loan_disbursed,

    COUNT_IF(label_mature_d45 = 1)                           AS n_matured,
    COUNT_IF(label_mature_d45 = 1 AND label_gross_dpd45 = 1) AS n_dpd45,
    COUNT_IF(label_mature_d45 = 1 AND label_gross_dpd45 = 1)
        / NULLIF(COUNT_IF(label_mature_d45 = 1), 0)          AS dpd45_rate,

    -- never_paid = matured & DPD45 & not repaid (the bad, resolved-unfavourably)
    COUNT_IF(label_mature_d45 = 1 AND label_gross_dpd45 = 1
             AND label_repaid_current_snapshot = 0)          AS n_never_paid,
    -- resolved bad-rate: never_paid / (repaid + never_paid)  -- lower is better
    COUNT_IF(label_mature_d45 = 1 AND label_gross_dpd45 = 1
             AND label_repaid_current_snapshot = 0)
        / NULLIF(COUNT_IF(label_repaid_current_snapshot = 1)
                 + COUNT_IF(label_mature_d45 = 1 AND label_gross_dpd45 = 1
                            AND label_repaid_current_snapshot = 0), 0) AS never_paid_rate,
    -- loss exposure: principal disbursed to advances that never paid
    SUM(IFF(label_mature_d45 = 1 AND label_gross_dpd45 = 1
            AND label_repaid_current_snapshot = 0, loan_amount, 0)) AS loss_disbursed

FROM brigit_data_science.SANDBOX_WLIU.fraud_advance_feature_base
GROUP BY 1, 2
ORDER BY
    1,
    CASE band
        WHEN 'EXTREMELY_LIKELY' THEN 1
        WHEN 'LIKELY'           THEN 2
        WHEN 'POSSIBLE'         THEN 3
        WHEN 'LOW'              THEN 4
        ELSE 5
    END;
