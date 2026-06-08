-- Quick-and-dirty heuristic-band comparison, month over month.
--
-- Companion to the scenario backtest (monthly_backtest.py). Reads the EXISTING
-- materialized snapshot table — no rebuild — so it is a single scan, no joins,
-- and runs in seconds. The snapshot was built for one output month (Dec 2025),
-- so "by month" here is effectively that month; re-point the FROM at a wider
-- materialization if/when one exists.
--
-- The heuristic band (LOW / POSSIBLE / LIKELY / EXTREMELY_LIKELY) is the
-- production rule-of-thumb score already computed in the snapshot. We do NOT
-- recompute it in the scenario backtest because it depends on the lifetime /
-- network features the backtest deliberately drops for speed — hence this
-- separate read against the pre-built table.
--
-- Metrics mirror the scenario output for apples-to-apples comparison:
--   dpd45_rate          n_dpd45 / n_matured           (matured only; the cut)
--   repaid_rate         repaid / (repaid + charged_off)  RESOLVED repayment —
--                       of advances that reached a verdict, the share paid back
--                       (fair to recent months, no maturity gate). Fraud -> ~0.
--   chargeoff_rate      charged_off / all advances    (raw loss rate)

SELECT
    DATE_TRUNC('month', feature_as_of_ts)            AS advance_month,
    heuristic_fraud_band                             AS band,

    COUNT(*)                                         AS n_advances,
    SUM(loan_amount)                                 AS total_loan_disbursed,

    COUNT_IF(label_mature_d45 = 1)                   AS n_matured,
    COUNT_IF(label_mature_d45 = 1 AND label_gross_dpd45 = 1) AS n_dpd45,
    COUNT_IF(label_mature_d45 = 1 AND label_gross_dpd45 = 1)
        / NULLIF(COUNT_IF(label_mature_d45 = 1), 0)  AS dpd45_rate,

    -- resolved repayment: repaid / (repaid + charged_off)
    COUNT_IF(label_repaid_current_snapshot = 1)
        / NULLIF(COUNT_IF(label_repaid_current_snapshot = 1
                          OR charge_off_timestamp IS NOT NULL), 0) AS repaid_rate,
    COUNT_IF(charge_off_timestamp IS NOT NULL)
        / NULLIF(COUNT(*), 0)                        AS chargeoff_rate

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

-- One-off: inspect how repayment/loss is encoded, to sanity-check the
-- definition above (run separately if you want to see the raw status split):
--
--   SELECT loan_status,
--          COUNT(*) AS n,
--          COUNT_IF(charge_off_timestamp IS NOT NULL) AS charged_off
--   FROM brigit_data_science.SANDBOX_WLIU.fraud_advance_feature_base
--   GROUP BY 1 ORDER BY 2 DESC;
