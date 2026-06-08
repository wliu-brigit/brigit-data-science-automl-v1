-- Heuristic-band month-over-month comparison, built to line up COLUMN-FOR-COLUMN
-- with the scenario backtest (monthly_backtest.py) so the two are directly
-- comparable: here a `band` plays the role a `scenario` plays there, and the
-- baseline_* columns are the month-wide figures (identical meaning to the
-- backtest's baselines). Reads the EXISTING materialized snapshot table -- no
-- rebuild, single scan. The snapshot spans Dec 2025 -> the build date (~10.7M
-- advances), so "by month" yields several months.
--
-- The heuristic band (LOW / POSSIBLE / LIKELY / EXTREMELY_LIKELY) is the
-- production rule-of-thumb score already in the snapshot. We do NOT recompute
-- it in the scenario backtest because it depends on the lifetime / network
-- features the backtest drops for speed -- hence this separate read.
--
-- Definitions (validated on this table; charge_off is ~never populated -- 399
-- of 10.7M rows -- so "loss" is delinquency, not write-off):
--   never_paid          = matured AND gross-DPD45 AND not repaid
--   dpd45_rate          n_dpd45 / n_matured                  (matured only)
--   never_paid_rate     never_paid / (repaid + never_paid)   RESOLVED bad-rate,
--                       lower-is-better like dpd45_rate; still-open excluded.
--   never_paid_principal  $ loan_amount disbursed to never-paid advances
--                       (principal out the door, NOT a net-of-payments balance).

WITH by_band AS (
    SELECT
        DATE_TRUNC('month', feature_as_of_ts)                    AS advance_month,
        heuristic_fraud_band                                     AS band,
        COUNT(*)                                                 AS n_band,
        SUM(loan_amount)                                         AS band_loan_disbursed,
        COUNT_IF(label_mature_d45 = 1)                           AS n_matured,
        COUNT_IF(label_mature_d45 = 1 AND label_gross_dpd45 = 1) AS n_dpd45,
        COUNT_IF(label_mature_d45 = 1 AND label_gross_dpd45 = 1
                 AND label_repaid_current_snapshot = 0)          AS n_never_paid,
        COUNT_IF(label_repaid_current_snapshot = 1
                 OR (label_mature_d45 = 1 AND label_gross_dpd45 = 1)) AS n_resolved,
        SUM(IFF(label_mature_d45 = 1 AND label_gross_dpd45 = 1
                AND label_repaid_current_snapshot = 0, loan_amount, 0)) AS never_paid_principal
    FROM brigit_data_science.SANDBOX_WLIU.fraud_advance_feature_base
    GROUP BY 1, 2
)
SELECT
    advance_month,
    band,
    -- month-wide denominator (same for every band in the month)
    SUM(n_band) OVER (PARTITION BY advance_month)            AS n_advances,
    n_band,
    n_band / NULLIF(SUM(n_band) OVER (PARTITION BY advance_month), 0) AS band_rate,

    SUM(band_loan_disbursed) OVER (PARTITION BY advance_month) AS total_loan_disbursed,
    band_loan_disbursed,

    n_matured,
    n_dpd45,
    n_dpd45 / NULLIF(n_matured, 0)                          AS dpd45_rate,
    SUM(n_dpd45)   OVER (PARTITION BY advance_month)
        / NULLIF(SUM(n_matured) OVER (PARTITION BY advance_month), 0) AS baseline_dpd45_rate,

    n_never_paid / NULLIF(n_resolved, 0)                    AS never_paid_rate,
    SUM(n_never_paid) OVER (PARTITION BY advance_month)
        / NULLIF(SUM(n_resolved) OVER (PARTITION BY advance_month), 0) AS baseline_never_paid_rate,

    never_paid_principal,
    SUM(never_paid_principal) OVER (PARTITION BY advance_month) AS baseline_never_paid_principal

FROM by_band
ORDER BY
    advance_month,
    CASE band
        WHEN 'EXTREMELY_LIKELY' THEN 1
        WHEN 'LIKELY'           THEN 2
        WHEN 'POSSIBLE'         THEN 3
        WHEN 'LOW'              THEN 4
        ELSE 5
    END;
