-- Frozen snapshot for the neobank_ncm v3 replication: one row per entity
-- (booked loan = known, theoretical loan = unknown).
--
-- Sources are the legacy v3 snapshot tables, pinned read-only in
-- sandbox_hyong (DDL provenance: upstream_*.sql in this folder; full legacy
-- home: data-science/models/underwriting/neobank/new_user/v3.0):
--   neobank_ncm_v3_spine                    population, split, label
--   neobank_ncm_v3_risk_features            underwriting-time features (SA + Plaid)
--   neobank_ncm_v3_synthetic_scores_final   RI soft labels for unknown rows
--
-- Source tables are deliberately NOT {database}.{schema}-substituted: the
-- harness materializes its copy under the session schema (sandbox_wliu)
-- while the legacy sources stay pinned to sandbox_hyong.
--
-- The harness wraps this SELECT in CREATE OR REPLACE TABLE and injects
-- SPLIT_PCT from split_group_key — do not emit SPLIT_PCT here.
--
-- No sampling: all unknown rows are kept. The legacy 200K downsample
-- (QUALIFY ROW_NUMBER() OVER (ORDER BY HASH(entity_id)) <= 200000) is a
-- modeling step left to trial code, reproducible from this snapshot.
--
-- Derived features replicate add_derived_features() from the legacy
-- notebooks (row-wise, deterministic). NULLs propagate exactly like the
-- pandas NaN passthrough (GREATEST/ABS/division return NULL on NULL input;
-- payday features: missing = no detected pay cycle = signal, never imputed).
-- VPN-time check: if the refreshed SA spec already ships any of these
-- derived columns inside risk_features, the duplicate alias below will fail
-- the CREATE — drop the redundant expression then.

WITH joined AS (
    SELECT
        s.sa_id,
        s.entity_id,
        s.user_id,
        s.split,
        s.is_known,
        s.went_dpd45,
        -- TIMESTAMP_NTZ (not DATE): Arrow delivers DATE as object-dtyped
        -- datetime.date, which breaks split predicates comparing against
        -- '2025-11-01' strings; timestamps land as datetime64 and compare fine.
        s.origination_date::TIMESTAMP_NTZ AS origination_date,
        s.original_due_date,
        s.amount,
        s.valid_from,
        s.valid_to,
        f.* EXCLUDE (entity_id, user_id, split, is_known),
        syn.synthetic_score
    FROM brigit_data_science.sandbox_hyong.neobank_ncm_v3_spine s
    JOIN brigit_data_science.sandbox_hyong.neobank_ncm_v3_risk_features f
        ON f.entity_id = s.entity_id
    LEFT JOIN brigit_data_science.sandbox_hyong.neobank_ncm_v3_synthetic_scores_final syn
        ON syn.entity_id = s.entity_id
)

SELECT
    j.*,

    -- income-normalized ratios (eps-clipped denominators, as in the notebook)
    j.balancesd / GREATEST(ABS(j.dailyincomemean), 1e-6)
        AS balancesdtodailyincomemeanratio,
    ABS(j.maxnegativebalpast30days) / GREATEST(ABS(j.dailyincomemean), 1e-6)
        AS maxnegbalance30dtodailyincomemeanratio,
    -- candidates from the experiment phase that did not survive feature
    -- selection (legacy names: balance_to_income, odnsf_to_income,
    -- income_regularity) — kept so the candidate set matches the legacy EDA
    j.balancemean / GREATEST(ABS(j.dailyincomemean), 1e-6)
        AS balancemeantodailyincomemeanratio,
    j.odandnsffeesdaily / GREATEST(ABS(j.dailyincomemean), 1e-6)
        AS odandnsffeesdailytodailyincomemeanratio,
    ABS(j.dailyincomeregularmean) / GREATEST(ABS(j.dailyincomemean), 1e-6)
        AS dailyincomeregularmeantodailyincomemeanratio,
    ABS(j.inflowsum14d) / GREATEST(ABS(j.outflowsum14d), 1e-6)
        AS inflowsumtooutflowsumratio14d,
    (ABS(j.inflowsum14d) - ABS(j.outflowsum14d))
        / (GREATEST(ABS(j.dailyincomemean), 1e-6) * 14)
        AS netflowtodailyincomemeanratio14d,

    -- balance depletion over the first day after payday
    (j.balancemeanafterpayday1 - j.balancemeanafterpayday0)
        / GREATEST(ABS(j.highestpaydepositmean), 1e-6)
        AS balancedepletionrate1d,

    -- balance-to-income buffer per day until next payday (NULL daystopayday → NULL)
    (j.balancemean / GREATEST(ABS(j.dailyincomemean), 1e-6))
        / GREATEST(j.daystopayday, 1)
        AS incomebuffertodaystopaydayratio,

    -- competitor borrowing intensity over 90 days
    (COALESCE(j.davesummarycreditninetydayamount, 0)
     + COALESCE(j.earninsummarycreditninetydayamount, 0)
     + COALESCE(j.othercompetitorsummarycreditninetydayamount, 0))
        / (GREATEST(ABS(j.dailyincomemean), 1e-6) * 90)
        AS competitorborrowintensity,

    -- tax season flag (Feb–Apr origination)
    CASE WHEN MONTH(j.origination_date) IN (2, 3, 4) THEN 1 ELSE 0 END
        AS istaxseason

FROM joined j
