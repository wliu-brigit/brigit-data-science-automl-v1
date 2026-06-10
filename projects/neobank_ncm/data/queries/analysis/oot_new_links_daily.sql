-- Daily D1–D30 LSA snapshots for the OOT new-links population — the frozen
-- legacy snapshot (created by the legacy run; provenance:
-- data/queries/upstream_neobank_ncm_v3_oot_new_links_daily.sql, plaid
-- VARIANTs unwrapped by data-science/models/ltv/util_unwrap_plaid.ipynb).
-- Read-only: this analysis never writes warehouse state.
--
-- {columns} is filled by analysis.data.load_daily() with the locked feature
-- list plus metadata/outcome columns (legacy financial notebook cell 6) to
-- bound the pulled width; pass columns="*" to pull everything.
SELECT
    {columns}
FROM brigit_data_science.sandbox_hyong.neobank_ncm_v3_oot_new_links_daily_plaid_unnested
