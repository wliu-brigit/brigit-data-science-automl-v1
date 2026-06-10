-- The final RI model's scores over the Jan–Feb 2026 new-links population —
-- frozen legacy snapshot (written by oot_new_links_ri_scoring.ipynb;
-- 139,916 rows: user_id, entity_id, is_known, went_dpd45, synthetic_score).
-- Consumed as-is: the RI model itself is never re-run here.
SELECT
    user_id,
    entity_id,
    is_known,
    went_dpd45,
    synthetic_score
FROM brigit_data_science.sandbox_hyong.neobank_ncm_v3_oot_new_links_ri_scores
