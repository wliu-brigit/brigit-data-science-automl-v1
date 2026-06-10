-- The oot-split slice of the final synthetic scores (frozen snapshot, also
-- joined into the training base table). Used only for the RI-consistency QA:
-- diffing the new-links RI scores against the Phase-4 scores
-- (legacy oot_new_links_ri_scoring.ipynb, score-comparison cell).
SELECT
    user_id,
    synthetic_score AS final_score
FROM brigit_data_science.sandbox_hyong.neobank_ncm_v3_synthetic_scores_final
WHERE split = 'oot'
