-- The SELECT that pulls training rows from the base table.
-- SPLIT_PCT flows through; keep it in the projection (SELECT t.* keeps it).
--
-- Label: is_fraud = 1 when the upstream heuristic band is EXTREMELY_LIKELY.
-- This is a heuristic proxy label until investigator-confirmed fraud labels
-- are available to join in.
--
-- Downsample (deterministic, not SAMPLE()): fixed 98/2 composition — every
-- non-LOW row is kept untouched (no upsampling), and LOW is hash-sampled on
-- advance_id at 875/10000 so it lands at ~98% of the pull. Band counts
-- measured 2026-06-05: LOW 10,647,985 · POSSIBLE 14,765 · LIKELY 938 ·
-- EXTREMELY_LIKELY 3,123. Expected pull ≈ 932k LOW + all 18.8k non-LOW
-- ≈ 950k rows; is_fraud prevalence ≈ 0.33%. (Doubled from the original 99/1
-- so the ~100k dry-run sample carries ~2x the positives per split.)
SELECT
    t.*,
    IFF(t.heuristic_fraud_band = 'EXTREMELY_LIKELY', 1, 0) AS is_fraud
FROM {database}.{schema}.{base_table} t
WHERE t.heuristic_fraud_band <> 'LOW'
   OR MOD(ABS(HASH(t.advance_id)), 10000) < 875
