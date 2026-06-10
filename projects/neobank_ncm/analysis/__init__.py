"""Post-training analysis: the legacy v3 downstream computations, ported.

Replicates what the legacy ran after training (legacy home:
data-science/models/underwriting/neobank/new_user/v3.0):

- ``data``    — loaders for the frozen new-links snapshots + the live LTV pull
- ``scoring`` — daily-grain feature engineering + chunked model scoring
- ``policy``  — KO rules, user-level collapse, threshold search, scenarios
- ``impact``  — LTV imputation, revenue decomposition, sample-size estimation

Everything here is analysis-layer only: effective-bad and every other
synthetic-score-derived number stays out of training, the leaderboard, and
the oot read (PROJECT_INSTRUCTIONS.md hard constraints).
"""
