# Leaderboard entries pinned to their dataset

**Status:** to-do (split out of the fraud-pilot feedback dossier, 2026-06-05)

**Symptom (hit live, twice):** the session ledger and proposer context
compared `average_precision` across trials run on *different data snapshots*
(0.96 at 8% prevalence vs 0.14 at 0.14% prevalence after a sampling change)
and recorded the comparison as a takeaway. AP is not comparable across
prevalences; the design already says comparability is pinned at the
experiment's data snapshot — the context packet just doesn't enforce it.

**Fix sketch:**

- Stamp each leaderboard/context entry with its `dataset_id` (the trial run
  already knows it; it needs to reach the rendered MLflow context summaries).
- In the proposer/manager context rendering, group or annotate by dataset id;
  one line in the prompts: compare only trials sharing the active dataset id,
  treat others as historical.

**Where:** MLflow context summaries / `automl/agent` context rendering; the
proposer prompt in `agent-skills`.
