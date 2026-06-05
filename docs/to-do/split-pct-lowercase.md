# Optional: lowercase SPLIT_PCT so "all columns are lowercase" has no exception

**Status:** to-do, optional (spun off the fraud-pilot dossier, 2026-06-05)

The pipeline lowercases every user column but preserves harness-injected
`SPLIT_PCT` uppercase (it's pipeline state, named by `SPLIT_PCT_COL`). With
case-insensitive predicate resolution landed, the exception is cosmetic — but
"one rule, zero exceptions" reads better, and the rename is cheap now
(forward-only posture, no external users) and expensive after rollout.

Scope if picked up: `SPLIT_PCT_COL` constant, pipeline special-casing,
scaffold/config templates and comments, example projects' splits, docs, and
the matching contract/unit tests in the same change.

Related low-priority note (no action planned): CSV/parquet sources truncate
dry-run samples to the *first N file rows* — fine unless a file's row order
correlates with something meaningful; production runs are unaffected.
