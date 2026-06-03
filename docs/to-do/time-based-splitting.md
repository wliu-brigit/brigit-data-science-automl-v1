# Time-based splitting

**Status: placeholder (2026-06-03).** Not scoped — needs a deep-dive session
before any design. This note captures the ask and where the current code
stands so that session can start oriented.

## The ask

Users want a **time split** alongside the existing deterministic random
split. Their concern: most of our models are built *now* to predict a
*particular future*, so an evaluation that ignores the time component can
overstate performance — honest evaluation wants train/test separated across
time, not just across rows.

## Current state

Splitting today is deterministic and row-identity based:

- `automl/data/split.py` — `add_split_id(...)` hashes the declared
  `hash_key` columns and buckets the hash into a stable `split_id`, so the
  same entity always lands in the same bucket across materializations.
- The split view exposes that `split_id`; eval datasets select buckets from
  it (`EvalDataset.split_view(...)` in `automl/eval/eval_dataset.py`).

There is no notion of time anywhere in the split path.

## Sketch (to pressure-test in the deep dive)

Likely shape: the pipeline produces (or the recipe declares) a **time
column**, and the split view carries it so a split can be expressed as a
condition on it — e.g. train ≤ T, evaluate > T — composing with, not
replacing, the hash-based `split_id`. Open questions for the session:

- Where does the time column come from — declared in the recipe
  (`DataSpec`), derived by the pipeline, or both?
- Is a time split a property of the dataset (materialized once, like
  `split_id`) or of the eval selection (an argument at eval-dataset
  creation, like buckets)?
- How do time splits and hash splits compose (e.g. time-bounded train set
  with a hash-based holdout inside it)?
- Leakage rules: does anything need to enforce that features are computed
  only from data before the split time?

Related: [out-of-sample eval & dataset management](out-of-sample-eval-and-dataset-management.md)
— a time-based holdout is one kind of out-of-sample evaluation.
