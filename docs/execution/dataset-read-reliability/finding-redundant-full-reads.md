# To-do (tiny): trial-data-contract re-reads the full dataset once per split

**Status:** noted 2026-06-10 while debugging the neobank_ncm full-data
`DataError` (content hash != manifest). Not a correctness bug — a cost/exposure
one. **Core change** (`automl/runner/trial.py`), flag before touching.

## What

`_trial_data_contract` (`automl/runner/trial.py:568`) builds a `SliceContract`
per split by calling `data.load_dataset_by_id(..., split_name=name)` for every
split that isn't the already-loaded fit split. Each call downloads and parses
the **entire** materialized parquet from GCS, then slices it — just to compute
the slice's `n_rows` + `content_hash`.

On neobank_ncm (full: 1,147,866 × 2,253) that parquet read is ~63s and multiple
GB each time. A single trial reads the full frame on: the train load, the eval
load, the (best-effort) train-eval diagnostic, **and** once per split here. So
the same multi-GB frame is read ~4–5×/trial.

## Why it matters

1. **Time:** `data_load` was 63s; the contract step pays it again per split.
2. **Failure exposure:** the full-data `DataError` fired *here* — a transient
   corrupted GCS read on the contract-builder's extra read, after training had
   already passed. Fewer full reads = smaller exposure surface. (A
   verification-based retry now self-heals that read in
   `automl/data/registry.py:_read_and_verify_dataset`; this to-do is about not
   doing the redundant reads at all.)

## Shape (suggested)

Load the full frame **once** per trial and derive every split's slice in memory
(apply each split predicate's mask to the already-loaded frame) instead of
re-reading per split. The fit split's frame is already in hand as `loaded_fit`;
the contract just needs each split's masked `n_rows` + `dataframe_content_hash`.
Reuse the predicate masks the splits already define.

## Don't forget

- Keep the per-slice `content_hash` semantics identical (hash the *sliced*
  frame, not the full one) so existing slice contracts/tag-lineage still verify.
- Relates to [[gcs-vpn-throughput]] and the retry added in
  `automl/data/registry.py` (2026-06-10).
- Delete this file once landed.
