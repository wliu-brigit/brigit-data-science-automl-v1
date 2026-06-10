# Dataset-read reliability — effort front door

**Start with [`design.md`](design.md).** This README is intentionally thin and
decision-free: it states the *problem* and *what we're looking for*. The *how*
— the cache design, the download strategy, the layering, every settled call —
lives in `design.md`.

## The problem (high level)

On the full-data loop, reading the materialized dataset from GCS is the single
biggest source of trial failures and wasted wall-clock — and it is **not** a
GCS-availability problem. Two things stack up:

1. **We read the same multi-GB frame ~6–7× per trial** — the fit load, the eval
   load, the train diagnostic, once per split in the contract builder, and once
   more in the serving-validation **subprocess**. Each read is a multi-GB
   download + parse (~63–73s on neobank's 1.15M × 2,253 frame).
2. **Each read is a single-shot whole-object download** (`blob.download_as_bytes()`
   in `automl/utils/io/gcs.py`) — one HTTP GET on one connection, no resumable
   transfer, no checksum enforcement, no tuned retry. A long-lived single
   connection moving GBs has a large interruption window; rare transients (a
   TLS reset, an oauth2 `/token` timeout, corrupted bytes) × ~7 reads/trial ×
   many trials = failures show up regularly, on **and** off VPN.

The current mitigation is a band-aid: an **outer retry loop** in
`automl/data/registry.py` (`_read_and_verify_dataset`, added 2026-06-10) that
re-reads the *entire* frame up to 3× on `DataError`/`StorageError`. It keeps the
loop alive but treats the symptom — a single failure still costs a full multi-GB
re-read, and the redundant reads remain.

## What we're looking for (goals — not solutions)

The bar the design should clear:

- **Read the bytes once per trial**, not ~7×. Repeat readers — including the
  serving-validation **subprocess**, which cannot share parent memory — get the
  already-fetched bytes, not a fresh download.
- **One robust fetch**, not many fragile ones: the failure surface shrinks from
  ~7 network reads to a single populate that uses the storage client's native
  robustness (resumable/chunked transfer, checksum validation, tuned retry).
- **The band-aid retry retires** — it should no longer be load-bearing.
- **Correctness stays provable**: the dataset is content-addressed (id + manifest
  hash, immutable), so any reuse/cache is verified against the manifest and GCS
  remains the single source of truth.
- **No unbounded local growth**: anything kept on disk is size-bounded and
  user-inspectable, with an explicit cleanup path.
- **Layering respected**: a generic mechanism low (utils), dataset-specific
  policy at the data read seam, no per-domain duplication.

## Companion change (in scope, small)

- **Snowflake `skip_live_check` → `RUN_CONFIG`.** Today the off-VPN
  live-probe bypass is a flag on `SnowflakeSource` (operational state bolted onto
  the source). Move it to a `RUN_CONFIG` run-knob, consistent with
  `serving_validation_seconds`. See `design.md` §"Companion".

## Status

**Design captured, not yet planned.** Promoted to `execution/` 2026-06-10 (user-
gated). Next session: read `design.md` + the migrated finding
([`finding-redundant-full-reads.md`](finding-redundant-full-reads.md)), finalize
the design, and write `plans/`. No implementation code has been ported from the
`neobank_ncm_v3_replicate` branch by design — the goal is a holistic fix, not the
band-aid.
