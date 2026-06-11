# Trial reliability — effort front door

**Start with [`design.md`](design.md).** This README states the *problem* and
*goals* only; every settled call lives in `design.md`. This effort merges the
former `dataset-read-reliability/` and `serving-validation-robustness/` efforts
into one design (ratified 2026-06-10) — the unifying theme is the **trial
runner's failure surface**: shrink it where we can, illuminate it where we
can't.

## The problem (high level)

On the full-data loop (`neobank_ncm`, 1.15M × 2,253), trials fail or go dark in
three stacked ways:

1. **The dataset read path is redundant and fragile.** Each trial downloads the
   same immutable multi-GB frame ~7× (each ~63–73s), every time via a
   single-shot `download_as_bytes()` with no resumable transfer or tuned retry.
   Rare per-read transients × 7 reads × many trials = regular failures, on and
   off VPN — and one transient kills the trial and halts the loop.
2. **Serving validation fails badly.** A subprocess timeout crashes the very
   handler meant to keep the trial alive (bytes in `json.dumps`), the 120s cap
   is boundary-tight on full data, and a signal-killed child produces only a
   generic report with an often-empty stderr tail.
3. **Infra failures are invisible.** Best-effort steps swallow exceptions
   silently (`_try_log_train_eval` has a bare `except: return`), and a native
   crash of the runner process itself leaves no MLflow record at all. We learn
   about these only by digging.

## What we're looking for (goals)

- **Read the bytes once per loop, not 7× per trial** — a content-addressed
  local cache means every read after the first is local disk; the one populate
  is robust (resumable, checksummed, tuned retry); correctness stays provable
  against the manifest; GCS remains the single source of truth.
- **A validation problem never crashes or silences a trial** — timeout and
  signal exits always leave a labeled failure report and tag.
- **A trial that finished can still tell you everything that went wrong
  inside it** — errors along the way are accumulated and published as a
  first-class record, not swallowed.
- **The trial stays fail-soft** for anything that runs to completion;
  `validation.status` remains the deployability signal. No new halt behavior.
- **Lay the foundation for, but do not build, the modular/retryable runner**
  — stable published records now, machinery later.

## Status

**Design ratified and plans written 2026-06-10** (working session with code
verification — the design corrects several claims the predecessor docs got
wrong; see `design.md` §"Corrections"). The three migrated findings in this
folder are the evidence base.

Implementation: execute [`plans/`](plans/) in numbered order — each is
independently landable:

1. `plan-1-serving-validation-hardening.md` — small, immediate value.
2. `plan-2-dataset-cache-and-robust-populate.md` — the big win.
3. `plan-3-read-once-contract-builder.md` — independent of plan 2; better after.
4. `plan-4-trial-context-and-issue-ledger.md` — widest diff; land last.
5. `plan-5-skip-snowflake-live-check.md` — companion, any time.
