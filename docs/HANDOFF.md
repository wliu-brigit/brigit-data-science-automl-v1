# Handoff — continue here

Temporary hand-over note: where the last session left off and what's open, so a
new session can pick up cold. **Not a changelog** — keep only what's current and
relevant. Rewritten at each wrap, not appended to.

> **These docs are best-effort documentation. The code is the source of truth
> for current behavior.** Anything in `execution/` describes *intent and the
> design as discussed* — before building on a claim about how the system behaves
> today, confirm it against the code. The `design.md` files flag the spots most
> worth verifying.

**Last updated:** 2026-06-10 — branch `core/dataset-read-reliability`, cut clean
off `main` (`cc688b5`). This session **scoped two core reliability efforts and
captured their designs**; **no implementation yet**, by intent.

## Why this branch exists

Running `neobank_ncm` on full data kept failing on the data-read path (transient
GCS read failures) and on serving validation (timeout crash + a silent native
crash). On the `neobank_ncm_v3_replicate` branch those were patched with
**band-aids** (an outer whole-frame retry loop; a bytes-decode fix + configurable
timeout). We deliberately **did not port the band-aids** — the goal is to fix the
root causes holistically. This branch carries the **designs only**, off `main`,
so a focused session can finalize and implement them and merge to `main`.

## How to pick up

Don't dive into code first. (1) Read this; (2) read the two efforts under
[`docs/execution/`](execution/) — each has a `README.md` front door, a `design.md`
with the recommended approach + caveats, and the migrated findings as evidence;
(3) **finalize each design and write its `plans/`** (this branch is the
`writing-plans` stage, not implementation). Verify the "where main is today" and
"caveats" sections against the actual code before planning — they're best-effort.

## The two efforts (in `docs/execution/`)

1. **`dataset-read-reliability/`** — the big one, mostly net-new design. A
   content-addressed **local cache** at the read seam so the full multi-GB frame
   is fetched **once** per trial (today ~6–7×, incl. the serving-validation
   subprocess), with a **robust populate** (resumable + checksum + tuned retry)
   that **retires the band-aid retry**, plus a read-once contract-builder tweak,
   LRU-on-write retention, and an `automl data cache` CLI verb. Includes a small
   **companion**: move Snowflake `skip_live_check` from the source onto
   `RUN_CONFIG`.
2. **`serving-validation-robustness/`** — a tighter cluster on one seam
   (`automl/runner/serving_validation.py` + the loop manager): make a validation
   **timeout or native crash** always leave a visible failure report/tag (the
   SIGSEGV path is silent today), lift the known-good timeout fixes from the
   neobank branch, and settle the **open decision**: should a validation
   timeout/crash **fail-soft** (keep the trained model + eval metric, continue)
   **vs halt** the loop. That decision likely touches the agent/loop layer, not
   just the runner.

## What this session changed on this branch (docs only)

- Archived the **completed** `snowflake-source-and-split-keys` effort
  `execution/ → archive/2026-06-04-snowflake-source-and-split-keys/` (it was
  "IMPLEMENTED" — `execution/` should stay ~empty).
- Created the two `execution/` efforts above (READMEs + `design.md` + migrated
  findings).
- Dropped three loose notes into [`to-do/`](to-do/) so they reach `main` and
  aren't lost: `tiny_eval-retry-orphaned-gcs-artifact.md` (eval-write
  idempotency — different seam, same reliability theme),
  `tiny_snowflake-metadata-query-helper.md`, and
  `tiny_per-trial-seconds-not-enforced.md` (already decided: leave as-is, fix
  docs).

## Where `main` stands (context)

- `main` (`cc688b5`) has the naive read path: single-shot
  `download_as_bytes()`, manifest-hash verification, slice — **no** retry, **no**
  cache, **no** read-once; the contract builder and the serving-validation
  subprocess each re-read the full frame. `serving_validation.py` has the
  hardcoded 120s timeout and the bytes-crash + silent-SIGSEGV gaps. None of the
  neobank-branch patches are on `main`.
- The `neobank_ncm_v3_replicate` branch (separate worktree) is the runnable
  integration branch and still carries the band-aids; when this effort merges to
  `main`, that branch rebases and its redundant core diff drops out.

## Constraints to respect

- **No band-aid porting.** Re-derive fixes from the design; the only safe lift is
  the two known-good serving-validation fixes (`_decode_tail` + configurable
  timeout), noted in that effort's `design.md`.
- **Content-addressing is the safety property** for any cache — key on
  `dataset_id` + content hash, verify against the manifest, GCS stays the single
  source of truth.
- **Layering:** generic mechanism in `automl/utils/`, dataset-specific policy at
  the data read seam, hashing stays in `automl/data/contract.py`. CLI verbs thin.
