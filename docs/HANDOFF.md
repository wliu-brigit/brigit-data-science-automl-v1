# Handoff — continue here

Temporary hand-over note: where the last session left off and what's open, so a
new session can pick up cold. **Not a changelog** — keep only what's current and
relevant. Rewritten at each wrap, not appended to.

> **These docs are best-effort documentation. The code is the source of truth
> for current behavior.**

**Last updated:** 2026-06-11 — branch `core/dataset-read-reliability`. The
trial-reliability effort is **designed, implemented, reviewed, and live-tested**
in one pass; the implementation is up as a PR for remote review/merge.

## What this session did

1. **Merged the two reliability efforts into one**:
   `docs/execution/trial-reliability/` (design ratified with three corrections
   to the predecessor docs — read `design.md` §"Corrections"; the predecessor
   folders are gone, findings migrated).
2. **Implemented all five plans** (subagent-driven; each passed spec + quality
   review, fixes folded in): serving-validation hardening, content-addressed
   dataset cache + robust populate, read-once contract builder, TrialContext +
   issue ledger, `skip_snowflake_live_check`. See
   `docs/execution/trial-reliability/README.md` for the map.
3. **Fixed what verification surfaced**: a `faulthandler.enable()` crash under
   sys-captured stderr (found only by the live e2e), integration-tier cache
   isolation + a pre-existing env-dependent integration test, and three stale
   e2e gates (archive-rename delete semantics, MLflow-3 logged-model artifact
   tree, `automl.trial.create` module-vs-function import collision).
4. **Docs honesty**: `per_trial_seconds` is now documented everywhere as an
   advisory budget (decided 2026-06-10; nothing enforces it); only
   `serving_validation_seconds` is a real timeout. The stale to-do was deleted.

## Verification at wrap

- `tests/unit` + `tests/contracts`: **647 passed, 1 skipped**.
- `tests/integration`: **41 passed, 0 failed** (now hermetic).
- `tests/e2e` (live GCS + local MLflow, Snowflake test excluded, notebooks
  gated off): **7 passed** including the walking skeleton — the live loop ran
  materialize → cached reads → trial → eval against the real bucket.
- Measured: `dataframe_content_hash` ≈ 34s per full-scale neobank slice — with
  network reads gone, hashing dominates the contract step. Future optimization
  candidate, deliberately out of scope.

## How to pick up

1. **Review/merge the PR** (head `core/dataset-read-reliability-impl`, base
   `core/dataset-read-reliability` — wendao merges on the remote). Then take
   `core/dataset-read-reliability` → `main` when ready.
2. **After it reaches `main`**: move `docs/execution/trial-reliability/` to
   `docs/archive/` (it records its own completion), and **rebase
   `neobank_ncm_v3_replicate`** — its band-aids drop out: the
   `_read_and_verify_dataset` retry (cache replaces it), the `_decode_tail` +
   timeout patches (landed properly), and `SnowflakeSource(skip_live_check=True)`
   in `projects/neobank_ncm/config.py` (move to
   `RUN_CONFIG.skip_snowflake_live_check`).
3. QA hygiene: this session's e2e routes (plus two stale 2026-06-09 neobank
   dry-run routes) are **archived** under `deleted/qa/...`; permanent disposal
   is `automl mlflow purge --scope qa --apply` whenever convenient (local/admin
   context; consider `mlflow_local gc-auth` after, per machine notes).

## Open / parked

- `docs/to-do/runner-crash-supervision.md` — a natively-crashed runner still
  leaves no finalized MLflow record (consciously accepted boundary #3 of the
  design; ledger JSONL + stderr traceback are the evidence until a supervisor
  seam exists).
- Design §3 boundaries #1–#2 remain consciously accepted: the loop still halts
  on a hard-failed trial, and correctness-failed models stay FINISHED with
  `validation.status` as the deployability signal (audit of leaderboard
  consumers still worth doing).
- `docs/to-do/tiny_eval-retry-orphaned-gcs-artifact.md` (eval-write
  idempotency) and `multi-runner-architecture.md` (cache eviction concurrency)
  are untouched, as scoped.
