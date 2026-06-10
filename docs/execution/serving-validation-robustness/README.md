# Serving-validation robustness — effort front door

**Start with [`design.md`](design.md).** This README states the *problem* and
*goals* only; the *how* and the open decision live in `design.md`.

## The problem (high level)

The post-fit **serving-validation subprocess** (`automl/runner/serving_validation.py`)
loads the just-trained model as an MLflow pyfunc and benchmarks it. Two distinct
ways it currently fails badly — both seen on the neobank_ncm full-data loop:

1. **A timeout crashes the handler that was meant to keep the trial alive.** On
   `TimeoutExpired`, `exc.stderr` is `bytes`; the old tail-slice kept it as bytes,
   so `json.dumps(report)` raised `TypeError` — the "don't crash the trial"
   handler crashed the trial. And the 120s cap was at the boundary on full data.
2. **A native crash (SIGSEGV) is completely silent.** A torch pyfunc reload under
   multiprocessing on macOS SIGSEGVs the subprocess (exit < 0), leaving **no
   result JSON, no MLflow run, no error tag** — invisible except in stdout.

And above both: **one FAILED validation halts the whole loop**, even when the
model trained fine and `eval.test.auc` was already logged.

## What we're looking for (goals)

- A problem in the validation subprocess — timeout **or** native crash — never
  leaves a trial silently dead: always a recorded failure report + error tag.
- A configurable, generous timeout (not a hardcoded boundary-tight cap).
- A clear, deliberate answer to: **should a validation *timeout/crash* fail-soft**
  (record it, keep the trained model + its eval metric, let the loop continue)
  **vs. halt** — distinct from a genuine *correctness* failure.

## Status

**Findings captured, design decision open.** Promoted to `execution/`
2026-06-10. Two migrated findings are the evidence:
[`finding-timeout-crash-and-halt.md`](finding-timeout-crash-and-halt.md) and
[`finding-nn-pyfunc-sigsegv.md`](finding-nn-pyfunc-sigsegv.md). Next session:
make the fail-soft-vs-halt call, then write `plans/`.
