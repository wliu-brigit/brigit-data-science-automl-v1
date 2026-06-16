# Runner best-effort visibility

> Part of the holistic [`runner-error-visibility/`](runner-error-visibility/)
> umbrella — this is **layer 1's missing piece** (a breadcrumb channel for
> swallowed, non-fatal degradations). Read that README for the full failure
> taxonomy and shared design before picking this up.

## Status

Captured 2026-06-09 (branch `neobank_NCM_V3_replicate`, neobank_ncm CSV QA
run). Not started. A one-site spot fix (tag on the swallowed exception) was
tried and deliberately reverted — this wants a holistic pass, not patches.

## Problem

The runner's best-effort paths swallow failures silently. Concrete,
reproduced case: `automl/runner/trial.py::_try_log_train_eval` is
`except Exception: return`, so when the train-split diagnostic eval fails,
the trial finishes FINISHED and `eval.<train_split>.*` is simply absent —
no tag, no artifact, no log explaining why.

This fires on every trial of any reject-inference-style project: the train
split legitimately contains NULL-target rows (unknown group carries
synthetic soft labels instead), and the metric computation raises
`ValueError: Input y_true contains NaN`. The skip itself is correct
behavior; the silence is the defect. It cost a debugging session to notice
the metric was missing and trace why.

## The ask

- Audit the runner (and harness paths generally) for **all** best-effort
  swallow sites (`except Exception: pass/return`), not just this one.
- Design **one consistent breadcrumb mechanism** for "degraded but not
  fatal": e.g. a warnings channel on the trial run — a tag and/or a
  `logs/warnings/` artifact, mirroring the existing `logs/errors/`
  convention for fatal failures — that every best-effort site reports
  through.
- Invariant to preserve: a diagnostic must still never fail the trial.

## What NOT to do

- Do not patch sites one at a time as they bite — that hides the pattern.
- Do not let the diagnostic-reporting path itself become able to fail a
  trial.

Related: [`logging-and-observability.md`](logging-and-observability.md)
(process-logging strategy; this entry is narrower — durable breadcrumbs on
the MLflow record for swallowed degradations).
