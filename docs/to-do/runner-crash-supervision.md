# To-do: a crashed runner process should still yield a finalized MLflow record

> Part of the holistic [`runner-error-visibility/`](runner-error-visibility/)
> umbrella — this is **layer 2** (out-of-process supervision). Read that README
> for the full failure taxonomy and shared design before picking this up.

**Status:** parked 2026-06-10 out of the `trial-reliability` design (its
consciously-accepted boundary #3). Real gap, deliberately not solved there.

## Problem

When the **runner parent process dies natively** (e.g. the torch SIGSEGV /
exit 139 observed on the neobank_ncm full-data loop, 2026-06-10), no in-process
handler runs: the MLflow run is left in RUNNING with no error tag, no failure
artifacts, no `AUTOML_ERROR` marker. The `trial-reliability` effort makes this
*diagnosable* — `faulthandler` traceback on stderr, crash-safe local issues
JSONL in the trial dir — but not *durable*: by the two-stores principle that
evidence isn't a record until it reaches MLflow.

## Shape (options, not decided)

A supervisor seam **outside** the runner process. Candidates:

- **(a) Launcher wrapper:** the trial entry (`run.py` / CLI verb) runs the
  trial in a child process and, on a nonzero/signal exit, finalizes MLflow —
  publish the local issues JSONL, mark the run FAILED with an error tag, emit
  the `AUTOML_ERROR` marker.
- **(b) Reconcile verb:** an `automl trial reconcile` the manager skill (or the
  next session) runs after observing a crashed trial — backfills the run from
  the local evidence.
- **(c)** Fold into the future **modular/retryable runner** design, which needs
  process supervision anyway for resume.

(a) is the most self-contained; (c) avoids designing supervision twice. Decide
when picked up.

## Don't forget

- The `trial-reliability` pillar-4 ledger (TrialContext, local JSONL,
  `trial/issues.json` schema) provides exactly the evidence a supervisor would
  publish post-hoc — build on it, don't invent a second record.
- Relates to `multi-runner-architecture.md` (process supervision overlaps) and
  the manager skill's stop-without-repair (`agent-skills/skills/automl/SKILL.md`
  item 13).
- Delete this file once a natively-crashed runner reliably yields a finalized
  FAILED run in MLflow.
