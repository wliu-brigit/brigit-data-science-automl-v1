# Runner error visibility — the holistic picture

**Status:** framing captured 2026-06-10, after the neobank_ncm full-data loop
hit a torch **SIGSEGV (exit 139)** during `model.fit` and left **no MLflow
record at all** (trial count unchanged, no run, no error tag — the failure was
visible only as the manager's stdout narration). Not started. This is an
**umbrella** that ties together the scattered runner-visibility to-dos so we
design one coherent model instead of patching sites. **No code yet — think it
through first.**

## Why this is one problem, not several

"The runner failed and I couldn't see why" has shown up repeatedly, but the
fixes have been filed as unrelated notes. They are the same problem at
different severities. The unifying question: **for every way a trial can go
wrong, does a durable record reach MLflow (the two-stores source of truth), or
does it only exist as transient stdout / local scratch that dies with the
session?**

## Failure taxonomy — what's visible today vs. silent

Grounded in the current code (verified 2026-06-10):

| Failure mode | Where it's handled | Durable record today? |
|---|---|---|
| Python exception in the main runner | `trial.py:258` `try/except Exception` → `_publish_failure_artifacts` | ✅ ledger + FAILED run + error artifacts |
| Validation **subprocess** timeout | `serving_validation.py` timeout path | ✅ report + ledger issue |
| Validation **subprocess** signal-kill (child SIGSEGV) | `serving_validation.py:586` (`returncode < 0` → `SignalExit`) | ✅ report + ledger issue |
| **Main runner process native crash** (SIGSEGV in `fit`/`log_model`) | — nothing | ❌ **silent** → `runner-crash-supervision.md` |
| **Best-effort swallow** (degraded but non-fatal, e.g. `_try_log_train_eval` `except: return`) | the swallow site itself | ❌ **silent** (result simply absent) → `runner-best-effort-visibility.md` |

The top three are already solved by the `trial-reliability` effort. The bottom
two are the open gaps — and they are **opposite ends of the same axis**.

## The core principle (the "logic behind it")

**A process cannot record its own death.** Every in-process mechanism —
`try/except`, `finally`, `atexit`, the pillar-4 issue ledger — only runs if the
process survives long enough to run it. That cleanly explains the table:

- A Python **exception** is survivable → the handler runs → recorded. ✅
- A **child subprocess** dying is observable *by the surviving parent* → the
  parent records it (`serving_validation.py:586`). ✅
- The **main runner process** taking a SIGSEGV is **not** survivable by itself →
  no in-process code runs → nothing recorded. ❌ Only an **outside** process can
  record this death.
- A **deliberately swallowed** degradation is survivable but the code *chooses*
  not to record it → recorded nowhere by design. ❌

So a complete visibility model needs **two layers**, and we currently have one
and a half:

1. **In-process recording** (mostly built): exceptions + observable subprocess
   exits land in the ledger / MLflow. **Missing piece:** a consistent
   "degraded-but-not-fatal" breadcrumb channel so swallow sites stop being
   silent (`runner-best-effort-visibility.md`).
2. **Out-of-process supervision** (entirely missing): a parent that runs the
   runner as a child, observes its exit code/signal, and finalizes the MLflow
   record when the child died in a way the child couldn't self-report
   (`runner-crash-supervision.md`).

## The two sub-problems (existing docs, now framed under this umbrella)

- **[`../runner-crash-supervision.md`](../runner-crash-supervision.md)** — layer
  2. Native crash → no record. Needs a supervisor seam outside the process.
  Options already sketched there: (a) launcher wrapper in `run.py`/CLI verb,
  (b) an `automl trial reconcile` verb, (c) fold into the multi-runner design.
- **[`../runner-best-effort-visibility.md`](../runner-best-effort-visibility.md)**
  — layer 1's missing piece. Swallowed degradations → no breadcrumb. Needs one
  consistent warnings channel (tag + `logs/warnings/` artifact), applied by an
  audit of *all* swallow sites, not patched one at a time.

## Process — how we should approach the work

1. **Design the model before any code.** Decide the full set of trial outcomes
   we want guaranteed-recorded (success, soft-fail/exception, native crash,
   degraded-but-finished) and the *one* shape each lands in (MLflow tag +
   artifact, reusing the pillar-4 `trial/issues.json` ledger as the evidence
   substrate — **do not invent a second record**).
2. **Pick the supervisor seam.** (a) is the most self-contained; (c) avoids
   designing supervision twice if `multi-runner-architecture.md` is near-term.
   This is the biggest open decision.
3. **Then** the best-effort-swallow audit + breadcrumb channel can be designed
   inside the same model so fatal and degraded paths report consistently.
4. Relate to **[`../loop-observability.md`](../loop-observability.md)** (the
   loop should narrate liveness) and **[`../logging-and-observability.md`](../logging-and-observability.md)**
   (the deferred general logging design) — those are the *agent/loop* layer; this
   umbrella is the *durable-record* layer. Keep the boundary explicit.

## What NOT to do

- **No site-by-site patches.** Both sub-docs explicitly warn against this; a
  one-site tag was tried and reverted. The value is the consistent model.
- **A diagnostic / reporting path must never be able to fail a trial.** The
  recording layer is strictly additive.
- **Don't duplicate the ledger.** The supervisor publishes the existing local
  evidence post-hoc; it does not define a parallel record.

## Open questions to think through

- Should the supervisor live in `run.py` (every trial is supervised by default)
  or in the orchestrator's trial-run step (the manager already *sees* exit 139 —
  it just doesn't persist it)? The crash was already *visible* at the agent
  layer; the gap is purely **durability**. Does that argue for the orchestrator
  recording it rather than a new launcher?
- Is a RUNNING-but-orphaned MLflow run left by a native crash something the
  supervisor should mark FAILED, or should the run only be created once we know
  the outcome? (Affects whether we need a reconcile verb regardless.)
- One umbrella effort, or keep the two sub-docs as separate executions that
  share this design note? (Leaning: design here, execute the two in sequence —
  supervision first, since it's the harder seam and the one that bit us.)

**Delete/retire** the two sub-docs and this umbrella once every trial outcome —
including a native crash — reliably yields a finalized MLflow record.
