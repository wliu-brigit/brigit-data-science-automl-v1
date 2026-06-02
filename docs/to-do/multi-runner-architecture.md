# Multi-Runner Architecture — Sibling Runner Types

## Status

Priority: P2. Deferred, focused iteration. Multi-runner support is plausibly
near-term, but it is **already cleanly separated** from the rest of the system
(the runner domain has a tight, contract-pinned boundary), so it can be designed
and built as its own pass rather than entangled with the current review-findings
cleanup.

This note is a future-work assessment captured on **2026-05-29** (branch
`refactor/four-layer`) during a structural review. It records the current shape,
the one real blocker, and the design options so a future implementer starts from
context — it is **not authoritative** and does **not** prescribe a solution.
Re-read the referenced code and re-verify the call chain before acting.

**Relationship to the cleanup pass:** one review finding (split
`runner/artifacts.py`) is effectively the *prep work* for this iteration. Doing
that cleanup first leaves the runner domain in the state this design wants. See
"What to do when this is picked up" below.

## Problem

Today there is exactly one runner: the straight-line trial chain in
`runner/trial.py` (`data → fit → eval → log`). The design (`spec/00` §13.3,
§17.2; `spec/08-runner.md`) deliberately ships **no stage abstraction** and **no
runner-type registry** — per `feedback_extension_points_follow_demand`, the
shared-primitive extraction waits until a real second runner exists.

A second runner shape is now anticipated (HPO sweep, feature ablation,
data-ratio sweep, ensemble). `runner/failures.py::RunnerFailureReport` already
carries a `runner_kind` field, so the data model anticipates this even though no
second runner exists. The question this note defers: **when the second runner
lands, what is the clean way to add it without a central dispatcher and without
duplicating the chain?**

## Current structure (verify before trusting)

- `runner/trial.py` (~665L) — the single chain, readable top-to-bottom with
  explicit phases wrapped in `timing.phase()`. **This is good and should not be
  re-abstracted into a stage framework speculatively.**
- `runner/paths.py`, `runner/contract.py`, `runner/failures.py`,
  `runner/session_lock.py`, `runner/template.py` — cohesive, single-purpose
  modules. The `runner ↔ trial` split is acyclic and contract-pinned
  (`tests/contracts/test_architecture.py:235`): `trial/` imports *down* into
  `runner`, `runner` never imports `automl.trial`.
- `runner/artifacts.py` (~937L) — **the blocker.** It fuses three unrelated
  responsibilities (see review finding "runner/artifacts.py monolith"):
  1. `TimingRecorder` (`:46-71`) — a chain phase-timer, not an artifact;
     imported by `trial.py`.
  2. thin seam pass-throughs (`log_model`, `log_data_contract`, `log_manifest`).
  3. a ~700-line **serving-validation subsystem** (`:155-877`) including a
     286-line embedded subprocess-script string (`:474-760`) doing fixture
     generation, pyfunc load, schema checks, and latency benchmarking.

## Why this is the real blocker

The intended extension shape (from the spec) is: when a second runner is needed,
extract the shared primitives into `runner/_stages.py` and add `runner/<new>.py`
as a sibling that composes them differently. **No central dispatcher; each new
runner gets its own CLI verb when promoted.**

But a second runner today would have to reach into `runner/artifacts.py` to reuse
the timer or the publish wrappers — i.e. it would inherit a 937-line junk drawer,
and the serving-validation subsystem (arguably the package's serving-readiness
*contract*, a first-class concern) has no findable home of its own. So the
monolith is exactly what makes "add a sibling runner" messy.

## Design options (for the future session, not decided)

1. **Extract shared primitives into `runner/_stages.py` at second-runner time**
   (the spec's plan). The shared set is likely: SIGALRM/timeout arming, the
   session lock, the pyfunc round-trip/serving validation, the manifest writer,
   and `TimingRecorder`. `trial.py` and `runner/<new>.py` both compose these.
   No `Stage` ABC unless projects need to *inject* custom stages (then promote to
   `runner/stages/base.py` — `spec/00` §17.3).
2. **Sibling files, no shared module** — only viable if the second runner shares
   almost nothing with the chain (unlikely; they share timeout/lock/logging).
3. **A `Runner` ABC / registry** — explicitly rejected by the spec
   (`feedback_extension_points_follow_demand`). Do not pre-build.

## Open decisions to settle when picked up

- **What is the first real second runner?** (HPO / ablation / data-ratio /
  ensemble.) The design should be against `runner/trial.py` + *one concrete*
  second runner as two real consumers — not against a hypothetical.
- **What primitives are genuinely shared** vs chain-specific? That set defines
  `runner/_stages.py`. Likely shared: timeout, session lock, pyfunc/serving
  validation, manifest, timing. Likely chain-specific: the data→fit→eval→log
  ordering itself.
- **Does the second runner reuse the `Proposal` contract** or need its own
  config shape (e.g. an HPO search space)? This intersects the agent domain
  (`multi-agent-orchestration.md`) if a proposer must emit the new shape.
- **CLI verb** — a sibling under a noun (e.g. `automl experiment run --runner hpo`
  or a new verb), decided at promotion time per the spec's lean-CLI rule.

## What to do when this is picked up

1. **First land the cleanup split of `runner/artifacts.py`** (a review finding,
   independently worth doing): `TimingRecorder → runner/timing.py`; the
   serving-validation subsystem → `runner/serving_validation.py` (it is the
   serving-readiness contract and deserves a named home); keep only the thin
   `log_*` wrappers in `artifacts.py` and route their raw writes through the
   MLflow seam (drop the `client.raw().log_artifact` bypass). After this,
   `runner/trial.py` + the cohesive modules are a clean template a sibling runner
   can sit beside.
2. **Then** design the `runner/_stages.py` seam against the cleaned chain + the
   one real second runner. Write a proper planning doc at that point.
