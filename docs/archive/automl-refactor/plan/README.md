# AutoML Refactor — Execution (`plan/`)

The **execution** half of the refactor: *how* we build it, *in what order*, the *current
status*, and *what to do next*. (The design — *what & why* — lives in [`../spec/`](../spec/);
the project front door is [`../README.md`](../README.md).)

> ⚠️ **The phase plan is closed.** Phases 1 through 7 have now validated the first vertical
> slice, data/model breadth, eval breadth, experiment/trial cleanup, agent/hook gates, and the
> CLI surface/isolation/cutover gates. When the build contradicts the plan or the
> spec, **flag it and update the docs** — don't silently deviate, and don't follow a spec you've found to be wrong. Ask the user
> whenever something's unclear.

---

## ▶ Status (2026-05-28)

- ✅ Implementation strategy **signed off** (`implementation-strategy.md`).
- ✅ **Phase 0** — completeness audit (symbol coverage complete) + behavior gates authored.
- ✅ **Phase 0 scaffold** — worktree `../automl_dev-refactor`, legacy package frozen as
  `automl_legacy/`, fresh `automl/` skeleton, `import automl` green.
- ✅ **Phase 1** — walking skeleton complete. A1.1-A1.4 passed on 2026-05-27, including the
  external Home Credit e2e against real GCS. The detailed command evidence is recorded in
  `acceptance-checklist.md`.
- ✅ **Phase 2** — data/model breadth complete. A2.1-A2.5 passed on 2026-05-27, including the
  external Home Credit e2e against real GCS + the original MLflow server. Evidence:
  `uv run pytest tests/unit tests/contracts tests/integration -v` -> `159 passed`;
  `AUTOML_PHASE2_E2E=1 MLFLOW_TRACKING_URI=http://127.0.0.1:54321 uv run pytest tests/e2e/test_phase2_data_model_breadth.py -v` -> `1 passed`;
  `uv run pytest tests/contracts -v` -> `9 passed`.
- ✅ **Phase 3** — eval breadth complete. A3.1-A3.2 passed on 2026-05-28, including the
  external eval e2e against real GCS + the original MLflow server. Evidence:
  `uv run pytest tests/unit tests/contracts tests/integration -v` -> `191 passed`;
  `AUTOML_PHASE3_E2E=1 MLFLOW_TRACKING_URI=http://127.0.0.1:54321 uv run pytest tests/e2e/test_phase3_eval_breadth.py -v` -> `1 passed`;
  `uv run pytest tests/contracts -v` -> `9 passed`.
- ✅ **Phase 4** — experiment/trial reads + cleanup complete. A4.1-A4.4 passed on 2026-05-28,
  including the external QA cleanup e2e against real GCS + the original MLflow server. Evidence:
  `uv run pytest tests/unit tests/contracts tests/integration -v` -> `222 passed`;
  `AUTOML_PHASE4_E2E=1 MLFLOW_TRACKING_URI=http://127.0.0.1:54321 uv run pytest tests/e2e/test_phase4_experiment_trial_cleanup.py -v` -> `1 passed`;
  `AUTOML_PHASE3_E2E=1 MLFLOW_TRACKING_URI=http://127.0.0.1:54321 uv run pytest tests/e2e/test_phase3_eval_breadth.py -v` -> `1 passed`;
  `uv run pytest tests/contracts -v` -> `9 passed`.
- ✅ **Phase 5** — agent domain + hooks complete. A5.1-A5.3 passed on 2026-05-28, including
  the external agent/hook e2e against real GCS + the original MLflow server. Evidence:
  `uv run pytest tests/unit tests/contracts tests/integration -v` -> `244 passed`;
  `AUTOML_PHASE5_E2E=1 MLFLOW_TRACKING_URI=http://127.0.0.1:54321 uv run pytest tests/e2e/test_phase5_agent_hooks.py -v` -> `1 passed`;
  `AUTOML_PHASE4_E2E=1 MLFLOW_TRACKING_URI=http://127.0.0.1:54321 uv run pytest tests/e2e/test_phase4_experiment_trial_cleanup.py -v` -> `1 passed`;
  `AUTOML_PHASE3_E2E=1 MLFLOW_TRACKING_URI=http://127.0.0.1:54321 uv run pytest tests/e2e/test_phase3_eval_breadth.py -v` -> `1 passed`;
  `uv run pytest tests/contracts -v` -> covered by the full local gate.
- ✅ **Phase 6** — surface + isolation breadth complete. A6.1-A6.4 passed on 2026-05-28,
  including the external namespace/dry-run e2e against real GCS + the original MLflow server.
  Evidence: `uv run pytest tests/unit tests/contracts tests/integration -v` -> `273 passed, 2 warnings`;
  targeted Phase 6 sweep -> `108 passed, 1 skipped, 2 warnings`;
  `AUTOML_PHASE6_E2E=1 MLFLOW_TRACKING_URI=http://127.0.0.1:54321 uv run pytest tests/e2e/test_phase6_surface_isolation.py -v` -> `1 passed`;
  Phase 5/4/3 external preservation gates -> `1 passed` each;
  `uv run pytest tests/contracts -v` -> `11 passed`.
- ✅ **Phase 7** — cutover complete. A7.1-A7.4 passed on 2026-05-28. Evidence:
  targeted Phase 7 gate -> `58 passed, 1 skipped, 3 warnings`;
  `uv run pytest tests/unit tests/contracts tests/integration -v` -> `283 passed, 2 warnings`;
  `uv run pytest -v` -> `283 passed, 7 skipped, 2 warnings`;
  `uv run pytest tests/contracts -v` -> `13 passed`;
  Phase 7/6/5/4/3 external gates each -> `1 passed`;
  migration checklist has zero un-dispositioned rows; `automl_legacy/` and `tests_legacy/`
  are deleted.
- ✅ **Final whole-refactor audit** — complete on 2026-05-28. Evidence: full local default
  suite -> `283 passed, 7 skipped, 2 warnings`; contracts + cleanup retest -> `16 passed,
  2 warnings`; widened ruff over active Python surfaces -> `All checks passed`; Phase 7/6/5/4/3
  external gates each -> `1 passed`; stale surface/import scans clean.
- ⚠️ **Follow-up spec coverage/readiness review** — open/deferred items are now recorded in
  `spec-coverage-review.md` and `final-review-open-items.md`. Accepted gates remain green, but
  the branch is **not merge-ready yet** until those items are triaged.
- ⚠️ **NEXT ACTION:** triage the open/deferred items and decide which are pre-merge blockers.

---

## What's here

| Doc | What it is | Use it to… |
|---|---|---|
| `implementation-strategy.md` | The overarching plan — migration approach, cross-domain **dependency graph**, the **phased roadmap (0–7)** (vertical-slice-first), acceptance gates, test strategy. Signed off. | Understand the whole build + the order. **Read first.** |
| `acceptance-checklist.md` | **Behavior** gates A0–A7 — "can the new tree do X against the Home Credit harness?" Living status. | See what's done / what the next gate is. |
| `migration-checklist.md` | **Symbol** coverage ledger — every legacy public symbol → ported `[x]` or dropped `[-]`. Cutover is blocked while any row is un-dispositioned. | Track per-domain coverage during a phase. |
| `spec-coverage-review.md` | Follow-up Phase 0-7 spec coverage ledger with known exceptions and deferred scope. | Check what is covered, what is intentionally deferred, and what must be triaged before merge. |
| `final-review-open-items.md` | Open/debatable items found after the final audit. | Decide merge readiness and pre-merge blockers. |
| `phases/` | Per-phase detailed TDD plans, written **just-in-time** right before each phase runs (not all up front, so they don't drift). | Get the step-by-step for the phase you're building. |

## How a phase runs

1. Confirm the phase's acceptance gate(s) in `acceptance-checklist.md`.
2. Write that phase's detailed plan into `phases/phase-N-*.md` (`writing-plans` skill) —
   citing the relevant `../spec/NN-*.md` sections + the `migration-checklist.md` rows it covers.
   Treat the roadmap as provisional: re-order or split tasks when dependency evidence says to.
3. Execute it (TDD; rebuild tests fresh per contract). Flip checklist rows `[ ]`→`[x]`/`[-]`
   only after the code and verification land.
4. Run the harness; mark the acceptance row `[x]`. Then plan the next phase.
5. Close out the phase in docs before opening a fresh session: update the front-door README,
   this file, `implementation-strategy.md` if reality changed the roadmap,
   `acceptance-checklist.md`, `migration-checklist.md`, and the phase plan. Update `spec/` only
   for real design corrections, with the correction called out explicitly.

From Phase 5 onward, run this as an autonomous phase loop: include one self-review of the plan
before implementation and one code-review pass after implementation, but do not pause for manual
review unless a true blocker or material scope/risk decision appears.

## Fresh-session handoff

New sessions should start at `../README.md`. Current handoff: Phase 0-7 implementation and final
audit are committed, but merge readiness is intentionally open. Read `spec-coverage-review.md`,
`final-review-open-items.md`, `acceptance-checklist.md`, `migration-checklist.md`, and the Phase 7
closeout notes before deciding what to fix next.

## The two checklists, in one line

`migration-checklist.md` proves **nothing was left behind** (symbols); `acceptance-checklist.md`
proves **it still works** (behaviors). Both must be green for cutover (Phase 7).

> While working this branch, ignore the workspace-root `CLAUDE.md`'s "work in `automl_dev/`"
> pointer. This `plan/` + `../README.md` are the authority for the refactor worktree.
