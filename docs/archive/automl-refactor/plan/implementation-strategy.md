# AutoML Refactor — Implementation Strategy & Sequencing

**Status:** ✅ SIGNED OFF approach; updated after the final whole-refactor audit (2026-05-28). This is the
**higher-level guidance** doc that sits *above* the detailed task plans. It settles *how* we
migrate, the dependency-driven order, and *how we stay honest* — not the per-function TDD steps.
Those detailed plans are produced **one phase at a time** (via the `writing-plans` skill) right
before each phase executes, so they don't rot before they're run.

**Reads with:** `../spec/00-structural-design.md` (the architecture) + the `../spec/NN-*.md`
sub-specs (per-domain interfaces); `migration-checklist.md` (symbol-coverage ledger) and
`acceptance-checklist.md` (behavior gates) in this same `plan/` folder. Front door: `../README.md`.

---

## 1. Migration approach (recap — settled)

- **Side-by-side, clean cut.** On a refactor branch (off the *current* branch, not `main`),
  `git mv automl/ automl_legacy/` to **freeze** today's package as read-only reference,
  then build a **fresh** `automl/` package with the new four-layer tree. The worktree directory
  is named `automl_dev-refactor`; the package under construction is still `automl/`.
- **Isolation via git worktree.** The refactor lives in its own worktree (per
  `superpowers:using-git-worktrees`) so the current environment is undisturbed. The
  worktree is *where* we work; the legacy-rename is *how* the branch is structured — they
  compose.
- **Port-and-reshape, not rewrite-from-scratch and not edit-in-place.** For each symbol:
  read the legacy file (reference), reshape its logic into its new home (renames, file
  split/merge, `session` convention, route writes through the mlflow seam), write fresh
  tests, check off its `migration-checklist.md` row(s).
- **No bridges.** New code imports **only** from the new tree, never from `automl_legacy/`.
  The two trees never call each other, so there is no shim/dual-write/temporary-interop
  code. (Matches `feedback_no_back_compat`.)
- **Atomic cutover at the end.** Phase 7 completed the swap: the checklist is all-green, the
  harness passed, and `automl_legacy/` plus `tests_legacy/` are deleted. One swap; no window
  where both are "live."

**Unit of work = a cohesive capability/module, never one function at a time.** The
checklist tracks *symbols* for completeness; TDD operates at the *behavior* level.

---

## 2. Cross-domain dependency graph

Derived from each sub-spec's declared outbound deps (00 §8). Arrows = "depends on."
`mlflow/` is deliberately decoupled: **domains call mlflow *functions*; mlflow imports
domain *types*** to return them — a one-way split that avoids a cycle (00 §9.1).

```
                         ┌─────────────────────────────────────────────┐
   LEAVES                │  utils/ (hashing, io, paths, logging)         │
                         │  errors.py                                    │
                         └───────────────▲───────────────▲──────────────┘
                                         │               │
   FRAMEWORK        ┌───────────────────────────┐   ┌────┴───────────────┐
                    │ mlflow/  (seam)            │   │ validate/ (frmwk)  │
                    │  imports domain *types*    │   │  imports checks.py │
                    └───────▲────────────▲───────┘   └────▲───────────────┘
                            │ (functions) │                │
   CONTEXT          ┌───────┴─────────────┴────────────────┴──────────────┐
                    │ project/  (ProjectConfig, Session — threads via      │
                    │           contextvar into EVERY domain)              │
                    └───▲──────────▲───────────▲──────────────▲────────────┘
                        │          │           │              │
   TIER-3 ANCHORS   ┌───┴───┐  ┌───┴───┐   ┌───┴────┐         │
                    │ model │  │ data  │   │  eval  │─────────┘  (eval→data, concept-level)
                    └───▲───┘  └───▲───┘   └───▲────┘
                        │          │           │
   ORCHESTRATION    ┌───┴──────────┴───────────┴───┐
                    │ runner/  (data→fit→eval→log)  │   ← FIRST end-to-end point
                    └───────────────▲──────────────┘
                                    │
   NOUN/LOOP        ┌───────────────┴───────────────────────────┐
                    │ trial/   experiment/   agent/              │
                    │ (trial→runner types; experiment→trial;     │
                    │  agent→experiment.views+trial+runner-via-  │
                    │  trial.promote)                            │
                    └───────────────▲───────────────────────────┘
                                    │
   SURFACE          ┌───────────────┴───────────────┐
                    │ cli/   hooks/ (→ agent.timeline)│
                    └────────────────────────────────┘
```

**Topological build order** (leaves first): `utils`+`errors` → `mlflow` seam + `validate`
→ `project` → `data`/`model`/`eval` → `runner` → `trial`/`experiment`/`agent` →
`cli`/`hooks`.

**Key observation:** the *integration risk* lives at the **contracts between layers** — the
mlflow-seam return types, the `session` threading, the Tier-3 ABCs, the `TrialDataContract`.
Not inside any one domain. So the strategy below front-loads exercising those contracts.

---

## 3. The strategy: vertical slice first, then thicken

A pure bottom-up build (finish each layer before the next) means six layers exist before
anything runs end-to-end — **migrating blind**. We invert it:

- **Phase 1 builds a "walking skeleton"** — the *thinnest* path through **every** layer that
  executes **one real Home Credit trial**. Minimal everything. Its job is to prove the
  architecture *integrates* (seam contracts, session threading, routing) with ~10% of the
  code, not 90%.
- **Phase 1 starts with a harness and ratchets, not broad module ports.** The first executable
  artifacts are the environment preflight, contract tests, and the exact Home Credit fixture/model
  that define the A1 gate. Only then do we port primitives and seam code. This prevents a pile of
  plausible modules from drifting away from the actual end-to-end acceptance path.
- **Phases 2..N thicken** each domain to full fidelity, with the harness green at every step.
  After Phase 1 we are **never migrating blind** — a running end-to-end system catches
  regressions continuously.

Safe (integration de-risked early) **and** fast (once contracts are proven, thicken in big
cohesive chunks, not function-by-function).

---

## 4. Phase roadmap

> ⚠️ **Closed roadmap.** Phases 1 through 7 have now run as code and validated the
> vertical slice, data/model breadth, eval breadth, experiment/trial cleanup, agent/hook gates,
> CLI surface/isolation gate, and final cutover gate. Future changes should use the final audit
> and open-items docs, not extend this migration roadmap. What §8's sign-off locks is the
> *approach* (side-by-side, vertical-slice-first, the dependency order); when reality and the plan
> disagree, update this doc + the checklists + flag the user.

Each phase ends at a **named, runnable acceptance gate** against the Home Credit harness
(local MLflow `127.0.0.1:54321`, GCS `gs://automl-homecredit-kaggle-wliu`, `local_csv`
adapter; runs in an `automl_runs/` copy — never the base sandbox). In this refactor worktree,
load the local `.env` copied from the original `automl_dev/.env` before hitting the original
MLflow server.

### Phase 0 — Pre-flight (no production code)
- **Completeness audit:** diff `migration-checklist.md` against the real `automl_legacy/`
  tree; every public symbol must have a disposition (ported-to-X or `[-]` dropped). Add
  missing rows.
- **Author `acceptance-checklist.md`:** behavior milestones (e.g. "runner executes one full
  trial," "proposer loop emits a validated Proposal," "cleanup removes one experiment's
  blobs"). This is the behavior-level complement to the symbol-level migration checklist.
- **Scaffold:** worktree + branch + `git mv` rename + empty new-tree skeleton + the new
  `tests/` layout + `pyproject.toml`/`testpaths` (00 §13.7 ratchet test updated in the same
  commit).
- **Gate:** `uv run python -c "import automl"` succeeds on the empty skeleton; checklist has
  zero un-dispositioned rows.

### Phase 1 — Walking skeleton (the vertical slice) ★ the big de-risk
Thinnest cut through every layer to run **one** trial. The internal execution order is now:

1. **P1.0 Environment/preflight**: `uv sync`, import check, dependency availability, and optional
   Home Credit service probes (local MLflow + GCS) with clear skip/fail behavior.
2. **P1.1 Contract ratchets**: architecture tests for package shape, no `automl_legacy` imports,
   no direct PyPI `mlflow` imports outside `automl/mlflow/`, layer boundaries, and pytest path
   policy. These are early guardrails, not a final-phase cleanup.
3. **P1.2 Harness model fixture**: add the exact Home Credit `model.py` early and test it
   standalone. The typed `config.py` wiring waits until the project/data/eval contracts exist,
   then lands before the runner gate. The model is robust enough for the real data: numeric
   feature selection + imputation + LogisticRegression with `predict_proba`. No categorical/WOE
   breadth in Phase 1.
4. **P1.3 Leaf primitives**: `errors`, `utils/hashing`, `utils/io/gcs`, `utils/paths`,
   `utils/logging`.
5. **P1.4 Project context + facade**: `ProjectConfig`, `Session`, `use_project`, `session`,
   `active_session`, `clear_session`, `update_session`, `RunConfig`, `Splits`, and early Tier-1
   exports.
6. **P1.5 MLflow seam**: `bind/bound`, route naming, tags, experiment ensure/next-trial-number,
   active trial context, metrics/json/artifact logging. Only this domain imports PyPI `mlflow`.
7. **P1.6 Data thin path**: `DataSpec`, `LocalCSVSource`, `DataPipeline`, materialize/load,
   `Dataset`/`LoadedDataset`/`LoadedSlice`, minimal `FeatureRegistry`, `TrialDataContract`, and
   L2 load-time integrity. Implement single-range fit/eval splits for A1; multi-range breadth
   landed in Phase 2.
8. **P1.7 Model/eval/validate thin path + typed harness config**: `BaseModel`, `save_model`,
   `Metric`, `Auc`, `EvalSpec`, `prepare_eval_dataset`, `evaluate`, `EvalResult`, the pre-fit
   model gate, and the new-style Home Credit `config.py` import test.
9. **P1.8 Runner/e2e**: runner loads **only the fit slice**, validates pre-fit, opens MLflow,
   fits, prepares eval recipe, calls `evaluate()` to own eval loading, logs metric/model/data/eval
   artifacts, and proves A1.1-A1.4.

- **Gate:** one Home Credit trial runs end-to-end — loads real data, fits the real-but-simple
  model, computes real AUC, opens an MLflow run, writes the trial contract + eval + model
  artifacts to local MLflow + GCS. **The architecture is proven on real data.**
- **Explicitly NOT in Phase 1:** agent loop / proposer-context / timeline / hooks;
  experiment views (leaderboard/compare/summary); cleanup cascade; profile; >1 metric or
  DataSource; `required_transformers` gate (empty default); `--namespace`/`--dry-run`
  breadth beyond the routing primitives; full CLI surface.

### Phase 2 — Data & model breadth
Completed 2026-05-27. Scope landed: source/index breadth (`local_csv`, `gcs_parquet`,
Snowflake stub), `build_dataset`/`list_datasets`, full `FeatureRegistry`
(`derived`/`source_columns`/`add_derived`), L1–L4 validators + multi-range loader,
`model/preprocessing.py` (`RequiredTransformer` + gate), Home Credit WOE requirement, and
`profile` through project-overview artifacts. The dependency order was:

1. Data source/index breadth with the existing Phase 1 runner still green.
2. FeatureRegistry breadth.
3. Full validators + multi-range loader.
4. Required-transformer contract + Home Credit WOE-gated trial.
5. Profile/project-overview artifact path.

**Gate:** a trial using a project-mandated transformer (Home Credit `WOEEncoder` on
`ORGANIZATION_TYPE`) runs green, and profile produces the expected project artifacts. Passed
2026-05-27 against `http://127.0.0.1:54321`.

### Phase 3 — Eval breadth
Completed 2026-05-28. Scope landed: `LogLoss`/`ThresholdSweep`, scalar metric extraction and
namespaced eval logging, durable split-view and external `EvalDataset`, `Augmentation`,
`Predictions`, `EvalIndex`, evaluate-owned persistence/cache-light reuse, seam-local model load,
and runner integration. Broader `eval/checks.py` and CLI eval surfaces remain later-phase work.

**Gate:** a trial evaluated against an external eval dataset + augmentation, predictions
persisted. Passed 2026-05-28 against `http://127.0.0.1:54321`.

### Phase 4 — Experiment & trial domains
Completed 2026-05-28. Scope landed: `experiment/` lifecycle/read views
(`ExperimentOverview`, leaderboard, compare, summary, `recent_failures`,
`strategies_attempted`), `trial/` read APIs (`TrialSummary`, `TrialDetails`, `show_trial`,
`load_model`), and cleanup cascade (`project/cleanup` + experiment/trial wrappers;
soft-delete default; hard-delete command path unit-tested). Trial create/fork/promote/packaging
remain later-phase rows unless a future gate needs them.

**Gate:** leaderboard + compare over several trials; `trial show`; `experiment delete`
removes one experiment's blobs in one QA namespace universe. Passed 2026-05-28 against
`http://127.0.0.1:54321`; Phase 3 external eval regression also stayed green.

### Phase 5 — Agent domain + hooks
`agent/`: `Proposal` + `proposal_schema`, `gather_proposer_context`, `build_launch`,
`timeline` (seam-routed) + thin `hooks/` stub. `AUTOML_INHERIT_DRY_RUN` transport.
**Gate:** the proposer→coder loop emits a validated `Proposal`, runs a trial, and the
timeline reconciles agent events into MLflow.

**Status:** complete on 2026-05-28. The Phase 5 gate is proven by the narrow A5 CLI wrappers,
the library launcher/context/proposal/timeline surfaces, and an opt-in harness test that validates
a proposal, builds the loop launch command, runs one real Home Credit trial, and publishes
agent reports/metrics through MLflow/GCS.

### Phase 6 — Surface & isolation breadth
Completed 2026-05-28. Scope landed: noun-first CLI dispatcher and modules
(`project`, `experiment`, `trial`, `data`, `eval`, `validate`), root session flags
(`--project`, `--project-root`, `--dry-run`, `--namespace`, `--experiment-id`), bounded
project metadata/scaffold/validation helpers, eval listing, `data materialize`, session-lock
CLI backed by `runner/session_lock.py`, and skill/agent/reference command updates. Trial
authoring (`trial create`/`fork`/`promote`) remains Phase 7 carryover.

**Gate:** `automl --namespace qa …` runs + cleans an isolated sandbox without touching real;
all skill commands resolve to new verbs. Passed 2026-05-28 against
`http://127.0.0.1:54321`; Phase 5, Phase 4, and Phase 3 external preservation gates also stayed green.

### Phase 7 — Cutover
Completed 2026-05-28. Scope landed: trial authoring (`trial.create`, `fork`, `promote`),
trial-folder packaging and source-artifact persistence, runner folder execution with project-model
fallback preserved, CLI `trial create <slug>` / `fork <slug>` / `promote <slug>`, final
testpath/architecture ratchets, migration-checklist disposition, `projects/payment_routing`
import cleanup, final e2e harness, and deletion of `automl_legacy/` plus `tests_legacy/`.

**Gate:** the fresh `automl/` package is the only package; legacy deleted; harness e2e passes.
Passed 2026-05-28 against `http://127.0.0.1:54321`; Phase 6, Phase 5, Phase 4, and Phase 3
external preservation gates also stayed green.

### Final whole-refactor audit
Completed 2026-05-28 after the Phase 7 commit. Scope covered spec alignment, migration
completeness, architecture/import safety, end-to-end behavior, and documentation closeout.
Obvious findings were fixed with tests/contracts first: retired dry-run env guidance, stale
`route_namespace` naming in the skill renderer, snapshot-era user-facing docs, migration-note
status drift, and one widened-ruff cleanup in an integration test. A follow-up readiness review
then recorded remaining deferred/spec-gap items in `spec-coverage-review.md` and
`final-review-open-items.md`; those items need triage before merge.

**Gate:** full local suite, contracts, widened ruff, import/stale-surface ratchets, and Phase
7/6/5/4/3 external gates were green at final audit. The branch is committed, but not merge-ready
until the follow-up open/deferred items are triaged.

---

## 5. Anti-drift & completeness (how we stay honest)

Four mechanisms, checked continuously — not one-time:
1. **`migration-checklist.md` (symbol coverage).** Every legacy public symbol → ported or
   dropped. Each task flips its rows; **cutover is blocked while any row is un-dispositioned.**
2. **`acceptance-checklist.md` (behavior coverage).** Each phase's gate is a row; proves
   *functionality preserved*, not just *symbols present*.
3. **Contract tests** (`tests/contracts/`) — pin the architectural invariants (domain import
   boundaries, `session` convention, seam-only mlflow access, the four-layer shape).
4. **The Home Credit harness as the live safety net** — green at every phase gate; after
   Phase 1 it runs continuously, so nothing migrates blind.

Every detailed task in a per-phase plan cites its **spec section** + its **checklist row(s)** —
the trace from design → task → coverage is explicit.

At every phase closeout, update the docs in the same pass: front-door README, `plan/README.md`,
this strategy doc if the roadmap changed, `acceptance-checklist.md`, `migration-checklist.md`,
and the just-run phase plan. Then scan for stale "next action" / old-phase language before
handing off. Specs stay stable unless implementation proves a design contract wrong or missing.

---

## 6. Test strategy (the ~992 legacy tests)

**Decision (2026-05-27): rebuild fresh per-domain (TDD).** Per `00 §13.7` (pruning is the
plan's job) + clean-cut:
- **Unit tests are rebuilt per-domain (TDD)** as each domain is ported — written fresh in
  `tests/unit/<domain>/`, using legacy tests **only as a behavior reference**, pruned/collapsed
  aggressively. **Not migrated 1:1** — legacy test debt is shed, not carried.
- **Contract tests** are authored fresh to pin the new architecture (Phase 0/1).
- **Integration + e2e** tests are organized by scenario; the harness e2e is the acceptance
  gate at each phase.
- The Phase 0 audit still **classifies** each legacy test tier (to know which behaviors to
  re-cover) and surfaces real counts — but the default disposition is rebuild-fresh, not port.

**Pruning rule added after Phase 1 plan review.** A legacy test is rewritten only when it
protects a stable user-visible contract, a new architecture invariant, or a cross-domain behavior
used by a phase gate. Tests tied to legacy filenames, private helpers, old compatibility shims,
old persisted-state formats, or implementation-only sequencing are dropped or replaced by a
smaller contract/integration test.

---

## 7. What this doc does NOT decide (left to per-phase plans)

- Exact per-function TDD steps, file-by-file (the `writing-plans` output, written
  just-in-time per phase).
- Exact argparse flag names / output formats (00 §11.1 defers these).
- Final test-prune counts (Phase 0 audit produces them).
- Git sub-branch hygiene within the refactor branch (execution detail).

---

## 8. Sign-off checklist — ✅ ALL SIGNED OFF (2026-05-27)

- [x] Migration approach (§1) — side-by-side / worktree / port-reshape / atomic cutover.
- [x] Dependency order + vertical-slice-first strategy (§2–§3).
- [x] Phase boundaries + acceptance gates (§4) — **7-phase cut kept** (data+model together;
      experiment/trial in Phase 4; agent in Phase 5).
- [x] Anti-drift mechanism + dual checklists (§5).
- [x] Test strategy (§6) — **rebuild fresh per-domain (TDD)**.
- [x] Phase 0 depth — **full Phase 0 first** (audit + acceptance-checklist + scaffold before
      any domain code).
- [x] Phase 1 skeleton fidelity — **real-but-simple model on real Home Credit data**.

**Current next action:** triage `spec-coverage-review.md` and `final-review-open-items.md`.
Decide which deferred/spec-gap items are pre-merge blockers, then fix blocker items with tests
first before any merge/cutover decision.
