# Phase 5 Agent Hooks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development`
> (recommended) or `superpowers:executing-plans` to implement this plan task-by-task.
> Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** pass A5.1-A5.3 by adding the agent domain, proposal validation, loop launch builder,
and timeline/hook reconciliation needed for a proposer-to-coder loop to emit a validated
`Proposal`, run a trial, and publish agent artifacts through the MLflow seam.

**Architecture:** Phase 5 is a relocate-and-boundary phase. New agent files host the Proposal
contract, proposer-context composer, launcher, and timeline logic; framework validation calls into
`agent.checks`; the hook becomes a transport stub. The only CLI work in this phase is the narrow
A5-facing wrappers for `experiment proposer-context`, `experiment run`, and `validate proposal`;
the full CLI catalog, top-level `--dry-run`/`--namespace` breadth, and skill-wide cleanup remain
Phase 6.

**Tech Stack:** Python 3.11 via `uv`; pytest; file-backed MLflow for unit/integration tests;
fake subprocess/GCS/MLflow clients for launcher and timeline tests; original local MLflow
`http://127.0.0.1:54321` with this worktree's `.env` loaded for opt-in external gates.

**Acceptance:** `plan/acceptance-checklist.md` rows **A5.1-A5.3**.

**Status:** complete on 2026-05-28. A5.1-A5.3 passed locally and against the external
Home Credit harness on `MLFLOW_TRACKING_URI=http://127.0.0.1:54321`.
The detailed checklist below is retained as the execution plan; all P5.0-P5.8 items were
completed unless explicitly called out in **Carryover to Phase 6**.

**Closeout evidence:**

- Baseline guard before code: `uv run pytest tests/unit tests/contracts tests/integration -v`
  -> `222 passed`.
- Phase 5 targeted/code-review sweep:
  `uv run pytest tests/unit/agent tests/unit/validate tests/unit/cli tests/contracts tests/e2e/test_phase5_agent_hooks.py -v`
  -> `41 passed, 1 skipped`.
- Ruff:
  `uv run ruff check automl hooks tests/unit/agent tests/unit/cli tests/unit/validate/test_proposal_validation.py tests/unit/validate/test_model_validation.py tests/e2e/test_phase5_agent_hooks.py`
  -> `All checks passed`.
- Full local gate: `uv run pytest tests/unit tests/contracts tests/integration -v`
  -> `244 passed, 2 warnings`.
- External Phase 5 gate:
  `AUTOML_PHASE5_E2E=1 MLFLOW_TRACKING_URI=http://127.0.0.1:54321 uv run pytest tests/e2e/test_phase5_agent_hooks.py -v`
  -> `1 passed, 31 warnings`.
- Previous external gates preserved:
  `AUTOML_PHASE4_E2E=1 MLFLOW_TRACKING_URI=http://127.0.0.1:54321 uv run pytest tests/e2e/test_phase4_experiment_trial_cleanup.py -v`
  -> `1 passed, 26 warnings`;
  `AUTOML_PHASE3_E2E=1 MLFLOW_TRACKING_URI=http://127.0.0.1:54321 uv run pytest tests/e2e/test_phase3_eval_breadth.py -v`
  -> `1 passed, 19 warnings`.
- Architecture/import ratchets: contracts passed in the full local gate; `rg 'automl_legacy'
  automl projects tests` found only the allowed package docstring/contract text; `rg
  '(^|\s)(import|from) mlflow' automl projects tests` found only `automl/mlflow/client.py`
  and an internal seam import path.

**Post-implementation review fixes landed:**

- `gather_proposer_context().trial_count` now includes unscored trials, not only displayed
  leaderboard rows.
- `build_launch()` now exports `AUTOML_PROJECT`, `AUTOML_EXPERIMENT_ID`, and
  `AUTOML_NAMESPACE`, so `hooks/hooks.json` can bootstrap the thin hook in multi-project repos.
- `hooks/agent_timeline.py` inserts `AUTOML_PROJECT_ROOT`/cwd into `sys.path` before importing
  library code, preserving hook subprocess execution from outside the repo.
- `agents/automl-proposer.md` no longer references dropped `top_trials`, learnings, or artifact
  URI/error packet keys.

**Carryover to Phase 6:**

- Phase 5 intentionally implemented only the A5 CLI wrappers in `automl/cli/__init__.py`.
  Full noun-first CLI catalog, top-level global flag handling, and skill/script command updates
  remain Phase 6.
- Timeline reconciliation is seam-routed and acceptance-covered for hook events, session reports,
  trial `agent/manifest.json`, and agent metrics. Rich transcript/tool-event mining should be
  reviewed during Phase 6/7 if a CLI/skill gate requires the legacy detail level.

---

## Plan/Design Review

This plan is grounded in specs 00/02/04/06/09/10/11, the Phase 4 code, the legacy
`propose/`, `cli/run_loop.py`, `loop_context/proposer_packet.py`, and `hooks/agent_timeline.py`.
The pre-implementation self-review resolves these decisions:

- **A5 is allowed a thin CLI slice.** A5.1/A5.2/A5.3 name CLI verbs, while Phase 6 owns the full
  CLI catalog. Implement only the verbs needed by the A5 gates: `experiment proposer-context`,
  `experiment run`, and `validate proposal`. Do not add project/data/eval/trial catalog breadth
  in Phase 5.
- **`experiment run` is a launcher gate, not an in-process state machine.** `build_launch()` is a
  pure builder and the CLI wrapper executes the returned `claude` command. Tests prove command,
  env, role parsing, and `AUTOML_INHERIT_DRY_RUN`; they do not fake a full Claude-driven coding
  loop as library logic.
- **Timeline relocation is behavior-preserving except the specified boundaries.** Port the
  reconciliation helpers intact into `automl.agent.timeline`, then replace direct PyPI MLflow
  usage, route parsing, and manifest merge with seam calls and a standalone `agent/manifest.json`.
  If the relocation exposes a spec/code mismatch, write a review-found test first and fix only
  that mismatch.
- **Proposal validation uses the dataclass as the field roster.** `Proposal` owns the accepted
  field list; `DISALLOWED=("parent_id",)` is the explicit retired-field list; `proposal_schema()`
  session-resolves allowed dependencies via `project.dependencies.allowed_dependencies`.
- **`SLUG_RE` lands in `utils`.** This is required by spec 10/11 to avoid `trial -> agent` cycles.
- **No direct `automl_legacy` imports and no direct PyPI `mlflow` imports outside
  `automl/mlflow`.** Legacy files are reference only.

### Self-Review Findings Before Code

- **Ambiguity:** The A5.3 wording says "full loop ... runs a trial"; the design says sequencing
  stays LLM-driven and `agent/` does not import `runner`. Decision: A5 proves the library-side
  loop launch command, proposal validation, an existing `runner.run_trial` path, and timeline
  publication/reconciliation around a trial run. It does not implement a driver state machine.
- **Scope creep risk:** The CLI dispatcher can expand quickly. Guardrail: only dispatch the three
  A5 verbs and leave unimplemented verbs with parser errors until Phase 6.
- **Missing-test risk:** Timeline is large. Guardrail: unit tests cover append/event conversion,
  coder-stop trial publish through seam, end-of-session publish through seam/GCS helpers, and the
  thin hook stub. Existing legacy reconciliation logic is ported first, then review tests target
  seam-boundary regressions.
- **Spec contradiction check:** No contradiction found between specs and running evidence. Phase 4
  already applied the leaderboard metric carry-back that proposer context depends on.

## Evidence Read

- `spec/11-agent.md`: defines the exact `agent/` files, public exports, Proposal contract,
  `proposal_schema`, `gather_proposer_context`, `build_launch`, `handle_event`, and `publish`.
- `spec/04-validate.md` plus 11 carry-back: proposal validation is a validate-framework
  orchestrator calling `agent.checks.proposal_schema`; 11 supersedes the older explicit
  `allowed_dependencies` parameter with session-resolved allow-list.
- `spec/02-mlflow-seam.md`: agent writes are loose-tier JSON/metrics through
  `mlflow.trial`/`mlflow.experiment`; agent-events GCS prefix is computed by `_routing.py`.
- Current code: `automl/agent/__init__.py` is empty; `validate.proposal` raises
  `NotImplementedError`; `automl/cli/__init__.py` is empty despite the `automl` console entry;
  root `hooks/agent_timeline.py` is still the legacy 1954-line implementation.
- Legacy source: `automl_legacy/propose/__init__.py` has the proposal checks;
  `automl_legacy/cli/run_loop.py` has `LaunchSpec`/`ClaudeRole`/agent parsing;
  `automl_legacy/loop_context/proposer_packet.py` and `mlflow/store.py::get_context` provide
  proposer packet behavior; `hooks/agent_timeline.py` is the relocation source for timeline.

## File Structure

Create agent domain files:

- `automl/agent/proposal.py`: `Proposal`, `DISALLOWED`, dataclass roster helpers.
- `automl/agent/checks.py`: `proposal_schema(proposal, *, session=None) -> list[Issue]`.
- `automl/agent/proposer_context.py`: `gather_proposer_context`, internal
  `find_prior_experiment`.
- `automl/agent/launch.py`: `LaunchSpec`, `ClaudeRole`, `build_launch`.
- `automl/agent/timeline.py`: relocated reconciliation, library entry points
  `handle_event(payload, *, session=None)` and `publish(session_id, *, session=None)`.
- `automl/agent/__init__.py`: Phase 5 Tier-2 exports.

Modify framework/surface/seam:

- `automl/validate/base.py`: add `schema_version`, `location`, `to_json`, `from_dict`.
- `automl/validate/targets.py`: implement `proposal`.
- `automl/project/dependencies.py`: allowed dependency parsing from `pyproject.toml`.
- `automl/project/__init__.py`: export dependency helpers if needed by CLI/tests.
- `automl/utils/__init__.py` or `automl/utils/slug.py`: expose `SLUG_RE`.
- `automl/mlflow/_routing.py`: add deterministic agent-events prefix helper used by timeline.
- `automl/mlflow/experiment/logging.py` or `automl/mlflow/experiment/__init__.py`: add
  experiment-overview loose JSON logging.
- `automl/cli/__init__.py`, `automl/cli/experiment.py`, `automl/cli/validate.py`: thin A5 CLI
  wrappers only.
- `hooks/agent_timeline.py`: replace with thin stub delegating to `automl.agent.timeline`.

Create tests:

- `tests/unit/agent/test_proposal.py`
- `tests/unit/agent/test_proposal_checks.py`
- `tests/unit/agent/test_proposer_context.py`
- `tests/unit/agent/test_launch.py`
- `tests/unit/agent/test_timeline.py`
- `tests/unit/validate/test_proposal_validation.py`
- `tests/unit/cli/test_agent_phase_cli.py`
- `tests/integration/agent/test_timeline_publish.py`
- `tests/e2e/test_phase5_agent_hooks.py`

## Task DAG

```
P5.0 baseline guard
  -> P5.1 Proposal dataclass + SLUG_RE
  -> P5.2 proposal_schema + validate.proposal
  -> P5.3 proposer context composer
  -> P5.4 launcher builder + narrow experiment run CLI wrapper
  -> P5.5 timeline relocation with seam/GCS routing + hook stub
  -> P5.6 A5 CLI wrappers and e2e gate
  -> P5.7 post-implementation review fixes
  -> P5.8 docs closeout and commit
```

## P5.0 - Baseline Guard

**Files:** no production changes.

**Steps:**
- [x] Run:
  ```bash
  uv run pytest tests/unit tests/contracts tests/integration -v
  ```
  Expected: PASS from the Phase 4 baseline.
- [x] Run:
  ```bash
  uv run pytest tests/contracts -v
  rg 'automl_legacy' automl projects tests
  rg '(^|\s)(import|from) mlflow' automl projects tests
  ```
  Expected: contracts pass; `automl_legacy` appears only in contract/doc text; PyPI `mlflow`
  imports appear only under `automl/mlflow/**` or tests that assert the ratchet.

## P5.1 - Proposal Dataclass And Slug Primitive

**Files:**
- Create: `automl/agent/proposal.py`
- Modify: `automl/agent/__init__.py`
- Create/modify: `automl/utils/slug.py` or `automl/utils/__init__.py`
- Test: `tests/unit/agent/test_proposal.py`

**Steps:**
- [x] Write failing tests for `Proposal.from_dict()` stripping unknown fields, `to_dict()`,
  optional `required_preprocessing`, and `DISALLOWED == ("parent_id",)`.
- [x] Run:
  ```bash
  uv run pytest tests/unit/agent/test_proposal.py -v
  ```
  Expected: FAIL because `automl.agent.proposal` does not exist.
- [x] Implement `Proposal`, `DISALLOWED`, roster helpers, and shared `SLUG_RE`.
- [x] Re-run the same test. Expected: PASS.

## P5.2 - Proposal Schema And Validate Orchestrator

**Files:**
- Create: `automl/agent/checks.py`
- Create: `automl/project/dependencies.py`
- Modify: `automl/validate/base.py`
- Modify: `automl/validate/targets.py`
- Modify: `automl/validate/__init__.py`
- Test: `tests/unit/agent/test_proposal_checks.py`
- Test: `tests/unit/validate/test_proposal_validation.py`

**Steps:**
- [x] Write failing tests covering missing required fields, bad `schema_version`, bad slug,
  non-empty-list rules, bad `seed_hint`, rejected `parent_id`, unknown-field warning,
  dependency allow-list errors, `required_preprocessing` allowed, and canonical
  `ValidationReport.to_json()`/`from_dict()`.
- [x] Run:
  ```bash
  uv run pytest tests/unit/agent/test_proposal_checks.py tests/unit/validate/test_proposal_validation.py -v
  ```
  Expected: FAIL because `proposal_schema`/`validate.proposal` are not implemented.
- [x] Implement `project.dependencies.allowed_dependencies(session)` from root `pyproject.toml`
  project dependencies and dependency groups.
- [x] Implement `agent.checks.proposal_schema()` using `Proposal` fields plus `DISALLOWED`.
- [x] Implement `validate.proposal(*, proposal: dict, session=None) -> ValidationReport`.
- [x] Re-run the same tests. Expected: PASS.

## P5.3 - Proposer Context Composer

**Files:**
- Create: `automl/agent/proposer_context.py`
- Modify: `automl/agent/__init__.py`
- Test: `tests/unit/agent/test_proposer_context.py`

**Steps:**
- [x] Write failing tests with monkeypatched typed sources proving
  `gather_proposer_context()` returns the required packet keys, uses
  `experiment.views.leaderboard`/`recent_failures`/`strategies_attempted`, includes
  `human_trials`, strips `top_trials`/learnings/artifact-error keys, reshapes
  `data_context` to `active_dataset`/`dataset_usage`, and calls `find_prior_experiment`
  only when the current leaderboard is empty.
- [x] Run:
  ```bash
  uv run pytest tests/unit/agent/test_proposer_context.py -v
  ```
  Expected: FAIL because `gather_proposer_context` does not exist.
- [x] Implement `gather_proposer_context()` as a composer over current public domain/seam APIs.
- [x] Re-run the same test. Expected: PASS.

## P5.4 - Launcher Builder

**Files:**
- Create: `automl/agent/launch.py`
- Modify: `automl/agent/__init__.py`
- Test: `tests/unit/agent/test_launch.py`

**Steps:**
- [x] Write failing tests for `LaunchSpec`, `ClaudeRole`, agent YAML parsing, model/effort role
  overrides from `session.config.models`, normalized slash command, `--agents` JSON, cwd,
  `AUTOML_PROJECT_ROOT`, shared `AUTOML_SESSION_ID`/`CLAUDE_SESSION_ID`, and
  `AUTOML_INHERIT_DRY_RUN`.
- [x] Run:
  ```bash
  uv run pytest tests/unit/agent/test_launch.py -v
  ```
  Expected: FAIL because `automl.agent.launch` does not exist.
- [x] Port the launcher builder from legacy `cli/run_loop.py` with the Phase 5 session
  signature.
- [x] Re-run the same test. Expected: PASS.

## P5.5 - Timeline Relocation And Hook Stub

**Files:**
- Create: `automl/agent/timeline.py`
- Modify: `automl/mlflow/_routing.py`
- Modify: `automl/mlflow/experiment/__init__.py`
- Modify/create: experiment-level loose JSON helper as needed.
- Replace: `hooks/agent_timeline.py`
- Test: `tests/unit/agent/test_timeline.py`
- Test: `tests/integration/agent/test_timeline_publish.py`

**Steps:**
- [x] Write failing tests for `handle_event()` appending a route/session event without parsing
  `sys.argv`, reading `AUTOML_INHERIT_DRY_RUN`, returning a JSON-serializable dict, and
  delegating coder-stop publication through seam functions.
- [x] Write failing tests for `publish()` staging `agent/sessions/<session_id>/report.json`,
  logging per-trial JSON/metrics through `mlflow.trial`, logging session JSON through the
  experiment seam, writing standalone `agent/manifest.json`, and using `_routing` for the
  agent-events GCS prefix.
- [x] Run:
  ```bash
  uv run pytest tests/unit/agent/test_timeline.py tests/integration/agent/test_timeline_publish.py -v
  ```
  Expected: FAIL because `automl.agent.timeline` does not exist and the hook is still legacy.
- [x] Mechanically relocate legacy timeline helpers into `automl/agent/timeline.py`.
- [x] Remove route-string/source-of-truth logic that specs kill: `--route`, sys.argv dry-run
  parsing, session-lock route lookup, dynamic `gcs_paths.py` import, direct PyPI `mlflow` import,
  and root manifest merge.
- [x] Replace `hooks/agent_timeline.py` with a thin argparse/stdin stub for `hook-event` and
  `publish` only.
- [x] Re-run the same tests. Expected: PASS.

## P5.6 - Narrow A5 CLI Wrappers And Phase Gate

**Files:**
- Modify: `automl/cli/__init__.py`
- Create: `automl/cli/experiment.py`
- Create: `automl/cli/validate.py`
- Test: `tests/unit/cli/test_agent_phase_cli.py`
- Test: `tests/e2e/test_phase5_agent_hooks.py`

**Steps:**
- [x] Write failing CLI tests for:
  ```bash
  uv run automl validate proposal --json proposal.json --output validated.json
  uv run automl experiment proposer-context --project example_homecredit --json
  uv run automl experiment run --project example_homecredit --max-budget-usd 1 --output-format json
  ```
  using monkeypatch/fakes so no real Claude subprocess is launched.
- [x] Run:
  ```bash
  uv run pytest tests/unit/cli/test_agent_phase_cli.py -v
  ```
  Expected: FAIL because `automl.cli.main` does not exist.
- [x] Implement only the A5 dispatcher branches and JSON output support.
- [x] Add opt-in e2e `tests/e2e/test_phase5_agent_hooks.py` gated by `AUTOML_PHASE5_E2E=1`.
  The gate should validate a proposal, build the launcher command, run one existing Home Credit
  trial path through `runner.run_trial`, append/publish timeline events for that run, and assert
  agent artifacts/metrics are visible through the seam.
- [x] Run:
  ```bash
  uv run pytest tests/unit/cli/test_agent_phase_cli.py -v
  ```
  Expected: PASS.

## P5.7 - Post-Implementation Review

**Files:** all Phase 5 changes.

**Steps:**
- [x] Review code against specs 00/02/04/06/09/10/11 and this plan. Check for layer violations,
  direct `mlflow` imports outside `automl/mlflow`, `automl_legacy` imports, stale `AUTOML_DRY_RUN`,
  leftover route-string parsing, manifest merge, missing `required_preprocessing`, and CLI scope
  creep.
- [x] For each material issue found, write a failing test first in the closest unit/integration
  file, run it to verify failure, then implement the fix and re-run it.

## P5.8 - Verification, Docs, Commit

**Steps:**
- [x] Run targeted tests:
  ```bash
  uv run pytest tests/unit/agent tests/unit/validate tests/unit/cli tests/integration/agent -v
  ```
- [x] Run local gate:
  ```bash
  uv run pytest tests/unit tests/contracts tests/integration -v
  ```
- [x] Run external Phase 5 gate after loading `.env`:
  ```bash
  set -a; source .env; set +a
  AUTOML_PHASE5_E2E=1 MLFLOW_TRACKING_URI=http://127.0.0.1:54321 uv run pytest tests/e2e/test_phase5_agent_hooks.py -v
  ```
- [x] Keep affected previous external gates green:
  ```bash
  AUTOML_PHASE4_E2E=1 MLFLOW_TRACKING_URI=http://127.0.0.1:54321 uv run pytest tests/e2e/test_phase4_experiment_trial_cleanup.py -v
  AUTOML_PHASE3_E2E=1 MLFLOW_TRACKING_URI=http://127.0.0.1:54321 uv run pytest tests/e2e/test_phase3_eval_breadth.py -v
  ```
- [x] Run architecture/import ratchets:
  ```bash
  uv run pytest tests/contracts -v
  rg 'automl_legacy' automl projects tests
  rg '(^|\s)(import|from) mlflow' automl projects tests
  ```
- [x] Update `../README.md`, `plan/README.md`, `implementation-strategy.md` if phase boundaries
  changed, `acceptance-checklist.md`, `migration-checklist.md`, and this phase plan with evidence
  and carryover notes.
- [x] Commit:
  ```bash
  git add automl hooks agents skills tests docs
  git commit -m "Complete Phase 5 agent hooks"
  ```
