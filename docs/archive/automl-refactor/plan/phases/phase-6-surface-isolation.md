# Phase 6 Surface Isolation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development`
> (recommended) or `superpowers:executing-plans` to implement this plan task-by-task.
> Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** pass A6.1-A6.4 by completing the noun-first CLI surface, wiring
session-level `--dry-run`/`--namespace` isolation through command entry points, and updating
skill/plugin commands to the new verbs.

**Architecture:** Phase 6 turns the Phase 5 CLI slice into a thin command layer over the
already-built domain APIs. `automl.cli` becomes a dispatcher with noun modules; project inference
and session bootstrap are centralized once, then each verb calls one library function and serializes
the result. Route isolation remains owned by `Session` + the MLflow seam; CLI wrappers only choose
the session. Skill scripts render the new noun-first commands and stop passing removed hook flags.

**Tech Stack:** Python 3.11 via `uv`; argparse; pytest; fake MLflow/GCS clients for unit and
integration checks; external Home Credit harness with `.env` loaded and
`MLFLOW_TRACKING_URI=http://127.0.0.1:54321` for the opt-in Phase 6 e2e gate.

**Acceptance:** `plan/acceptance-checklist.md` rows **A6.1-A6.4**.

**Baseline evidence before code:** `uv run pytest tests/unit/cli tests/unit/project/test_cleanup.py
tests/unit/mlflow/test_client_and_routing.py tests/integration/cleanup/test_experiment_delete.py -v`
-> `24 passed, 2 warnings`.

---

## Plan/Design Review

This plan is grounded in specs 00/01/02/03/04/05/06/07/09/10/11, the Phase 5 plan closeout, the
current `automl/cli`, `project.session`, `_routing`, cleanup, runner, and skill renderer code, plus
the legacy CLI only for flag compatibility.

Self-review decisions:

- **Global project bootstrap.** Spec 00 only names `--dry-run` and `--namespace` as top-level
  session modifiers, while exact argparse names are explicitly deferred in the implementation
  strategy. Phase 6 will add top-level `--project`, `--project-root`, and `--experiment-id` as
  bootstrap flags so every noun uses one session convention:
  `automl --project example --namespace qa experiment leaderboard --json`. This keeps
  `--dry-run` and `--namespace` top-level as required, removes per-verb routing flags from new
  skill commands, and preserves cwd/single-project inference for human use.
- **`data materialize` is promoted because the skill renderer has a live prep command.** Spec 00's
  deferred list says no skill/CLI use case existed, but `skills/automl/scripts/render_context.py`
  still emits a snapshot preparation command. A6.4 requires skill commands to hit new verbs, so
  Phase 6 adds a thin `automl data materialize` wrapper over `data.materialize()`. This is the
  smallest documented reconciliation of spec text with running evidence.
- **Trial authoring stays carryover.** `safe_commands.create_trial` already names a noun-first
  `automl trial create` command, but the fresh tree does not yet have the trial-authoring domain
  needed for generated trial folders. Phase 6 will not port authoring logic; Phase 7's full e2e
  loop owns that broader workflow. The skill command string remains noun-first, and the phase docs
  will carry the implementation gap.
- **No compatibility aliases for retired verbs.** Legacy `automl run`, `automl inspect`,
  `automl loop-context`, `automl profile`, `automl propose validate`, and `python -m automl.*`
  commands remain retired. Tests and skill docs move to new verbs instead of keeping shims.
- **`experiment run` is still a launcher.** A6.2 checks the command builds a dry-run launch with
  inherited dry-run env and session route. Actual trial execution for external isolation is proven
  through `trial run` and cleanup because launching Claude from the external gate would be
  nondeterministic.
- **Structured output policy.** Verbs with structured values print JSON. `--json` is accepted where
  specified and by skill-facing read/delete verbs, but the wrappers may print JSON by default to
  keep machine output stable.

Ambiguity/risk:

- The specs and current skills are not fully aligned around trial creation. This does not block
  A6 because the command naming can be corrected now; executable generated-trial authoring is a
  Phase 7 gate.
- `validate project` was deferred after the early thin path, but setup/validate skills call it.
  Phase 6 implements a bounded structural validator only: loaded recipe fields, required env
  values, and type/import errors surfaced by `ProjectConfig.load()`. Broader project-specific
  contract breadth remains Phase 7/final audit if open checklist rows require it.

---

## Task DAG

### P6.0 Baseline Guard

**Files:** no edits.

- [x] Run the baseline guard:
  `uv run pytest tests/unit/cli tests/unit/project/test_cleanup.py tests/unit/mlflow/test_client_and_routing.py tests/integration/cleanup/test_experiment_delete.py -v`.
  Expected: PASS before Phase 6 edits.

### P6.1 Write Failing CLI Catalog Tests

**Files:**
- Create: `tests/unit/cli/test_phase6_cli_catalog.py`
- Modify: `tests/unit/cli/test_agent_phase_cli.py`

- [x] Add tests proving root flags are parsed before nouns and passed once to `use_project`:
  `automl --project demo --project-root <tmp> --dry-run --namespace qa --experiment-id exp experiment proposer-context --json`.
- [x] Add tests for the v1 noun catalog wrappers:
  project `list`, `deps`, `init`, `delete`; experiment `list`, `run`, `delete`, `leaderboard`,
  `compare`, `summary`, `proposer-context`; trial `list`, `run`, `show`, `delete`, `lock`;
  data `list`, `profile`, `materialize`; eval `list`, `compute`; validate `project`, `model`,
  `proposal`.
- [x] Add negative tests that retired top-level verbs (`run`, `inspect`, `loop-context`,
  `profile`, `propose`) exit through argparse instead of dispatching.
- [x] Run:
  `uv run pytest tests/unit/cli/test_phase6_cli_catalog.py tests/unit/cli/test_agent_phase_cli.py -v`.
  Expected: FAIL because the Phase 5 CLI only has narrow A5 wrappers and per-verb project flags.

### P6.2 Write Failing Project Metadata, Scaffold, and Validation Tests

**Files:**
- Create: `tests/unit/project/test_metadata_and_scaffold.py`
- Create: `tests/unit/validate/test_project_validation.py`

- [x] Add tests for `project.metadata.list_projects()` and project inference from
  `projects/<name>/...` cwd or a single configured project.
- [x] Add tests for `project.scaffold.create_project()` creating `projects/<name>/config.py`,
  `PROJECT_INSTRUCTIONS.md`, SQL templates, and new import paths (`automl.project`,
  `automl.model`, `automl.data`, `automl.eval`), not `automl.core`.
- [x] Add tests for `validate.project(session=...)`:
  complete config + env passes; missing recipe fields and missing `GCS_BUCKET`/
  `GCS_PREFIX`/`MLFLOW_TRACKING_URI` produce canonical `ValidationReport` issues.
- [x] Run:
  `uv run pytest tests/unit/project/test_metadata_and_scaffold.py tests/unit/validate/test_project_validation.py -v`.
  Expected: FAIL because the metadata/scaffold helpers and project validator do not exist yet.

### P6.3 Implement CLI Foundation and Noun Modules

**Files:**
- Create: `automl/cli/_common.py`
- Create: `automl/cli/project.py`
- Create: `automl/cli/experiment.py`
- Create: `automl/cli/trial.py`
- Create: `automl/cli/data.py`
- Create: `automl/cli/eval.py`
- Create: `automl/cli/validate.py`
- Create: `automl/cli/__main__.py`
- Modify: `automl/cli/__init__.py`

- [x] Implement a root parser with top-level `--project`, `--project-root`, `--dry-run`,
  `--namespace`, and `--experiment-id` before the noun.
- [x] Centralize session construction in `automl.cli._common.session_from_args()`, using project
  inference when `--project` is omitted.
- [x] Centralize `_jsonable()` and `print_json()` for dataclasses, `to_dict()`, enums,
  mappings, and lists.
- [x] Implement thin noun modules:
  - project: `list`, `deps`, `init`, `delete`
  - experiment: `list`, `run`, `delete`, `leaderboard`, `compare`, `summary`, `proposer-context`
  - trial: `list`, `run`, `show`, `delete`, `lock acquire|release`
  - data: `list`, `profile`, `materialize`
  - eval: `list`, `compute`
  - validate: `project`, `model`, `proposal`
- [x] Re-run:
  `uv run pytest tests/unit/cli/test_phase6_cli_catalog.py tests/unit/cli/test_agent_phase_cli.py -v`.
  Expected: PASS.

### P6.4 Implement Project Metadata, Scaffold, Validation, and Eval Listing

**Files:**
- Create: `automl/project/metadata.py`
- Create: `automl/project/scaffold.py`
- Create: `automl/eval/registry.py`
- Modify: `automl/project/__init__.py`
- Modify: `automl/eval/__init__.py`
- Modify: `automl/validate/targets.py`

- [x] Implement `list_projects(repo_root=None)`, `infer_project_name(repo_root=None, start=None)`,
  and `project_metadata(...)` over `projects/*/config.py`.
- [x] Implement `create_project(project_name, *, project_root, template="snowflake")` with the
  new four-layer import paths and lower-snake validation.
- [x] Implement bounded `validate.project(session=None)` structural checks.
- [x] Implement `eval.list_eval_datasets(session=None)` by listing routed eval dataset manifest
  prefixes and reading manifests via `utils.io.gcs`.
- [x] Re-run:
  `uv run pytest tests/unit/project/test_metadata_and_scaffold.py tests/unit/validate/test_project_validation.py tests/unit/cli/test_phase6_cli_catalog.py -v`.
  Expected: PASS.

### P6.5 Write and Pass Isolation/E2E Gates

**Files:**
- Create: `tests/e2e/test_phase6_surface_isolation.py`
- Modify: `tests/unit/cli/test_phase6_cli_catalog.py`

- [x] Add unit tests proving `automl --dry-run experiment run` passes a dry-run session into
  `build_launch()` and that launcher env includes `AUTOML_INHERIT_DRY_RUN=1`.
- [x] Add unit tests proving delete wrappers pass `namespace` and `dry_run` sessions into cleanup
  and do not expose per-verb `--route`, `--route-namespace`, or `--dry-run` variants.
- [x] Add opt-in external e2e:
  - create one real namespace experiment and one `dry_run` experiment;
  - materialize and run one trial in each through CLI/library surfaces;
  - delete only the dry-run experiment and assert real MLflow/GCS artifacts remain;
  - run a full-fidelity `--namespace qa` trial, then `automl --namespace qa experiment delete
    <id> --apply`, and assert the real namespace is untouched;
  - include the composed `qa/dry_run/...` prefix check.
- [x] Run local targeted tests:
  `uv run pytest tests/unit/cli tests/unit/project tests/unit/validate tests/unit/eval tests/integration/cleanup/test_experiment_delete.py -v`.
  Expected: PASS.
- [x] Run external Phase 6 gate after loading `.env`:
  `set -a; source .env; set +a; AUTOML_PHASE6_E2E=1 MLFLOW_TRACKING_URI=http://127.0.0.1:54321 uv run pytest tests/e2e/test_phase6_surface_isolation.py -v`.
  Expected: PASS.

### P6.6 Update Skill Scripts, Agents, and Contract Docs

**Files:**
- Modify: `skills/automl/scripts/render_context.py`
- Modify: `skills/automl/scripts/preflight.py`
- Modify: `skills/automl/SKILL.md`
- Modify: `skills/automl-guide/SKILL.md`
- Modify: `skills/coder/SKILL.md`
- Modify: `skills/propose/SKILL.md`
- Modify: `skills/inspect/SKILL.md`
- Modify: `skills/profile/SKILL.md`
- Modify: `skills/setup/SKILL.md`
- Modify: `skills/validate/SKILL.md`
- Modify: `agents/automl-coder.md`
- Modify: `agents/automl-proposer.md`
- Modify: `references/setup/model-contract.md`
- Modify: `references/loop/mlflow-context.md`
- Modify: `references/loop/protocol.md`
- Modify: `references/implement/dependencies.md`
- Create/modify: `tests/contracts/test_phase6_skill_commands.py`

- [x] Add contract tests scanning skills/references/agents for retired executable commands:
  `automl loop-context`, `automl propose validate`, `python -m automl.core.dependencies`,
  `python -m automl.session.lock`, top-level `automl inspect`, top-level `automl profile`,
  `automl project create`, hook `--route`, and hook `--publish-mlflow`.
- [x] Update `render_context.py` to use new project APIs, render top-level session flags, call
  `experiment proposer-context`, `validate proposal`, `data materialize`, and hook `publish`
  without removed flags.
- [x] Update model contract docs and coder/proposer prompts from `automl.core.*` and
  snapshot wording to `automl.model.BaseModel`, `automl.data.FeatureRegistry`, dataset wording,
  and the required-preprocessing handoff.
- [x] Run:
  `uv run pytest tests/contracts/test_phase6_skill_commands.py -v`.
  Expected: PASS.

### P6.7 Post-Implementation Code Review Pass

**Files:** all Phase 6 touched files.

- [x] Review against specs 00/01/02/03/04/05/06/07/09/10/11 and this plan. Check for:
  stale legacy verbs, per-verb dry-run/namespace leakage, direct PyPI MLflow imports outside
  `automl/mlflow`, `automl_legacy` imports, trial-authoring scope creep, unsafe cleanup routing,
  docs still naming old commands, and missing `--json` structured outputs.
- [x] For each material finding, write a failing test first in the closest unit/contract/e2e
  location, then fix.
- [x] Re-run the targeted Phase 6 tests after every fix.

### P6.8 Verification, Docs, and Commit

**Files:**
- Modify: `docs/superpowers/automl-refactor/README.md`
- Modify: `docs/superpowers/automl-refactor/plan/README.md`
- Modify: `docs/superpowers/automl-refactor/plan/implementation-strategy.md`
- Modify: `docs/superpowers/automl-refactor/plan/acceptance-checklist.md`
- Modify: `docs/superpowers/automl-refactor/plan/migration-checklist.md`
- Modify: this phase plan with closeout evidence and carryover.

- [x] Run targeted Phase 6 gate:
  `uv run pytest tests/unit/cli tests/unit/project tests/unit/validate tests/unit/eval tests/contracts/test_phase6_skill_commands.py tests/integration/cleanup/test_experiment_delete.py tests/e2e/test_phase6_surface_isolation.py -v`
  without external env. Expected: PASS with external test skipped unless `AUTOML_PHASE6_E2E=1`.
- [x] Run full local gate:
  `uv run pytest tests/unit tests/contracts tests/integration -v`.
- [x] Run architecture/import ratchets:
  `uv run pytest tests/contracts/test_architecture.py -v`;
  `rg 'automl_legacy' automl projects tests`;
  `rg '(^|\s)(import|from) mlflow' automl projects tests`.
- [x] Run external gates with `.env` loaded and
  `MLFLOW_TRACKING_URI=http://127.0.0.1:54321`: Phase 6 plus affected Phase 5, Phase 4, and
  Phase 3 e2e gates.
- [x] Update acceptance/migration checklists and root/plan docs with exact commands/results.
- [x] Commit:
  `git add ... && git commit -m "Complete Phase 6 surface isolation"`.

## Post-Implementation Review

Review completed after the first implementation pass, using specs 00/01/02/03/04/05/06/07/09/10/11,
the acceptance rows, and the touched CLI/agent/skill/cleanup files.

Findings fixed with tests first:

- `validate proposal` incorrectly skipped session context whenever project inference failed, even
  when the caller explicitly supplied session selectors. Added CLI tests for inferred/no-session
  and explicit-session behavior, then changed `_optional_session()` to re-raise explicit bootstrap
  failures.
- `automl trial lock` was initially a no-op-style JSON wrapper rather than real lock behavior.
  Added runner session-lock unit tests and CLI coverage, then implemented `runner/session_lock.py`
  with acquire/release/is_locked/session-lock context APIs and wired `trial lock acquire|release`.
- The external Phase 6 gate hit MLflow's run-URL stdout lines before CLI JSON, so the test parsed
  the wrong stdout prefix. The runtime command already emitted valid trailing JSON; the e2e harness
  now scans stdout for the JSON object instead of assuming the entire stream is JSON.

No spec-blocking contradictions were found. The documented carryover remains trial authoring
(`trial create`/`fork`/`promote`) for Phase 7.

## Verification Evidence

- Baseline guard before code:
  `uv run pytest tests/unit/cli tests/unit/project/test_cleanup.py tests/unit/mlflow/test_client_and_routing.py tests/integration/cleanup/test_experiment_delete.py -v`
  -> `24 passed, 2 warnings`.
- Targeted Phase 6/code-review sweep:
  `uv run pytest tests/unit/cli tests/unit/project tests/unit/validate tests/unit/eval tests/unit/agent/test_launch.py tests/unit/runner/test_session_lock.py tests/contracts/test_phase6_skill_commands.py tests/integration/cleanup/test_experiment_delete.py tests/e2e/test_phase6_surface_isolation.py -v`
  -> `108 passed, 1 skipped, 2 warnings`.
- Ruff:
  `uv run ruff check automl hooks tests/unit/agent tests/unit/cli tests/unit/project tests/unit/validate tests/unit/runner/test_session_lock.py tests/contracts/test_phase6_skill_commands.py tests/e2e/test_phase6_surface_isolation.py`
  -> `All checks passed`.
- Contracts:
  `uv run pytest tests/contracts -v` -> `11 passed`.
- Full local gate:
  `uv run pytest tests/unit tests/contracts tests/integration -v` -> `273 passed, 2 warnings`.
- Import ratchets:
  `rg -n '(^|\s)(import|from) mlflow' automl projects tests -g '*.py'` -> only
  `automl/mlflow/client.py` imports PyPI `mlflow`;
  `rg -n 'automl_legacy' automl projects tests -g '*.py'` -> only the top-level package docstring
  and architecture contract text.
- External gates with `.env` loaded and `MLFLOW_TRACKING_URI=http://127.0.0.1:54321`:
  Phase 6 -> `1 passed, 29 warnings`;
  Phase 5 -> `1 passed, 31 warnings`;
  Phase 4 -> `1 passed, 26 warnings`;
  Phase 3 -> `1 passed, 19 warnings`.

## Carryover to Phase 7

- Finish the trial-authoring domain (`trial create`, `fork`, `promote`, packaging/source extraction)
  before the final agent-loop cutover gate depends on generated trial folders.
- Close the remaining migration-checklist rows with explicit `[x]` or `[-]` dispositions rather
  than broadening Phase 6 after the A6 gate.
- Re-run the whole-refactor spec/migration/architecture/docs audit after Phase 7 commit before
  declaring the refactor complete.
