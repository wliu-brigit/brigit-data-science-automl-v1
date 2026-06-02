# Phase 7 Cutover Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development`
> (recommended) or `superpowers:executing-plans` to implement this plan task-by-task.
> Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** pass A7.1-A7.4 by completing the final trial-authoring gap, proving the
final agent-loop harness, disposing every migration-checklist row, and deleting the frozen
legacy trees.

**Architecture:** Phase 7 keeps the same thin-layer shape: `trial/` authors local trial
folders and metadata, `runner/` executes a verified trial folder or the committed project-model
fallback, `mlflow/` remains the only PyPI-MLflow importer, and `cli/` only wraps library verbs.
The cutover is a clean deletion of `automl_legacy/` and `tests_legacy/`; no bridge, shim, or
dual-read path is introduced.

**Tech Stack:** Python 3.11 via `uv`; argparse; pytest; local fake/seam tests; external Home
Credit harness with `.env` loaded and `MLFLOW_TRACKING_URI=http://127.0.0.1:54321`.

**Acceptance:** `plan/acceptance-checklist.md` rows **A7.1-A7.4**.

**Baseline before code:** Phase 6 commit `4e174c5` is green. Evidence from closeout:
`uv run pytest tests/unit tests/contracts tests/integration -v` -> `273 passed, 2 warnings`;
Phase 6/5/4/3 external gates each passed against `http://127.0.0.1:54321`.

---

## Plan/Design Review

This plan is grounded in specs 00/01/02/03/04/05/06/07/08/09/10/11, Phase 6 closeout,
the open migration rows, current `trial/`, `runner/`, `mlflow/trial/artifacts`, CLI modules,
project fixtures, and the frozen legacy trial-authoring source.

Self-review decisions:

- **Trial authoring is now in scope.** Structural spec 00 deferred `trial create`/`fork`/`promote`
  until real demand. A7.3 creates that demand: the final loop gate must run a generated trial
  folder, not only the committed project-model fallback used by Phases 1-6.
- **Runner keeps the project-model fallback.** `run_trial("example_homecredit", session=...)`
  is a proven preservation path used by earlier external gates. Phase 7 adds folder execution
  when the argument is an existing path; it does not remove the fallback.
- **Seeded forks are clean-cut.** Phase 7 writes and reads new source artifacts for runs created
  from trial folders. Seed selection may use only new-format runs with a recoverable
  `source/model.py`; older Phase 1-6 runs have no source artifact and should fail clearly when
  selected as a seed. This follows the no-back-compat persisted-state rule.
- **Final e2e is deterministic.** The gate exercises propose -> validate -> create -> implement
  (copy/package model source) -> run -> eval -> leaderboard -> timeline publication. It does not
  launch a live Claude subprocess, because an interactive LLM driver is nondeterministic and was
  already covered as a launch-spec surface in Phase 5.
- **Checklist closure is evidence-based.** Rows that are implemented get `[x]`; rows explicitly
  out of scope per specs get `[-]`; truly debatable items go to the final-review open-items doc
  during the post-Phase-7 audit, not silently forced during cutover.

Ambiguity/risk:

- Existing project `payment_routing` still imports legacy module paths. Phase 7 will update it to
  the new four-layer imports while preserving its placeholder configuration semantics.
- Deleting `tests_legacy/` is not spelled out in A7.4, but it is the frozen legacy test tree from
  Phase 0. Keeping it after `automl_legacy/` deletion would leave stale legacy debt in the final
  package. Phase 7 deletes both frozen trees.

Self-review outcome before implementation: no blockers. The only material scope expansion is
trial authoring, and it is justified by A7.3. The live-driver e2e limitation is documented and
kept deterministic. Placeholder scan found no unresolved implementation placeholders; the
`<TBD_...>` strings mentioned later are intentional project-fixture placeholders.

---

## Task DAG

### P7.0 Baseline Guard

**Files:** no edits.

- [x] Run:
  `uv run pytest tests/unit/trial tests/unit/runner tests/unit/cli/test_phase6_cli_catalog.py tests/contracts -v`.
  Expected: PASS before Phase 7 edits.
- [x] Run the migration-row inventory:
  `rg -n '\[ \]|\[/\]|\[\?\]' docs/superpowers/automl-refactor/plan/migration-checklist.md`.
  Expected: rows remain; this is the A7.1 red inventory.

### P7.1 Write Failing Trial-Authoring Tests

**Files:**
- Create: `tests/unit/trial/test_authoring.py`
- Create: `tests/unit/runner/test_trial_folder_execution.py`
- Modify: `tests/unit/cli/test_phase6_cli_catalog.py`

- [x] Add tests for `TrialMetadata`, `SeedSelection`, `ModelSource`, `TimingReport`, and
  `TrialManifest` `from_dict()`/`to_dict()` behavior. The metadata assertion must prove no
  `run_mode` or `dry_run` field is serialized.
- [x] Add tests for `package_model()` writing imports, class source, and `Model = <ClassName>`
  alias when the source class is not named `Model`.
- [x] Add tests for `trial.create()` creating:
  `projects/<project>/experiments/<namespace>/dry_run/<project>/<experiment>/<slug>/`,
  `model.py`, `metadata.json`, `proposal/proposal.json`, and `run.py`.
- [x] Add tests for `fork()` delegating to `create(seed=..., training_origin="human")` and
  `promote()` creating a folder then calling `runner.run_trial(trial_dir, session=...)`.
- [x] Add runner tests proving an existing path is verified with `runner.paths.verify_trial_dir`,
  loads `<trial_dir>/model.py`, reads metadata slug/strategy/hypothesis/training_origin, and
  rejects a path outside the session route.
- [x] Add CLI tests for `automl trial create`, `trial fork`, and `trial promote`.
- [x] Run:
  `uv run pytest tests/unit/trial/test_authoring.py tests/unit/runner/test_trial_folder_execution.py tests/unit/cli/test_phase6_cli_catalog.py -v`.
  Expected: FAIL because the authoring modules and runner path execution do not exist.

### P7.2 Implement Trial Authoring and Source Packaging

**Files:**
- Create: `automl/trial/metadata.py`
- Create: `automl/trial/packaging.py`
- Create: `automl/trial/create.py`
- Create: `automl/trial/fork.py`
- Create: `automl/trial/promote.py`
- Create: `automl/runner/template.py`
- Modify: `automl/trial/__init__.py`
- Modify: `automl/mlflow/tags.py`
- Modify: `automl/mlflow/trial/artifacts/model.py`
- Modify: `automl/mlflow/trial/artifacts/__init__.py`

- [x] Implement frozen metadata dataclasses with `schema_version: int = 1`, `from_dict()`,
  and `to_dict()`. `TrialMetadata.to_dict()` must omit `run_mode`/`dry_run`.
- [x] Port `package_model()` from legacy `trial/packaging.py` into `automl/trial/packaging.py`
  using stdlib inspection only.
- [x] Implement `create(slug, strategy, *, hypothesis="", seed=None, model_source=None,
  training_origin="automl", proposal=None, session=None) -> Path`, using `runner.paths.trial_dir()`
  and `utils.slug.SLUG_RE`.
- [x] Implement seed resolution against new-format source artifacts:
  choose best/latest/strategy via `mlflow.experiment` summary queries, then read
  `source/model.py` via the MLflow trial artifact seam. If the selected run lacks source, raise
  `FileNotFoundError` with the run id and selector.
- [x] Implement `fork()` as `create(..., seed=seed, training_origin="human")`.
- [x] Implement `promote()` as `create(..., model_source=model_path, training_origin="human")`
  followed by `runner.run_trial(trial_dir, session=session)`.
- [x] Add `runner/template.py` as a copied `run.py` shim that calls
  `automl.runner.run_trial(Path(__file__).parent)`.
- [x] Add `write_model_source()` / `load_model_source()` to the MLflow trial artifact seam,
  storing `source/model.py` through the existing GCS-or-MLflow byte writer and tagging the URI.
- [x] Re-run:
  `uv run pytest tests/unit/trial/test_authoring.py -v`.
  Expected: PASS.

### P7.3 Implement Runner Folder Execution

**Files:**
- Modify: `automl/runner/trial.py`
- Modify: `automl/runner/paths.py`
- Modify: `tests/integration/runner/test_one_trial_local.py`

- [x] Introduce a small execution context inside `runner/trial.py` carrying
  `session`, `trial_dir`, `metadata`, and `model_source_path`.
- [x] When `path_or_project` is an existing path, resolve/bind the session, verify the path is
  under `runner.paths.route_root(session)`, load `metadata.json`, and import `model.py` from the
  trial folder.
- [x] When `path_or_project` is not an existing path, preserve the Phase 1-6 project-model
  fallback.
- [x] Use metadata values for the MLflow run slug/strategy and set hypothesis/training-origin
  tags; keep trial-number and trial-id assignment at execution time.
- [x] If executing a trial folder, write `source/model.py` through the artifact seam before
  returning a successful result.
- [x] Re-run:
  `uv run pytest tests/unit/runner/test_trial_folder_execution.py tests/integration/runner/test_one_trial_local.py -v`.
  Expected: PASS.

### P7.4 Wire CLI Trial Authoring

**Files:**
- Modify: `automl/cli/trial.py`
- Modify: `tests/unit/cli/test_phase6_cli_catalog.py`
- Modify: `skills/automl/scripts/render_context.py` if generated safe commands need argument shape updates.

- [x] Add `trial create <slug> --strategy <strategy> --hypothesis <text>
  [--model-source <path>] [--proposal-json <path>] [--json]`.
- [x] Add `trial fork <slug> [--seed best|latest|strategy:<name>] [--strategy <strategy>]
  [--hypothesis <text>] [--json]`.
- [x] Add `trial promote <slug> --model-path <path> --hypothesis <text>
  [--strategy <strategy>] [--json]`.
- [x] Keep root `--project`/`--project-root`/`--dry-run`/`--namespace`/`--experiment-id` as the only session selectors.
- [x] Re-run:
  `uv run pytest tests/unit/cli/test_phase6_cli_catalog.py tests/contracts/test_phase6_skill_commands.py -v`.
  Expected: PASS.

### P7.5 Add Final Cutover Contract and Testpath Ratchet

**Files:**
- Modify: `tests/contracts/test_pytest_structure.py`
- Modify: `tests/contracts/test_architecture.py`
- Modify: `pyproject.toml`

- [x] Add a contract asserting `automl_legacy/` and `tests_legacy/` do not exist once A7.4 lands.
- [x] Add a contract asserting `pyproject.toml` testpaths include `tests/unit`,
  `tests/contracts`, `tests/integration`, and `tests/e2e`.
- [x] Update `pyproject.toml` testpaths accordingly; `tests/e2e` tests remain opt-in/skip-gated
  unless their phase environment variables are set.
- [x] Run:
  `uv run pytest tests/contracts/test_pytest_structure.py tests/contracts/test_architecture.py -v`.
  Expected before deletion: FAIL on legacy-tree existence; after P7.8: PASS.

### P7.6 Add Final E2E Agent-Loop Harness

**Files:**
- Create: `tests/e2e/test_phase7_cutover.py`

- [x] Add an opt-in test gated by `AUTOML_PHASE7_E2E=1`, `GCS_BUCKET`, `GCP_PROJECT`, and
  `MLFLOW_TRACKING_URI`.
- [x] In the test: load `example_homecredit` in a unique namespace/experiment, materialize data,
  gather proposer context, validate a proposal payload, build the launch spec, create a trial
  folder from the validated proposal and committed model source, run it through the runner,
  assert eval metrics and leaderboard include the run, publish a synthetic agent timeline event,
  then clean only that namespace.
- [x] Run without the env flag:
  `uv run pytest tests/e2e/test_phase7_cutover.py -v`.
  Expected: SKIPPED.
- [x] Run with `.env` loaded:
  `set -a; source .env; set +a; AUTOML_PHASE7_E2E=1 MLFLOW_TRACKING_URI=http://127.0.0.1:54321 uv run pytest tests/e2e/test_phase7_cutover.py -v`.
  Expected: PASS.

### P7.7 Migration Checklist and Project Fixture Closure

**Files:**
- Modify: `docs/superpowers/automl-refactor/plan/migration-checklist.md`
- Modify: `projects/payment_routing/config.py`
- Modify: root `README.md`, `CLAUDE.md`, `.env.example`, and project/reference docs if scans show stale legacy behavior.

- [x] Update `projects/payment_routing/config.py` to import from `automl.project`, `automl.data`,
  and `automl.eval`; preserve its `<TBD_...>` placeholders.
- [x] Audit every remaining `[ ]`, `[/]`, and `[?]` row in `migration-checklist.md`.
  Mark implemented rows `[x]`; mark spec-deferred/out-of-scope rows `[-]` with a concise reason.
- [x] Add a contract or targeted unit test first for any row that exposes a real behavior gap.
- [x] Run stale scans:
  `rg -n 'from automl\.core|import automl\.core|automl_legacy|tests_legacy|python -m automl\.|automl run|automl inspect|automl loop-context|automl profile|automl propose validate|automl project create' README.md CLAUDE.md .env.example docs references skills agents projects -g '*.md' -g '*.py'`.
  Expected: only historical refactor-doc references or no matches after docs are updated.

### P7.8 Cutover Delete

**Files:**
- Delete: `automl_legacy/`
- Delete: `tests_legacy/`
- Modify: `pyproject.toml` packaging comments if they still mention excluding `automl_legacy`.
- Modify: `tests/contracts/test_architecture.py` if its ignore list still references deleted legacy trees.

- [x] Run:
  `git rm -r automl_legacy tests_legacy`.
- [x] Remove stale package-exclude comments now that the deleted tree is not packaged by default.
- [x] Re-run:
  `uv run pytest tests/contracts/test_pytest_structure.py tests/contracts/test_architecture.py -v`.
  Expected: PASS.

### P7.9 Post-Implementation Code Review Pass

**Files:** all Phase 7 touched files.

- [x] Review against specs 00-11, this plan, acceptance A7.1-A7.4, and the migration checklist.
  Check for: trial path isolation leaks, direct PyPI MLflow imports outside `automl/mlflow`,
  new imports from deleted legacy trees, runner path fallback regressions, stale skill commands,
  docs claiming cutover before verification, and seed/fork behavior that silently uses old-format runs.
- [x] For each material finding, write a failing unit/contract/e2e test first, then fix.
- [x] Re-run the closest targeted suite after each fix.

### P7.10 Verification, Docs, and Commit

**Files:**
- Modify: `docs/superpowers/automl-refactor/README.md`
- Modify: `docs/superpowers/automl-refactor/plan/README.md`
- Modify: `docs/superpowers/automl-refactor/plan/implementation-strategy.md`
- Modify: `docs/superpowers/automl-refactor/plan/acceptance-checklist.md`
- Modify: `docs/superpowers/automl-refactor/plan/migration-checklist.md`
- Modify: this phase plan with evidence and carryover.

- [x] Run targeted Phase 7 gate:
  `uv run pytest tests/unit/trial tests/unit/runner tests/unit/cli tests/contracts tests/integration/runner tests/e2e/test_phase7_cutover.py -v`
  without external env. Expected: PASS with Phase 7 e2e skipped.
- [x] Run full local gate:
  `uv run pytest tests/unit tests/contracts tests/integration -v`.
- [x] Run default local gate:
  `uv run pytest -v`.
- [x] Run ruff:
  `uv run ruff check automl hooks projects/payment_routing/config.py projects/example_homecredit/config.py projects/example_homecredit/model tests/unit/trial tests/unit/runner/test_trial_folder_execution.py tests/unit/cli/test_phase6_cli_catalog.py tests/unit/mlflow/test_trial_artifacts.py tests/contracts tests/e2e/test_phase7_cutover.py`.
- [x] Run import ratchets:
  `rg -n '(^|\s)(import|from) mlflow' automl projects tests -g '*.py'`;
  `rg -n 'automl_legacy|tests_legacy' automl projects tests -g '*.py'`.
- [x] Run external gates with `.env` loaded and `MLFLOW_TRACKING_URI=http://127.0.0.1:54321`:
  Phase 7 plus Phase 6, Phase 5, Phase 4, and Phase 3 preservation gates.
- [x] Update closeout docs and commit:
  `git add ... && git commit -m "Complete Phase 7 cutover"`.

## Post-Implementation Review Findings

- **CLI shape mismatch:** the first implementation used `--slug` for trial authoring verbs.
  The plan/spec surface is noun-first (`trial create <slug>`, `trial fork <slug>`,
  `trial promote <slug>`). Added a failing CLI dispatch assertion, then changed the parser to
  positional slugs.
- **Runner/trial domain boundary:** a review ratchet caught `runner` importing
  `automl.trial.metadata` at runtime. The runner now reads only a private minimal metadata view
  from JSON, preserving the orchestration/domain boundary.
- **Active-session leakage:** full local pytest caught `trial.create()` tests leaving the
  contextvar session active. The test now clears the session after creation.
- **Model-source external assertion:** the first Phase 7 e2e asserted direct GCS model-source
  writes appeared in raw MLflow artifact listing. The root cause was a test-contract mismatch:
  source is intentionally tag-backed through `automl.trial.model_source.uri`. The e2e now asserts
  `load_model_source()`, the source URI tag, and GCS object existence.
- **Standard module decorators:** review considered dynamic module loading risk for decorated
  trial models. A unit test now covers `@dataclass` models; the current loader already supports
  the case.

## Closeout Evidence

- Targeted Phase 7 gate:
  `uv run pytest tests/unit/trial tests/unit/runner tests/unit/cli tests/contracts tests/integration/runner tests/e2e/test_phase7_cutover.py -v`
  -> `58 passed, 1 skipped, 3 warnings`.
- Full local gate:
  `uv run pytest tests/unit tests/contracts tests/integration -v`
  -> `283 passed, 2 warnings`.
- Default local gate:
  `uv run pytest -v`
  -> `283 passed, 7 skipped, 2 warnings`.
- Contract gate:
  `uv run pytest tests/contracts -v`
  -> `13 passed`.
- Ruff:
  `uv run ruff check automl hooks projects/payment_routing/config.py projects/example_homecredit/config.py projects/example_homecredit/model tests/unit/trial tests/unit/runner/test_trial_folder_execution.py tests/unit/cli/test_phase6_cli_catalog.py tests/unit/mlflow/test_trial_artifacts.py tests/contracts tests/e2e/test_phase7_cutover.py`
  -> `All checks passed`.
- Import ratchets:
  `rg -n '(^|\s)(import|from) mlflow' automl projects tests -g '*.py'`
  -> only `automl/mlflow/client.py` imports PyPI `mlflow`; other hits are `automl.mlflow.*`.
  `rg -n 'automl_legacy|tests_legacy' automl projects tests -g '*.py'`
  -> only contract-test string literals.
- External gates with `.env` loaded and
  `MLFLOW_TRACKING_URI=http://127.0.0.1:54321`:
  Phase 7 -> `1 passed`; Phase 6 -> `1 passed`; Phase 5 -> `1 passed`;
  Phase 4 -> `1 passed`; Phase 3 -> `1 passed`.

## Final Audit Follow-Up

- [x] Re-read specs 00-11 after the Phase 7 commit and ran the user-requested whole-refactor
  review.
- [x] Fixed obvious findings with tests/contracts first: retired `AUTOML_DRY_RUN` guidance,
  stale `route_namespace` naming, snapshot-era active skill/reference docs, migration-note status
  drift, and one widened-ruff cleanup in an integration test.
- [x] Re-ran full local, contracts, widened ruff, import/stale-surface scans, and Phase 7/6/5/4/3
  external gates.
- [x] No policy-like or debatable finding remains; see
  `docs/superpowers/automl-refactor/plan/final-review-open-items.md`.
