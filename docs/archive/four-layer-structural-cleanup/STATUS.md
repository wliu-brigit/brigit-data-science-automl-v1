# Execution Status — Four-Layer Structural Cleanup

**Last updated:** 2026-05-30 (Wave E complete; cleanup finished)
**Current wave:** COMPLETE
**Overall:** 5 / 5 waves complete

> Update rules are in [`README.md`](README.md) → "Self-driving execution
> protocol." TL;DR: execute Wave A autonomously (pre-approved); for each later
> wave, author its plan then **pause for user approval before executing it**
> (rule 4a); within an approved wave keep moving task-by-task with targeted
> commits; update STATUS per wave, blocker, or handoff rather than per task; also
> pause to raise a concern or get an intent decision you can't make from
> code/tests (rule 5).

---

## Handoff log (most recent first)

- **2026-05-30** — Post-Wave E environment surface cleanup completed. Removed
  migration-only env surfaces from active code/docs/tests: the auto-confirm env
  alias, the domain-specific e2e gates (collapsed to `AUTOML_E2E=1`, with
  `AUTOML_E2E_NOTEBOOKS=1` kept separate for notebook/agent execution), and the
  hidden MLflow hard-delete backend/artifacts envs. MLflow hard delete now takes
  explicit `--backend-store-uri` / `--artifacts-destination` options instead of
  hidden env/config state. Kept real service/runtime envs
  such as `GCS_BUCKET`, `GCS_PREFIX`, `GCP_PROJECT`, `MLFLOW_TRACKING_URI`,
  MLflow auth envs, launcher transport envs, and project-specific Kaggle/Snowflake
  envs. Verification: targeted cleanup/e2e suite → `56 passed, 8 skipped, 10
  warnings`; `uv run pytest tests/unit tests/integration tests/contracts -q` →
  `457 passed, 14 warnings`; collection → `465 tests collected`; retired-env
  grep over active code/docs/tests/notebooks/skills returned no matches.
- **2026-05-30** — Post-Wave E Home Credit notebook 2 live agent smoke completed
  with `RUN_AGENT=True` and `opus/high` routes in dry-run mode. The agent
  produced trial `5_example_notebook_2_elasticnet` and MLflow run
  `f98e7481e8394f7c9babd81cf3dbcffc` under
  `dry_run/example_homecredit/example-homecredit`, status `FINISHED`, AUC
  `0.5`. Active code/test/doc surfaces were cleaned of temporary numbered-stage
  terminology and retired snapshot-era paths; config loading now requires
  `PROJECT_CONFIG`, and retired `RunConfig(split=...)` compatibility was
  removed. Historical execution-plan text was left intact.
- **2026-05-30** — Wave E complete, including live Home Credit notebook e2e.
  Gate evidence: `uv run pytest tests/contracts -q` → `45 passed, 10
  warnings`; `uv run pytest tests/unit tests/integration tests/contracts -q` →
  `455 passed, 14 warnings`; `uv run pytest -m "unit or contract" tests/unit
  tests/contracts -q` → `422 passed, 12 warnings`; `set -a; source .env; set
  +a; AUTOML_E2E_NOTEBOOKS=1 uv run pytest
  tests/e2e/test_homecredit_notebooks.py -q -rs` → `1 passed, 77 warnings`;
  `uv run pytest tests/e2e -q` → `8 skipped, 10 warnings`; README/pyproject
  smoke printed `python version docs ok`; `git diff --check` and e2e
  retired-stage-token grep were clean; `uv run pytest --collect-only -q tests/unit
  tests/contracts tests/integration tests/e2e` → `463 tests collected`. Final
  Wave E review approved with no blockers and no unintended MLflow/GCS artifact
  schema drift.
- **2026-05-30** — Wave E plan amended after user review. Key changes:
  removed the proposed `DataPipeline.load_training_data` addition and aligned
  docs to the existing named split surface (`data.load_dataset(split_name=...)`);
  notebook work now includes an opt-in full e2e execution test in addition to
  static facade contracts; e2e file/function/env gate names now remove temporary
  phase terminology; README Python guidance must follow `pyproject.toml`;
  `TrialProposal` prose still moves to `Proposal`. Execution remains paused at
  the rule-4a plan gate.
- **2026-05-30** — Wave E detailed plan authored in
  `docs/execution/cleanup-plan.md` under "Wave E — Docs/notebook truth +
  test-tier durability — DETAILED". Execution is paused at the rule-4a plan
  gate. This original approval-point list was superseded by the Wave E plan
  amendment above.
- **2026-05-30** — Wave D complete and gate green. Gate evidence:
  `uv run pytest tests/unit tests/integration tests/contracts -q` → `433
  passed, 14 warnings`; `uv run pytest tests/unit/cli tests/unit/validate
  tests/contracts/test_architecture.py -q` → `72 passed, 10 warnings`;
  `uv run automl --help >/tmp/automl-help.txt` exits 0; JSON flag surface smoke
  prints `json flag surface ok`; CLI parser files are under budget (`project`
  69, `experiment` 54, `trial` 67, `data` 48, `eval` 47, `validate` 28). Final
  Wave D review approved with no blockers and no MLflow/GCS artifact drift.
  Next action: author the detailed Wave E plan, post the path + task/risk
  summary, and wait for explicit "go".
- **2026-05-30** — Wave D plan amended after user review. `data materialize`
  now plans a core `materialize(include_rows=False)` metadata-only return shape
  for CLI use instead of CLI-side result patching. D4 now explicitly avoids new
  `automl.runner` / `automl.trial` facade exports; trial proposal defaults stay
  inside the existing `trial.create` verb and runner exit policy stays in
  `automl.runner.results`.
- **2026-05-29** — Wave D detailed plan authored in
  `docs/execution/cleanup-plan.md` under "Wave D — CLI discipline + validation
  uniformity — DETAILED". Execution is paused at the rule-4a plan gate. Key
  approval point: the plan reserves `--json` for `experiment run` output, so
  `validate proposal --json <path>` becomes `validate proposal --proposal-json
  <path>` and active skill commands move with it.
- **2026-05-29** — Wave C complete and gate green. Gate evidence:
  `uv run pytest tests/unit tests/integration tests/contracts -q` → `414 passed,
  14 warnings`; `runner/artifacts.py` is 26 lines; `agent/timeline/` package is
  live and `automl/agent/timeline.py` is gone; no raw `client.raw()` calls
  outside `automl/mlflow`; `uv run automl --help` and
  `uv run python hooks/agent_timeline.py publish --help` both exit 0. Final
  Wave C review found no blockers. Next action: author the detailed Wave D plan,
  post the path + task/risk summary, and wait for explicit "go".
- **2026-05-29** — Wave C is in progress. Blocker/design decision resolved:
  trial owns draft authoring artifacts (`paths`, `template`, metadata/manifest
  schemas, per-trial read types); runner owns execution and may consume only
  approved pure trial leaves; MLflow remains the persistence seam; CLI/workflow
  code may compose create/fork/promote authoring with run execution. Next action:
  execute the new `cleanup-plan.md` Task C7.5 before C8/C9.
- **2026-05-29** — Wave C plan amended after user review for consistency and
  intent alignment. Removed unrequested `automl agent publish`, skipped
  `agent/roles.py`, kept skill helper scripts, moved eval persistence to an
  experiment-scoped seam pattern, and constrained manifest typing to exact
  round-trip of the existing `manifest.json`. README now carries persistent
  consistency/intent guardrails for later waves.
- **2026-05-29** — Wave C detailed plan authored in
  `docs/execution/cleanup-plan.md` under "Wave C — DETAILED". No Wave C code
  changes have started. Next action: post the plan path + task/risk summary and
  wait for explicit "go" before executing Wave C.
- **2026-05-29** — Wave B complete and gate green. Gate evidence:
  `uv run pytest tests/unit tests/integration tests/contracts -q` → `385 passed,
  14 warnings`. Structural checks: zero live `_bound_for`; raw
  `active/session.experiment_id` reads only in `automl/project/session.py`;
  direct `mlflow_client.bind(` guarded to the session boundary; private
  `_routing` imports confined to `automl/mlflow`; non-storage `StorageError`
  raises removed outside the persistence seams. Final review passed after fixing
  the remaining data/runner binding bypasses in `48da520`. Next action: author
  the detailed Wave C plan, post the plan path + task/risk summary, and wait for
  explicit "go".
- **2026-05-29** — Wave B plan amended after plan review. Approved binding
  direction: config-backed `Session.active_experiment_id` is the normal source;
  CLI overrides enter only through `use_project`/`session_from_args`; domain code
  must not read raw `Session.experiment_id`; project-scoped reads may list all
  project experiments; trial reads use active experiment with optional override;
  cleanup uses explicit destructive targets. Execution protocol changed to
  update `STATUS.md` per wave, blocker, or handoff, not per task.
- **2026-05-29** — Wave A complete and gate green. Wave B detailed plan authored
  in `docs/execution/cleanup-plan.md`; execution is paused pending explicit user
  approval. The plan includes mandatory rule-5 stops for route-encoding
  equivalence and `bound_for` semantics before behavior can change.
- **2026-05-29** — Plan written and relocated to `docs/execution/`. Wave A is
  fully detailed; Waves B–E carry scope + acceptance and get detailed
  just-in-time. **Next action: execute Wave A, Task A1.** Nothing edited in
  `automl/` yet; working tree clean re: this plan.

---

## Wave A — Hygiene & code-side naming — COMPLETE

Detailed steps: `cleanup-plan.md` → "Wave A". Gate: `tests/unit tests/contracts`
green; single-owner `TrialStatus` contract test added; `import automl` clean.

- [x] A1 — delete vestigial `_RESET_FOR_TESTS` — DONE (`b3e6ee0`)
- [x] A2 — fix `automl.utils.__all__` — DONE (`f85e2b1`)
- [x] A3 — strip incidental imports from CLI `__all__` — DONE (`0bdd40d`)
- [x] A4 — de-dup facade `Experiment`/`ExperimentOverview` — DONE (`4bfb4b6`)
- [x] A5 — de-reference point-in-time docs in `errors.py` — DONE (`5988581`)
- [x] A6 — single public `TrialStatus` (canonical in `trial/types.py`; runner status → `str`) — DONE (`a83b89a`)
- [x] **Wave A gate** — acceptance checklist green; Wave A complete (`275 passed`; import smoke clean; no runner `TrialStatus`)

## Wave B — Routing + bind seam single-source (+ StorageError) — COMPLETE

Scope: `cleanup-plan.md` → "Wave B". **De-risk first**: characterize each route
encoding before unifying; keep config-backed experiment resolution simple; never
infer destructive cleanup targets. Gate: route round-trip + dry_run-isolation
tests green; route grammar built only in `mlflow/_routing.py`; one `bound_for`;
no `StorageError` outside the seam.

- [x] Detailed plan authored, amended, and approved
- [x] Cluster 3 — routing single-source
- [x] Cluster 4 — `bound_for` seam
- [x] StorageError-misuse fix + `delete_prefix` raise
- [x] Remaining data/runner direct bind bypasses fixed after gate review
- [x] **Wave B gate** — `385 passed`; structural greps clean; final review PASS

## Wave C — Seam adherence + monolith splits — COMPLETE

Scope: `cleanup-plan.md` → "Wave C". Gate: `runner/artifacts.py` < ~250L;
`agent/timeline/` package; existing hook publish behavior preserved; no new
`automl agent` CLI noun; no-`client.raw()` contract test; runner + cleanup
integration green; **MLflow artifact paths preserved**.

- [x] Detailed plan authored, amended, and approved
- [x] Pre-C8 trial/runner boundary correction — trial owns draft artifacts; runner owns execution
- [x] Cluster 5 — seam adherence
- [x] Cluster 6 — monolith splits
- [x] **Wave C gate** — `414 passed`; structural checks green; final review PASS

## Wave D — CLI discipline + validation uniformity — COMPLETE

Scope: `cleanup-plan.md` → "Wave D". Gate: `--json` only on `experiment run`;
`--max-iter` round-trip test; `data materialize` prints no row data;
`project/checks.py` + `_safe()`; CLI catalog + validate green.

- [x] Detailed plan authored
- [x] Cluster 7 — CLI discipline & correctness
- [x] Cluster 8 — validation uniformity
- [x] **Wave D gate** — `433 passed`; focused CLI/validate/contracts green; final review APPROVED

## Wave E — Docs/notebook truth + test-tier durability — COMPLETE

Scope: `cleanup-plan.md` → "Wave E". Gate: notebook facade contracts plus
opt-in notebook e2e; active docs use named split loading instead of
`load_training_data`; README/pyproject python versions agree; e2e
file/function/env gates named by domain with markers applied; `tests/contracts`
green.

- [x] Detailed plan authored, amended, and approved
- [x] Cluster 9 — docs/notebook truth
- [x] Cluster 10 — test-tier durability
- [x] **Wave E gate** — `455 passed`; contracts/markers/live notebook e2e/e2e skips/collect-only green; final review APPROVED

---

## Commits landed (this cleanup)

- `b3e6ee0` — `refactor(validate): drop vestigial _RESET_FOR_TESTS`
- `f85e2b1` — `fix(utils): __all__ lists only resolvable package exports`
- `0bdd40d` — `fix(cli): __all__ exports only CLI entry points`
- `4bfb4b6` — `refactor(facade): expose Experiment only`
- `5988581` — `docs(errors): describe intent without sub-spec citations`
- `a83b89a` — `refactor: single public TrialStatus`
- `8c4dd30` — `docs(execution): mark Wave A complete`
- `28413aa` — `test(routing): characterize route encodings before unifying`
- `3b5fa42` — `test(routing): characterize skill route cache paths`
- `37a013a` — `test(routing): characterize overview and eval registry roots`
- `b4cc448` — `refactor(mlflow): add public routing helpers`
- `d02be0e` — `fix(routing): round-trip multi-segment namespaces`
- `5a58754` — `refactor(routing): route eval and data paths through mlflow helpers`
- `a64fdd5` — `refactor(routing): migrate route callers to public helper`
- `0d31d2b` — `refactor(routing): centralize cleanup gcs prefixes`
- `967aa8d` — `test(mlflow): characterize session binding helper`
- `dde1b6e` — `refactor(mlflow): centralize session binding`
- `1fef591` — `fix(errors): use domain leaves for non-storage failures`
- `441d0f1` — `fix(gcs): raise on delete_prefix failures`
- `48da520` — `fix(mlflow): route remaining session binds through bound_for`
- `a46f8d6` — `docs(execution): mark Wave B complete`
- `9d15b49` — `docs(execution): author Wave C plan`
- `9e926a4` — `docs(execution): add consistency guardrails`
- `0a33b6a` — `test(seams): characterize Wave C preservation points`
- `e12b6c6` — `refactor(cleanup): route mlflow crud through seam`
- `4a77dd3` — `refactor(project): own ProjectOverview domain type`
- `4c84240` — `refactor(eval): route dataset persistence through experiment seam`
- `c1a7c49` — `refactor(mlflow): share json artifact path helper`
- `d5388a5` — `refactor(runner): split timing helpers`
- `7918912` — `refactor(runner): split serving validation`
- `557b2b9` — `docs(execution): amend trial runner boundary plan`
- `bd57b97` — `refactor(trial): own draft paths and templates`
- `7bbd85e` — `fix(trial): keep runner boundary imports lazy`
- `7452328` — `refactor(runner): type manifest and use trial artifact seam`
- `9616606` — `test(runner): pin manifest artifact preservation`
- `be9094a` — `refactor(runner): read trial metadata via domain schema`
- `834d5f2` — `fix(runner): preserve trial metadata json diagnostics`
- `c141c8c` — `refactor(agent): split timeline package`
- `fc6fe61` — `test(agent): pin timeline path formula`
- `591ef3c` — `fix(agent): avoid timeline publish module shadow`
- `e2fb89e` — `refactor(runner): thin artifact facade`
- `7c30a6b` — `docs(execution): mark Wave C complete`
- `9f44db8` — `docs(execution): author Wave D plan`
- `833d65c` — `docs(execution): amend Wave D plan`
- `fb16aa6` — `refactor(cli): reserve json flag for experiment run`
- `022c6ff` — `fix(agent): forward experiment run loop options`
- `749f3e0` — `fix(agent): preserve run route flags`
- `41c6895` — `fix(agent): preserve run confirmation flag`
- `e54df48` — `fix(agent): keep preflight imports lightweight`
- `e6c6fa7` — `fix(data): allow metadata-only materialize return`
- `5e36185` — `refactor(cli): move trial policies into domains`
- `03fff31` — `refactor(cli): keep verb parser files thin`
- `1399b77` — `fix(cli): update action split test patches`
- `e5c8031` — `fix(cli): update trial action test patches`
- `649c53f` — `fix(cli): remove stale facade patch`
- `83abaa2` — `refactor(validate): route project checks through domain`
- `d95d6fc` — `refactor(validate): make model checks domain-owned`
- `f445c04` — `fix(validate): pass session through runner validation`
- `ab7a0c2` — `docs(execution): mark Wave D complete`
- `b3c4976` — `docs(execution): author Wave E plan`
- `b69b3e9` — `docs(execution): amend Wave E plan`
- `51ab4d3` — `docs(data): document named split loading`
- `9b7d43f` — `docs(data): tighten split loading examples`
- `f72fd5e` — `docs: align user guidance with final facade`
- `80259e2` — `docs(model): clarify project baseline model guidance`
- `76210e2` — `docs(readme): clarify domain module imports`
- `551fb10` — `test(docs): guard readme domain import guidance`
- `e6e0788` — `test(notebooks): pin final facade surface`
- `6983dc0` — `docs(notebooks): align homecredit notebooks with facade`
- `8684898` — `docs(notebooks): fix facade alignment review gaps`
- `c21fd7c` — `docs(notebooks): align trial slice prediction target`
- `c5caf48` — `test(notebooks): add homecredit notebook e2e`
- `287f56f` — `test(notebooks): clear state after notebook e2e`
- `cc2ec45` — `test: make pytest tiers explicit`
- `bbf9e7c` — `test: harden e2e phase token contract`
- `cdaceca` — `test(skills): parse rendered cli commands`
- `0b20b41` — `test(contracts): pin cli surface and layer checks`
- `c6ad562` — `fix(notebooks): align eval artifact guidance`
- `b5a33ac` — `docs(notebooks): sync guide with notebook files`
- `0cc1637` — `docs(execution): mark Wave E complete`
- `d697ef7` — `fix(notebooks): make homecredit e2e executable`
