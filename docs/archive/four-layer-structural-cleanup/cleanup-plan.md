# brigit-automl Four-Layer Structural Cleanup — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pay down the ~102 verified findings from the 2026-05-29 architecture review so the four-layer `automl/` package is maintainability-merge-ready, without changing intended behavior.

**Architecture:** The findings collapse — by shared root cause / shared touchpoint — into 10 work-clusters, sequenced into 5 waves. Each wave is a self-contained, independently-testable, reviewable unit and a checkpoint. Wave A is fully detailed here; Waves B–E carry concrete scope + acceptance gates now and get their bite-sized task breakdown authored **just-in-time at wave start** (matching this repo's `plan/phases/` convention), because their exact steps depend on the post-prior-wave state.

**Tech Stack:** Python version from `pyproject.toml`, `uv` (`uv run pytest tests/<tier>/`), MLflow seam (`automl/mlflow/`), GCS via `automl/utils/io/gcs.py`, argparse CLI (`automl/cli/`), Claude Code plugin (`skills/`, `agents/`, `hooks/`). Tests: `tests/{unit,integration,contracts,e2e}`.

---

## How this plan is structured

- **One file, five waves.** Wave A = full bite-sized TDD below. Waves B–E = scope + findings + acceptance gate now; detailed steps written just-in-time before each wave runs (re-invoke `writing-plans` per wave).
- **Behavior-preserving.** Almost every change is a move / rename / dedup / single-source / doc fix. The safety net is the existing test suite plus a new architecture-contract assertion per structural invariant. "Tests still green + new contract pins the invariant" is the acceptance shape for relocations; genuine new behavior (e.g. route round-trip) gets real TDD.
- **Commit per task.** Branch is `refactor/four-layer`; commit after each task's tests pass.

## Out of scope (deferred — do NOT do in this plan)

- **Multi-runner & multi-agent architecture** → `docs/to-do/multi-runner-architecture.md`, `docs/to-do/multi-agent-orchestration.md`. (Wave C's monolith splits are prereqs, but the new abstractions are not built here.)
- **Logging wiring** → `docs/to-do/logging-and-observability.md`. Keep `automl/utils/logging.py` untouched; do not wire, do not delete.
- **`write_overview` disposition** (project + experiment overview-state writers, no callers / silent no-op) → deferred; sits in the overview-state area, a separate pass.
- **Heavy 168-site typed-error rewrite.** Only (a) fix the ~10 `StorageError` misuses (Wave B) and (b) adopt typed leaves *opportunistically* in files already being edited + a "new/edited code raises a typed error" rule. Keep the hierarchy and its currently-unraised leaf classes.
- **`O1–O7` merge-readiness business decisions** (`plan/final-review-open-items.md`) — product calls, not structure.
- **Keep-don't-delete:** `utils/paths.py`, `data.split.split_report`, the unraised error leaves, `configure_logging` — all legitimate-but-unwired; leave them.

## Wave overview & ordering constraints

| Wave | Clusters | Theme | Risk |
|---|---|---|---|
| **A** | 1 (reconcile loose/unwired surface), 2 (code-side naming) | hygiene + one-name-per-concept | low |
| **B** | 3 (routing single-source), 4 (bind seam), + `StorageError`-misuse fix | single-source the two cross-cutting seams | **med-high** (storage paths / dry_run isolation) |
| **C** | 5 (MLflow seam adherence), 6 (monolith splits) | kill the bypass; decompose the two 900-line files | med |
| **D** | 7 (CLI discipline & correctness), 8 (validation uniformity) | push policy down; uniform per-domain checks | med |
| **E** | 9 (docs/notebook truth), 10 (test-tier durability) | realign the DS surface; durable test tiers | low |

**Hard ordering constraints:**
1. **Code-side naming (A) before CLI-policy (D) and before docs (E)** — so policy and docs build on final names.
2. **Route + bind helpers (B) before the monolith splits (C)** — so split files call clean seams, not fresh copies.
3. **Docs/notebooks (E) last** — realign once, against the final API.
4. Each wave ships its own tests; the test-tier reorg (Wave E) is the tail.

## Finding → cluster → wave traceability

Source of truth for the finding list: the 2026-05-29 review output (`/tmp/review_areas.json`; 102 surviving findings = 11 high, 42 medium, 49 low; 1 refuted dropped). Cluster → wave mapping (every surviving finding lands in exactly one cluster):

- **Wave A** — Cluster 1: `_RESET_FOR_TESTS`, `utils.__all__`, CLI `__all__` leaks, `errors.py` stale docstrings (keep: `paths.py`, `split_report`, logging, error leaves; defer: `delete_prefix`→B, `write_overview`, `LoadedEvalDataset.registry`-typing-unverified). Cluster 2: two `TrialStatus` enums, facade `Experiment`/`ExperimentOverview` dup (defer to later waves: `TrialProposal`→`Proposal` prose→E, `registry.py` naming, import-style sweep, `Metric`-as-ABC).
- **Wave B** — Cluster 3: route grammar ×6 (`_routing` single-source), Dataset URI suffix-strip, eval GCS route ×3, mlflow route/`runs/` layout, cleanup route re-derive, private `_routing` imports, `.cache` path re-derive, `data/pipeline` GCS probe. Cluster 4: `_bound_for` → `mlflow.client.bound_for` + delete 5 copies + fix agent→experiment private import. Plus `StorageError`-misuse fix (~10 sites) and `delete_prefix` raise-on-failure (cleanup caller adapts).
- **Wave C** — Cluster 5: `client.raw()` leaks, runner `_log_artifact_file` bypass, untyped validation/timing/failure writers, cleanup raw MLflow + project↔trial seam-reach, `ProjectOverview` type home, `_json_artifact_path` dup, `data.profile`↔mlflow cycle, eval heavy-bytes-direct, leaderboard double-search. Cluster 6: split `runner/artifacts.py` (→ `timing.py` + `serving_validation.py` + thin `artifacts.py`), split `agent/timeline.py` (→ `timeline/{ingest,reconcile,publish}.py`), runner-uses-`TrialMetadata.read`, and type the current runner `manifest.json` payload in the trial domain without changing the artifact. Explicit deferrals: no `automl agent publish` noun, no `agent/roles.py` registry, and no skill-stub deletion in Wave C.
- **Wave D** — Cluster 7: dead `--json` removal, `experiment run` 3-parser/`--max-iter`, `materialize` DataFrame-dump, trial run/promote success-policy→domain, `trial lock`/`trial create` policy→domain, CLI ≤80-line budget, CLI-uses-facades. Cluster 8: `project/checks.py` + `_safe()`, model "errors-only" reconcile, validate→domain cycle documentation, `check_required_transformers` double-fetch, `_try_fit` predict dup.
- **Wave E** — Cluster 9: realign every Home Credit notebook, remove stale `load_training_data` hook docs in favor of named split loading, README python-version/facade, `Session.config` & `MODEL_CLASS` docs, `automl-guide` path, `TrialProposal`→`Proposal` prose sweep, `Metric`/`Model` doc, project-local model/eval example consolidation. Cluster 10: e2e-by-phase reorg, apply pytest markers, parse skill→CLI strings vs real CLI, fill/remove `shared/`+`regression/`, contract-test exemptions, pin surface isolation.

## Per-wave acceptance gates

- **A:** `uv run pytest tests/unit tests/contracts -q` green; new contract test pins single-owner `TrialStatus`; `import automl` + every domain `__all__` resolves.
- **B:** new route round-trip + dry_run-isolation contract/unit tests green; grep shows route grammar built only in `mlflow/_routing.py`; `bound_for` defined once in `mlflow/client.py`, zero `_bound_for` copies; no `StorageError` raised outside the mlflow seam for non-storage conditions; full `tests/unit tests/integration` green.
- **C:** `runner/artifacts.py` < ~250 lines; `agent/timeline.py` replaced by a `timeline/` package; existing `hooks/agent_timeline.py publish` compatibility preserved; no new `automl agent` CLI noun; contract test: no domain calls `client.raw()`; integration runner + cleanup tests green.
- **D:** CLI verb files ≤ ~80 lines (or justified); `--json` only on `experiment run`; `automl experiment run --max-iter N` round-trips into the spawned skill command (new contract test); `project/checks.py` exists with `_safe()`; CLI catalog + validate tests green.
- **E:** notebook facade contracts and opt-in notebook e2e checks exist; README python-version matches `pyproject`; active docs describe named split loading instead of `load_training_data`; e2e tests/env gates named by domain/behavior with markers applied; `tests/contracts` green.

---

## Wave A — Hygiene & code-side naming (DETAILED)

**Files touched:** `automl/validate/targets.py`, `automl/utils/__init__.py`, `automl/cli/*.py`, `automl/__init__.py`, `automl/errors.py`, `automl/runner/trial.py`, `automl/runner/__init__.py`, `automl/trial/promote.py`, `automl/cli/trial.py`, `tests/contracts/test_architecture.py`, `tests/unit/...`.

**Pre-flight (run once):**

- [ ] Confirm baseline green: `uv run pytest tests/unit tests/contracts -q`

### Task A1: Delete vestigial `_RESET_FOR_TESTS`

**Files:** Modify `automl/validate/targets.py:126-130`.

- [ ] **Step 1: Confirm it is truly vestigial (a no-op with no callers)**

Run: `grep -rn "_RESET_FOR_TESTS" automl/ tests/`
Expected: only the definition (`targets.py:126`) and `__all__` (`targets.py:130`) — no callers. (Body is `return None`; it reset the check-registry deleted in sub-spec 04.)

- [ ] **Step 2: Delete the function and its export**

In `automl/validate/targets.py`, remove:
```python
def _RESET_FOR_TESTS() -> None:
    return None
```
and change the `__all__` line to:
```python
__all__ = ["model", "project", "proposal"]
```

- [ ] **Step 3: Verify the validate suite is green**

Run: `uv run pytest tests/unit/validate -q`
Expected: PASS (nothing referenced the symbol).

- [ ] **Step 4: Commit**

```bash
git add automl/validate/targets.py
git commit -m "refactor(validate): drop vestigial _RESET_FOR_TESTS (registry deleted in sub-spec 04)"
```

### Task A2: Fix `automl.utils.__all__`

**Files:** Modify `automl/utils/__init__.py`; Test: `tests/unit/utils/test_public_api.py` (create).

- [ ] **Step 1: Write the failing test**

Create `tests/unit/utils/test_public_api.py`:
```python
import automl.utils as u


def test_utils_all_names_resolve():
    for name in u.__all__:
        assert hasattr(u, name), f"automl.utils.__all__ lists unresolvable name: {name}"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/unit/utils/test_public_api.py -v`
Expected: FAIL — `hashing`/`io`/`logging`/`paths` are submodules not bound as attributes of the package.

- [ ] **Step 3: Trim `__all__` to the real package-level export**

In `automl/utils/__init__.py` set:
```python
"""AutoML-agnostic utility helpers."""

from automl.utils.slug import SLUG_RE

__all__ = ["SLUG_RE"]
```
(Submodules stay importable by path: `from automl.utils.io import gcs`, `from automl.utils import hashing`. They do not belong in the package `__all__`.)

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/unit/utils/test_public_api.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add automl/utils/__init__.py tests/unit/utils/test_public_api.py
git commit -m "fix(utils): __all__ lists only resolvable package exports"
```

### Task A3: Remove incidental imports from CLI `__all__`

**Files:** Modify `automl/cli/project.py:73` and any other `automl/cli/*.py` with the same leak; Test: `tests/unit/cli/test_cli_public_api.py` (create).

- [ ] **Step 1: Enumerate the leaks**

Run: `grep -rn "__all__" automl/cli/`
Expected to find at least `automl/cli/project.py:73: __all__ = ["Path", "add_parser"]`. Note every `__all__` entry that is an incidental import (`Path`, `subprocess`, `use_project`, etc.) rather than a real CLI export (`add_parser`, `main`).

- [ ] **Step 2: Write the failing test**

Create `tests/unit/cli/test_cli_public_api.py`:
```python
import importlib

CLI_MODULES = ["project", "experiment", "trial", "data", "eval", "validate"]
INCIDENTAL = {"Path", "subprocess", "use_project", "argparse"}


def test_cli_all_has_no_incidental_imports():
    for name in CLI_MODULES:
        mod = importlib.import_module(f"automl.cli.{name}")
        leaked = INCIDENTAL.intersection(getattr(mod, "__all__", []))
        assert not leaked, f"automl.cli.{name}.__all__ leaks incidental imports: {leaked}"
```

- [ ] **Step 3: Run it to verify it fails**

Run: `uv run pytest tests/unit/cli/test_cli_public_api.py -v`
Expected: FAIL on `cli.project` (`Path`).

- [ ] **Step 4: Fix each leaking `__all__`**

For `automl/cli/project.py` set:
```python
__all__ = ["add_parser"]
```
Apply the same trim to any other file flagged in Step 1 (keep only `add_parser` / `main`).

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run pytest tests/unit/cli/test_cli_public_api.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add automl/cli tests/unit/cli/test_cli_public_api.py
git commit -m "fix(cli): __all__ exports only CLI entry points, not incidental imports"
```

### Task A4: De-duplicate the facade `Experiment` / `ExperimentOverview`

**Files:** Modify `automl/__init__.py:22,36`; Test: `tests/unit/test_facade.py` (create).

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_facade.py`:
```python
import automl


def test_experiment_is_single_top_level_name():
    assert hasattr(automl, "Experiment")
    assert "ExperimentOverview" not in automl.__all__


def test_experiment_overview_still_reachable_in_domain():
    from automl.experiment import ExperimentOverview  # noqa: F401
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/unit/test_facade.py -v`
Expected: FAIL — `ExperimentOverview` is in `automl.__all__`.

- [ ] **Step 3: Drop the duplicate top-level alias**

In `automl/__init__.py`, change the import to:
```python
from automl.experiment import Experiment
```
and remove `"ExperimentOverview"` from `__all__`. (`Experiment` is the public noun; `ExperimentOverview` stays available at `automl.experiment.ExperimentOverview`.)

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/unit/test_facade.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add automl/__init__.py tests/unit/test_facade.py
git commit -m "refactor(facade): expose Experiment only; ExperimentOverview stays in automl.experiment"
```

### Task A5: De-reference point-in-time docs in `errors.py` docstrings

**Files:** Modify `automl/errors.py` (docstrings only). No behavior change; existing `tests/unit/test_errors.py` is the guard.

- [ ] **Step 1: Edit the docstrings to stop citing point-in-time specs as authoritative**

Replace spec-number citations (e.g. "(sub-spec 02)", "(sub-spec 01)", "filled in as each domain lands") with intent-only descriptions. Example for the module docstring:
```python
"""Exception hierarchy for brigit-automl.

Lives at the package top (not under ``utils/``) because the exception hierarchy
is part of the public surface. ``StorageError`` wraps backend errors at the
MLflow/GCS persistence seam; the per-domain leaves describe where a failure
originated.
"""
```
and drop the trailing "(sub-spec NN)" from each class docstring.

- [ ] **Step 2: Verify nothing broke**

Run: `uv run pytest tests/unit/test_errors.py -q`
Expected: PASS (docstring-only change).

- [ ] **Step 3: Commit**

```bash
git add automl/errors.py
git commit -m "docs(errors): describe intent, drop point-in-time sub-spec citations"
```

### Task A6: One public `TrialStatus` (canonical in `trial/types.py`; runner status is `str`)

> **Superseded in Wave C:** This Wave A isolation contract was intentionally
> corrected by the approved pre-C8 ownership boundary in Task C7.5. The durable
> rule is no `automl.trial.*` imports `automl.runner`; runner may import only
> approved pure trial leaves such as schemas and path verification. Runner must
> not own trial draft paths/templates.

**Files:** Modify `automl/runner/trial.py:44-46,51,230,275,665`, `automl/runner/__init__.py:3,5`, `automl/trial/promote.py`, `automl/cli/trial.py`; Test: `tests/contracts/test_architecture.py`.

Historical Wave A constraint: the contract test
`test_runner_domain_does_not_import_trial_domain_at_runtime` forbade `runner`
importing `automl.trial`, so the runner did **not** import the canonical enum and
`TrialResult.status` became a plain `str` whose values match the canonical enum's
(`"FINISHED"`/`"FAILED"`). Task C7.5 supersedes the blanket import ban with the
approved pure-trial-leaf rule.

- [ ] **Step 1: Write the failing contract test**

Append to `tests/contracts/test_architecture.py`:
```python
def test_trial_status_has_single_public_owner():
    import automl.runner
    import automl.trial

    assert hasattr(automl.trial, "TrialStatus"), "canonical TrialStatus must live in automl.trial"
    assert not hasattr(automl.runner, "TrialStatus"), (
        "runner must not export a second TrialStatus; TrialResult.status is a plain str"
    )
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/contracts/test_architecture.py::test_trial_status_has_single_public_owner -v`
Expected: FAIL — `automl.runner.TrialStatus` exists.

- [ ] **Step 3: Remove the runner-local enum; make `status` a str**

In `automl/runner/trial.py`:
- Delete the enum:
```python
class TrialStatus(str, Enum):
    FINISHED = "FINISHED"
    FAILED = "FAILED"
```
- Change the dataclass field `status: TrialStatus` → `status: str`.
- Change the two construction sites: `status=TrialStatus.FINISHED` → `status="FINISHED"` (line ~230) and `status=TrialStatus.FAILED` → `status="FAILED"` (line ~275).
- Remove `TrialStatus` from the module `__all__` (line ~665) → `__all__ = ["TrialResult", "run_trial"]`.
- Drop the now-unused `Enum` import if nothing else uses it.

In `automl/runner/__init__.py`:
```python
from automl.runner.trial import TrialResult, run_trial

__all__ = ["TrialResult", "run_trial"]
```

- [ ] **Step 4: Adapt the two consumers that read `result.status`**

Run: `grep -n "\.status" automl/trial/promote.py automl/cli/trial.py`
For each comparison against the old runner enum, compare against the string instead, e.g. `result.status == "FINISHED"` (or `== TrialStatus.FINISHED.value` if the file already imports the canonical `automl.trial.TrialStatus`). During Wave A, do not add an `automl.trial` import inside `automl/runner/`; Task C7.5 later replaces that broad rule with the approved pure-trial-leaf allowlist.

- [ ] **Step 5: Run the affected suites**

Run: `uv run pytest tests/unit/runner tests/unit/trial tests/unit/cli tests/contracts -q`
Expected: PASS, including the new single-owner test and `test_runner_domain_does_not_import_trial_domain_at_runtime`.

- [ ] **Step 6: Commit**

```bash
git add automl/runner automl/trial/promote.py automl/cli/trial.py tests/contracts/test_architecture.py
git commit -m "refactor: single public TrialStatus (canonical in trial.types); runner status is a str"
```

### Wave A acceptance

- [ ] `uv run pytest tests/unit tests/contracts -q` is green.
- [ ] `uv run python -c "import automl; import automl.utils, automl.cli.project, automl.experiment"` exits 0.
- [ ] `grep -rn "TrialStatus" automl/runner` returns nothing.
- [ ] Hand off to wave review (see "Execution handoff").

---

## Wave B — Routing & bind seam single-source (+ StorageError fix) — DETAILED

**Detail authored:** 2026-05-29. **Do not execute until the Wave B plan is approved.**

**Cluster 3 — routing/path grammar → one home in `automl/mlflow/_routing.py`.** Add pure helpers taking explicit `(namespace, dry_run, project, experiment)` (not a bound session) so non-seam callers can build routes without binding. Migrate the stragglers onto them — start with `mlflow/project/overview.py:_route_root` (a verbatim duplicate already inside the seam), then `eval/eval_dataset.py`, `eval/registry.py`, `data/pipeline.py` (delete the `route[:-len(project_suffix)]` suffix-strip), `runner/session_lock.py`, `runner/paths.py` (local-path variant), `project/cleanup.py` (incl. the `runs/<YYYY-MM>/<run_id>` layout → `_routing.bucket_uri_for`), and the skill `render_context.py` `.cache` path. Replace private `automl.mlflow._routing` imports in `agent/`/`project/` with a public routing entry point.

**Cluster 4 — Session→MLflow bind seam.** Promote one public `automl.mlflow.client.bound_for(session, *, experiment_id=None)`; delete the 5 private `_bound_for` copies (`experiment/lifecycle.py`, `trial/show.py`, `trial/cleanup.py`, `project/cleanup.py`, and the duplicate body) and the cross-boundary `from automl.experiment.lifecycle import _bound_for` in `agent/timeline.py` + `agent/proposer_context.py`. Approved semantics: `Session.active_experiment_id` is the config-backed resolved experiment id; CLI overrides enter only when constructing the `Session`; domain code must not read raw `Session.experiment_id`; project-scoped reads may bind without an experiment; experiment/trial-scoped reads pass an explicit resolved experiment id; destructive cleanup never infers what to delete from the active session.

**Plus:** fix the ~10 `StorageError` misuses (raise the semantically correct leaf — `EvalError`/`DataError`/runner validation error) at `runner/artifacts.py:178,182,806`, `eval/prepare.py:78,132,146`, `data/pipeline.py:217,316`, `project/cleanup.py:239`; and make `gcs.delete_prefix` raise on failure instead of returning `int|str`, updating its one caller `project/cleanup.py:207` to catch-and-record per the continue-and-collect model.

**Acceptance gate:**
- New tests: route round-trip (`build → parse` for each universe incl. `dry_run` + `namespace`) and a dry_run/real isolation test that asserts the two universes never share a prefix.
- `grep` shows route/`runs/` grammar constructed only inside `automl/mlflow/_routing.py`; `bound_for` defined once; zero `_bound_for`; no domain read of raw `active.experiment_id`.
- No `StorageError` raised outside `automl/mlflow/` for a non-storage condition.
- `uv run pytest tests/unit tests/integration tests/contracts -q` green.

### Wave B pre-flight

- [ ] Re-run the Wave A gate as the baseline for this wave:
```bash
uv run pytest tests/unit tests/contracts -q
```
- [ ] Confirm the expected current duplication before editing:
```bash
rg -n "def _bound_for|from automl\.experiment\.lifecycle import _bound_for|automl\.mlflow\._routing|route\[:-len|runs/<|runs/\{|\bdry_run\b" automl skills tests
```
- [ ] If any baseline test fails, stop and debug before touching Wave B files.

### Task B1: Characterize every existing route encoding before unifying it

**Files:** Create or extend `tests/unit/mlflow/test_routing_characterization.py`, `tests/unit/runner/test_session_lock.py`, `tests/unit/runner/test_paths.py`, `tests/unit/eval/test_eval_dataset_routes.py`, `tests/unit/data/test_pipeline_routes.py`, `tests/unit/project/test_cleanup.py`, `tests/unit/skills/test_render_context_routes.py`.

- [ ] **Step 1: Add characterization tests for all current builders**

Pin the route strings and URI prefixes that exist before migration. Include namespace and dry-run cases because this wave is specifically protecting storage isolation:
```python
def test_project_and_experiment_routes_are_characterized():
    from automl.mlflow import _routing

    assert _routing.project_route_for("demo") == "demo"
    assert _routing.project_route_for("demo", dry_run=True) == "dry_run/demo"
    assert _routing.project_route_for("demo", namespace="qa") == "qa/demo"
    assert _routing.project_route_for("demo", dry_run=True, namespace="qa") == "qa/dry_run/demo"
```

Add focused tests that compare the duplicated call sites to the seam's current output, not to a new desired output:
```python
def test_session_lock_route_matches_current_experiment_route():
    from automl.runner.session_lock import route_for_session

    route = route_for_session(project_name="demo", experiment_id="phase6", namespace="qa", dry_run=True)
    assert route == "qa/dry_run/demo/phase6"
```

For dataset/eval/GCS routes, assert the complete current prefix strings, including the dataset suffix-strip behavior that currently lands datasets under the bound GCS prefix followed by `/project/data/datasets/`.

- [ ] **Step 2: Run characterization tests while they still hit the old code**

Run the narrow tests:
```bash
uv run pytest tests/unit/mlflow/test_client_and_routing.py tests/unit/runner/test_session_lock.py tests/unit/project/test_cleanup.py tests/unit/eval tests/unit/data -q
```

- [ ] **Step 3: Mandatory route-equivalence decision gate**

If any duplicated route builder does not match the existing seam output, do not "fix" it inside this task. Record the exact old outputs and stop for a user decision: either preserve the divergent encoding as intended or classify it as a bug and approve the behavior change. This is the Wave B route-encoding concern from rule 5.

- [ ] **Step 4: Commit characterization only**
```bash
git add tests/unit/mlflow tests/unit/runner tests/unit/project tests/unit/eval tests/unit/data tests/unit/skills
git commit -m "test(routing): characterize route encodings before unifying"
```

### Task B2: Add the public routing entry point and pure route helpers

**Files:** `automl/mlflow/_routing.py`, `automl/mlflow/routing.py` (new), `automl/mlflow/__init__.py`, `tests/unit/mlflow/test_client_and_routing.py`, `tests/contracts/test_architecture.py`.

- [ ] **Step 1: Write failing tests for explicit route construction and parsing**

Add tests that define the public behavior before implementation:
```python
def test_route_build_parse_round_trip_with_namespace_and_dry_run():
    from automl.mlflow import routing

    route = routing.experiment_route_for(
        project_name="demo",
        experiment_id="exp-1",
        namespace="qa",
        dry_run=True,
    )

    assert route == "qa/dry_run/demo/exp-1"
    assert routing.parse_experiment_route(route) == {
        "namespace": "qa",
        "dry_run": True,
        "project_name": "demo",
        "experiment_id": "exp-1",
    }
```

Add dry-run isolation:
```python
def test_dry_run_and_real_experiments_do_not_share_prefixes():
    from automl.mlflow import routing

    real = routing.experiment_route_for(project_name="demo", experiment_id="exp")
    dry = routing.experiment_route_for(project_name="demo", experiment_id="exp", dry_run=True)

    assert real == "demo/exp"
    assert dry == "dry_run/demo/exp"
    assert not real.startswith(dry)
    assert not dry.startswith(real + "/")
```

- [ ] **Step 2: Implement helpers in the private single source**

Keep all grammar in `automl/mlflow/_routing.py`. Add pure helpers that do not read the current bound client:
```python
def experiment_route_for(
    *,
    project_name: str,
    experiment_id: str,
    namespace: str = "",
    dry_run: bool = False,
) -> str:
    base = project_route_for(project_name, dry_run=dry_run, namespace=namespace)
    return "/".join((base, _component("experiment_id", experiment_id)))
```

Add `parse_experiment_route(route: str) -> dict[str, object]` that reverses `experiment_route_for` for the supported grammar:
- `project/experiment`
- `dry_run/project/experiment`
- `namespace/project/experiment`
- `namespace/dry_run/project/experiment`

Reject malformed routes with `StorageError` because route parsing is part of the persistence seam.

Add local path helpers only as thin joins over route strings:
```python
def experiment_local_path(root: Path, *, project_name: str, experiment_id: str, namespace: str = "", dry_run: bool = False) -> Path:
    return root / "experiments" / Path(
        experiment_route_for(
            project_name=project_name,
            experiment_id=experiment_id,
            namespace=namespace,
            dry_run=dry_run,
        )
    )
```

- [ ] **Step 3: Add a public re-export module**

Create `automl/mlflow/routing.py` as the public entry point:
```python
"""Public route helpers backed by the MLflow routing seam."""

from automl.mlflow._routing import (
    bucket_uri_for,
    experiment_local_path,
    experiment_route,
    experiment_route_for,
    parse_experiment_route,
    project_route,
    project_route_for,
    project_route_prefix,
    route_prefix_for,
)

__all__ = [
    "bucket_uri_for",
    "experiment_local_path",
    "experiment_route",
    "experiment_route_for",
    "parse_experiment_route",
    "project_route",
    "project_route_for",
    "project_route_prefix",
    "route_prefix_for",
]
```

Expose the module from `automl/mlflow/__init__.py`:
```python
from automl.mlflow import routing
```

- [ ] **Step 4: Architecture guard**

Add a contract test that domain code imports the public module, not the private one:
```python
def test_domains_do_not_import_private_mlflow_routing():
    import pathlib

    offenders = []
    for path in pathlib.Path("automl").rglob("*.py"):
        if path.parts[:2] == ("automl", "mlflow"):
            continue
        text = path.read_text()
        if "automl.mlflow._routing" in text or "import _routing" in text:
            offenders.append(str(path))

    assert offenders == []
```

- [ ] **Step 5: Verify and commit**
```bash
uv run pytest tests/unit/mlflow/test_client_and_routing.py tests/contracts/test_architecture.py -q
git add automl/mlflow tests/unit/mlflow/test_client_and_routing.py tests/contracts/test_architecture.py
git commit -m "refactor(mlflow): add public routing helpers"
```

### Task B3: Migrate GCS/eval/data route callers to the routing helpers

**Files:** `automl/mlflow/project/overview.py`, `automl/eval/eval_dataset.py`, `automl/eval/registry.py`, `automl/data/pipeline.py`, related route tests.

- [ ] **Step 1: Replace duplicate seam-internal project overview routing**

In `automl/mlflow/project/overview.py`, remove the local `_route_root` string construction and call the pure project route helper from inside the seam.

- [ ] **Step 2: Replace eval route construction**

Route all eval dataset and augmentation roots through one helper path. The target shape must match the B1 characterization:
```python
route = routing.experiment_route_for(
    project_name=session.project_name,
    experiment_id=session.active_experiment_id,
    namespace=session.namespace,
    dry_run=session.dry_run,
)
```
Then append eval-specific suffixes locally, e.g. `eval/datasets/{name}` or `eval/augmentations/{name}`. Do not change suffix names or MLflow/GCS logged paths.

- [ ] **Step 3: Replace data pipeline suffix-strip behavior with explicit parent helper**

Delete the `route[:-len(project_suffix)]` pattern. If the characterization says the current dataset prefix is the bound GCS prefix followed by `/[namespace/][dry_run/]project/data/datasets/`, add a named helper in `_routing.py`:
```python
def project_data_route_for(*, project_name: str, namespace: str = "", dry_run: bool = False) -> str:
    return "/".join((project_route_for(project_name, namespace=namespace, dry_run=dry_run), "data"))
```
Use that helper from `automl/data/pipeline.py`.

- [ ] **Step 4: Verify route characterization still passes**
```bash
uv run pytest tests/unit/eval tests/unit/data tests/unit/mlflow/test_client_and_routing.py -q
```

- [ ] **Step 5: Commit**
```bash
git add automl/mlflow/project/overview.py automl/eval automl/data automl/mlflow tests/unit/eval tests/unit/data tests/unit/mlflow
git commit -m "refactor(routing): route eval and data paths through mlflow helpers"
```

### Task B4: Migrate local/session/cleanup/skill route callers and remove private route imports

**Files:** `automl/runner/session_lock.py`, `automl/runner/paths.py`, `automl/project/cleanup.py`, `automl/agent/timeline.py`, `skills/automl/scripts/render_context.py`, related tests.

- [ ] **Step 1: Migrate runner session/local paths**

Make `route_for_session` call `routing.experiment_route_for`. The then-runner-owned
trial route helper called `routing.experiment_local_path` or the route helper plus
`Path`, preserving the existing `project_dir / "experiments" / route` shape.
Task C7.5 later moves that draft path ownership to `automl.trial.paths`.

- [ ] **Step 2: Migrate project cleanup routes and run artifact prefixes**

Use `routing.experiment_route_for`, `routing.project_route_for`, and `routing.bucket_uri_for` for cleanup preview/apply. Preserve the current `runs/<YYYY-MM>/<run_id>/` layout exactly; if a test shows a different layout, stop before changing MLflow/GCS delete prefixes.

- [ ] **Step 3: Migrate agent/project imports away from private `_routing`**

Replace non-seam imports with:
```python
from automl.mlflow import routing
```
No file outside `automl/mlflow/` may import `automl.mlflow._routing` after this task.

- [ ] **Step 4: Migrate skill `.cache` route rendering**

In `skills/automl/scripts/render_context.py`, use the same route helper for proposal handoff and timeline cache paths. Keep file names and `.cache` placement unchanged.

- [ ] **Step 5: Verify and grep**
```bash
uv run pytest tests/unit/runner tests/unit/project tests/contracts -q
rg -n "automl\.mlflow\._routing|import _routing" automl skills
```
The `rg` command must return no non-seam route imports.

- [ ] **Step 6: Commit**
```bash
git add automl/runner automl/project automl/agent skills/automl/scripts/render_context.py tests/unit/runner tests/unit/project tests/contracts
git commit -m "refactor(routing): migrate route callers to public helper"
```

### Task B5: Characterize and pin approved binding semantics

**Files:** `tests/unit/mlflow/test_client_bound_for.py`, `tests/unit/experiment`, `tests/unit/trial`, `tests/unit/project`.

- [ ] **Step 1: Write characterization tests for current drift and approved behavior**

Pin the current drift so the migration is deliberate:
- `experiment/lifecycle._bound_for(active=None)` yields without binding.
- `experiment/lifecycle._bound_for(active, experiment_id="override")` binds the override.
- trial/show and trial/cleanup bind `active.active_experiment_id`.
- project/cleanup currently binds raw `active.experiment_id`, which can be `None` even when config has a resolved active experiment.

Then add target tests for the approved rule:
- `client.bound_for(active)` binds project/dry-run/namespace/tracking/GCS without forcing an experiment id.
- experiment-scoped callers pass `experiment_id=active.active_experiment_id`.
- explicit CLI/argument override wins only because it is already stored in the session or passed as an explicit `experiment_id`.
- project experiment listing can run without an experiment id and list all experiments under the project route.
- trial listing uses `active.active_experiment_id` by default and can override via its experiment argument.
- cleanup tests assert destructive targets come from explicit cleanup arguments / parent references, not inferred from `active.experiment_id`.

- [ ] **Step 2: Run the tests**
```bash
uv run pytest tests/unit/experiment tests/unit/trial tests/unit/project -q
```

- [ ] **Step 3: Stop only if implementation evidence contradicts the approved rule**

The binding decision is no longer open-ended. Continue if tests fit the rule above. Stop only if the code shows a legitimate operation that needs raw `Session.experiment_id` directly, or a destructive cleanup flow that cannot name its target explicitly.

Target `client.bound_for` behavior:
```python
@contextmanager
def bound_for(active: Session | None, *, experiment_id: str | None = None) -> Iterator[None]:
    if active is None:
        yield
        return

    with bind(
        tracking_uri=active.config.mlflow_tracking_uri,
        bucket=active.config.gcs_bucket,
        gcs_prefix=active.config.gcs_prefix,
        project_name=active.project_name,
        experiment_id=experiment_id,
        dry_run=active.dry_run,
        namespace=active.namespace,
    ):
        yield
```

- [ ] **Step 4: Commit characterization if tests were added**
```bash
git add tests/unit/mlflow tests/unit/experiment tests/unit/trial tests/unit/project
git commit -m "test(mlflow): characterize bound session semantics"
```

### Task B6: Implement `mlflow.client.bound_for` and delete all `_bound_for` copies

**Files:** `automl/mlflow/client.py`, `automl/mlflow/__init__.py`, `automl/experiment/lifecycle.py`, `automl/experiment/views/*.py`, `automl/trial/show.py`, `automl/trial/cleanup.py`, `automl/project/cleanup.py`, `automl/agent/timeline.py`, `automl/agent/proposer_context.py`, tests.

- [ ] **Step 1: Implement only the approved semantics**

Add `bound_for` to `automl/mlflow/client.py` and `__all__`. Import `Session` under `TYPE_CHECKING` to avoid runtime cycles:
```python
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from automl.project import Session


@contextmanager
def bound_for(active: "Session | None", *, experiment_id: str | None = None) -> Iterator[None]:
    if active is None:
        yield
        return

    with bind(
        tracking_uri=active.config.mlflow_tracking_uri,
        bucket=active.config.gcs_bucket,
        gcs_prefix=active.config.gcs_prefix,
        project_name=active.project_name,
        experiment_id=experiment_id,
        dry_run=active.dry_run,
        namespace=active.namespace,
    ):
        yield
```

- [ ] **Step 2: Replace every private copy without leaking raw session overrides**

At each call site, import `client` from `automl.mlflow` and wrap the existing bound operation in `with client.bound_for(...)`. Do not introduce any new private bind helper while migrating.

Use these call-site rules:
- project-scoped reads/listing: `with client.bound_for(active):`
- experiment-scoped operations: `with client.bound_for(active, experiment_id=explicit_id if explicit_id is not None else active.active_experiment_id):`
- trial-scoped operations: use the parent experiment id / resolved active experiment id explicitly.
- cleanup: use the explicit cleanup target or parent experiment reference; do not infer delete targets from the active session.

Delete private `_bound_for` functions from experiment/trial/project modules. Replace agent imports from `automl.experiment.lifecycle` with the public client seam.

- [ ] **Step 3: Verify zero private copies**
```bash
rg -n "def _bound_for|_bound_for|from automl\.experiment\.lifecycle import _bound_for" automl tests
rg -n "active\.experiment_id|session\.experiment_id" automl tests
```
Expected: no live `_bound_for` implementation/import, and no raw active-session experiment reads outside `automl/project/session.py` and CLI session construction tests. Test references are allowed only if they assert absence.

- [ ] **Step 4: Run affected tests and commit**
```bash
uv run pytest tests/unit/mlflow tests/unit/experiment tests/unit/trial tests/unit/project tests/contracts -q
git add automl/mlflow automl/experiment automl/trial automl/project automl/agent tests/unit/mlflow tests/unit/experiment tests/unit/trial tests/unit/project tests/contracts
git commit -m "refactor(mlflow): centralize session binding in client.bound_for"
```

### Task B7: Replace non-storage `StorageError` raises with typed domain errors

**Files:** `automl/runner/artifacts.py`, `automl/eval/prepare.py`, `automl/data/pipeline.py`, `automl/project/cleanup.py`, error tests.

- [ ] **Step 1: Write or update tests for typed leaves**

Add focused tests that assert the exact leaf class for each category:
- runner validation shape/precondition failures raise `ValidationError`;
- eval preparation partial-object failures raise `EvalError`;
- data pipeline partial-object and identity mismatch failures raise `DataError`;
- project cleanup MLflow hard-delete configuration failure raises `ProjectError`.

- [ ] **Step 2: Implement the typed replacements**

Use these imports and leaves:
```python
from automl.errors import ValidationError
```
for `runner/artifacts.py:178,182,806`.

Use:
```python
from automl.errors import EvalError
```
for `eval/prepare.py:78,132,146`.

Use:
```python
from automl.errors import DataError
```
for `data/pipeline.py:217,316`.

Use:
```python
from automl.errors import ProjectError
```
for `project/cleanup.py:239`.

Do not change exception messages unless a test requires matching the old wording.

- [ ] **Step 3: Add a contract guard for future misuse**

Extend `tests/contracts/test_architecture.py` with a text-level guard that fails if `StorageError(` appears outside `automl/mlflow/` and `automl/utils/io/gcs.py`:
```python
def test_storage_error_is_not_used_for_domain_conditions():
    import pathlib

    allowed = {
        pathlib.Path("automl/mlflow"),
        pathlib.Path("automl/utils/io/gcs.py"),
    }
    offenders = []
    for path in pathlib.Path("automl").rglob("*.py"):
        if any(path == root or root in path.parents for root in allowed):
            continue
        if "StorageError(" in path.read_text():
            offenders.append(str(path))

    assert offenders == []
```

- [ ] **Step 4: Verify and commit**
```bash
uv run pytest tests/unit/runner tests/unit/eval tests/unit/data tests/unit/project tests/contracts -q
git add automl/runner/artifacts.py automl/eval/prepare.py automl/data/pipeline.py automl/project/cleanup.py tests/unit tests/contracts/test_architecture.py
git commit -m "fix(errors): use domain leaves for non-storage failures"
```

### Task B8: Make `gcs.delete_prefix` raise on delete failure and let cleanup record failures

**Files:** `automl/utils/io/gcs.py`, `automl/project/cleanup.py`, `tests/unit/utils/io/test_gcs.py`, `tests/unit/project/test_cleanup.py`.

- [ ] **Step 1: Write failing tests**

For `gcs.delete_prefix`, assert blob-delete failure raises instead of returning a string:
```python
class FailingDeleteBlob:
    name = "automl-root/home_credit/baseline/a.json"

    def delete(self):
        raise RuntimeError("boom")


class FailingDeleteClient:
    def list_blobs(self, bucket, prefix):
        assert bucket == "automl-test-bucket"
        assert prefix == "automl-root/home_credit/baseline/"
        return [FailingDeleteBlob()]


def test_delete_prefix_raises_when_a_blob_delete_fails():
    with pytest.raises(RuntimeError, match="failed"):
        gcs.delete_prefix(
            "gs://automl-test-bucket/automl-root/home_credit/baseline/",
            client=FailingDeleteClient(),
        )
```

For cleanup apply, assert the continue-and-collect model records the failure:
```python
def test_apply_records_gcs_delete_failure_and_continues(tmp_path, monkeypatch):
    active = _session(tmp_path)
    client.bind(
        tracking_uri=active.config.mlflow_tracking_uri,
        bucket=active.config.gcs_bucket,
        gcs_prefix=active.config.gcs_prefix,
        project_name=active.project_name,
        experiment_id=active.active_experiment_id,
    )
    experiment.ensure()
    local_root = active.config.project_dir / "experiments" / "home_credit" / "baseline"
    local_root.mkdir(parents=True)
    monkeypatch.setattr(gcs, "_gcs_client", lambda: FailingDeleteClient())

    report = delete("baseline", scope="experiment", apply=True, session=active)

    assert report.result.gcs[
        "gs://automl-test-bucket/automl-root/home_credit/baseline/"
    ].startswith("failed: failed to delete")
    assert report.result.local[str(local_root)] == "deleted"
```

- [ ] **Step 2: Change `delete_prefix` return type to `int`**

In `automl/utils/io/gcs.py`, make all blob delete failures raise `RuntimeError` with enough context to identify the prefix/blob. Keep successful return as the deleted blob count:
```python
def delete_prefix(prefix: str, *, client: storage.Client | None = None) -> int:
    bucket, blob_prefix = parse_gcs_uri(prefix.rstrip("/") + "/_")
    blob_prefix = blob_prefix.removesuffix("_")
    deleted = 0
    try:
        blobs = _client_or_default(client).list_blobs(bucket, prefix=blob_prefix)
    except Exception as exc:
        raise RuntimeError(f"failed to list GCS prefix {prefix!r}: {exc}") from exc

    for blob in blobs:
        try:
            blob.delete()
            deleted += 1
        except Exception as exc:
            name = getattr(blob, "name", "<unknown>")
            raise RuntimeError(f"failed to delete {name!r} under {prefix!r}: {exc}") from exc
    return deleted
```

- [ ] **Step 3: Update project cleanup caller**

Replace any comprehension that stores the raw `delete_prefix` return with an explicit loop:
```python
gcs_results: dict[str, int | str] = {}
for prefix in plan.gcs_prefix_patterns:
    try:
        gcs_results[prefix] = gcs.delete_prefix(prefix)
    except Exception as exc:
        gcs_results[prefix] = f"failed: {exc}"
```

- [ ] **Step 4: Verify and commit**
```bash
uv run pytest tests/unit/utils/io tests/unit/project -q
git add automl/utils/io/gcs.py automl/project/cleanup.py tests/unit/utils/io tests/unit/project
git commit -m "fix(gcs): raise on delete_prefix failures"
```

### Task B9: Wave B gate, review, status update

- [ ] **Step 1: Run the full Wave B gate**
```bash
uv run pytest tests/unit tests/integration tests/contracts -q
rg -n "def _bound_for|_bound_for|from automl\.experiment\.lifecycle import _bound_for" automl tests
rg -n "automl\.mlflow\._routing|import _routing" automl skills
rg -n "active\.experiment_id|session\.experiment_id" automl tests
rg -n "StorageError\(" automl | grep -v "automl/mlflow" | grep -v "automl/utils/io/gcs.py" || true
```

- [ ] **Step 2: Request review**

Use `superpowers:requesting-code-review` after the gate is green. Review focus:
- route output preservation, especially dry-run and namespace isolation;
- no MLflow/GCS path or metadata logging drift beyond the approved plan;
- single `bound_for` semantic matches the approved rule: config-backed resolution via `Session.active_experiment_id`, CLI override at the boundary, project reads can bind without experiment, destructive cleanup uses explicit targets;
- `StorageError` only represents persistence-seam failures.

- [ ] **Step 3: Commit STATUS after review fixes**

Update `docs/execution/STATUS.md`: Wave B complete, Wave C at plan gate, include gate command result and commit hashes.
```bash
git add docs/execution/STATUS.md
git commit -m "docs(execution): mark Wave B complete"
```

## Wave C — Seam adherence + monolith splits — DETAILED

**Detail authored:** 2026-05-29. **Do not execute until the Wave C plan is approved.**

**Goal:** Remove the remaining durable-seam bypasses and split the two 900-line
modules without changing logged MLflow/GCS/local artifact shapes.

**Architecture:** Keep `automl/mlflow/` as the only low-level MLflow/GCS
persistence seam. Keep domain modules as orchestration/read-model code. Split
large files by behavior: runner timing, runner serving validation, agent
timeline ingestion, timeline reconciliation, and timeline publishing. Keep skill
scripts and current agent role strings unchanged in this wave.

**Trial/runner ownership boundary, approved before C8/C9:** surface/CLI code may
compose trial authoring and runner execution workflows. The trial domain owns
draft trial artifacts and authoring primitives: create/fork, draft paths,
generated `run.py` template, metadata/manifest schemas, and per-trial read
types. The runner owns execution of one trial chain and may consume only
approved pure trial leaves for schemas and path verification. The MLflow seam
persists/loads domain objects and hides raw MLflow/GCS. Do not preserve
runner-owned path/template compatibility shims; stale spec text loses to domain
ownership.

**Key rule-5 stops for this wave:**
- If a characterization test shows that a planned seam migration would change an
  MLflow experiment name, run tag, metric key, artifact path, GCS URI, or local
  cache path, stop and ask whether that drift is intended.
- If typing the runner `manifest.json` payload cannot preserve the current JSON
  shape exactly, stop and leave the artifact free-form until a dedicated schema
  design.
- Do not add `automl agent publish`, do not add an `agent/roles.py` registry, and
  do not delete `skills/inspect`, `skills/profile`, or `skills/propose` helper
  scripts in this wave.
- Stop if an implementation would make `automl.trial.*` import
  `automl.runner`, keep `automl.runner.paths`/`automl.runner.template` as
  compatibility shims, or put create+run orchestration in trial core. That
  composition belongs at the CLI/workflow surface.

### Wave C pre-flight

- [ ] **Step 1: Confirm baseline still matches the Wave B gate**

```bash
uv run pytest tests/unit tests/integration tests/contracts -q
```

Expected: `385 passed` or more, with only known MLflow/Pydantic warnings.

- [ ] **Step 2: Measure current large-file and bypass state**

```bash
wc -l automl/runner/artifacts.py automl/agent/timeline.py
rg -n "client\.raw\(|mlflow_client\.raw\(" automl
```

Expected at start: `runner/artifacts.py` and `agent/timeline.py` are both about
900 lines; non-seam `client.raw()` hits are in `automl/project/cleanup.py`.

### Task C1: Characterize remaining seam and artifact-path behavior

**Files:**
- Modify: `tests/contracts/test_architecture.py`
- Modify: `tests/integration/runner/test_one_trial_local.py`
- Modify: `tests/integration/cleanup/test_experiment_delete.py`
- Modify: `tests/unit/agent/test_timeline.py`
- Modify: `tests/unit/skills/test_render_context_routes.py`

- [ ] **Step 1: Add a ratchet for domain `client.raw()` usage**

In `tests/contracts/test_architecture.py`, add a known-offender set for the
current cleanup bypass and a test that scans only `automl/` production files:

```python
KNOWN_DOMAIN_MLFLOW_RAW_OFFENDERS = {
    "automl/project/cleanup.py",
}


def test_domains_do_not_call_mlflow_raw_directly():
    offenders = []
    for file_path in sorted(AUTOML_ROOT.rglob("*.py")):
        relative = _relative(file_path)
        if file_path.is_relative_to(AUTOML_ROOT / "mlflow"):
            continue
        text = file_path.read_text(encoding="utf-8")
        if "client.raw(" in text or "mlflow_client.raw(" in text:
            offenders.append(relative)

    unexpected = sorted(set(offenders) - KNOWN_DOMAIN_MLFLOW_RAW_OFFENDERS)
    assert unexpected == []
```

- [ ] **Step 2: Pin runner artifact paths before splitting**

Extend the existing runner integration assertions in
`tests/integration/runner/test_one_trial_local.py` so the path set includes the
paths Wave C must preserve:

```python
expected_paths = {
    "data/contract.json",
    "eval/manifest.json",
    "features/dataset_feature_registry.csv",
    "features/feature_registry.csv",
    "model/MLmodel",
    "timing/summary.json",
    "validation/data/input.csv",
    "validation/data/input.parquet",
    "validation/data/expected.parquet",
    "validation/latency_detail.json",
    "validation/report.json",
    "manifest.json",
}
assert expected_paths <= artifact_paths
```

If a proposal artifact is present, keep asserting
`agent/proposer/proposal.json`.

- [ ] **Step 3: Pin cleanup and agent timeline behavior before moving seams**

Make the cleanup integration assert that hard/soft delete still records the same
MLflow experiment/run target names and GCS/local prefixes. Make
`tests/unit/agent/test_timeline.py` assert:

```python
assert result["event"]["phase"] == "proposer"
assert result["event"]["agent_type"] == "automl-proposer"
assert "agent/proposer/report.json" in artifact_paths
assert "agent/coder/report.json" in artifact_paths
```

- [ ] **Step 4: Run and commit**

```bash
uv run pytest tests/contracts/test_architecture.py tests/integration/runner/test_one_trial_local.py tests/integration/cleanup tests/unit/agent/test_timeline.py tests/unit/skills/test_render_context_routes.py -q
git add tests/contracts/test_architecture.py tests/integration/runner/test_one_trial_local.py tests/integration/cleanup tests/unit/agent/test_timeline.py tests/unit/skills/test_render_context_routes.py
git commit -m "test(seams): characterize Wave C preservation points"
```

### Task C2: Move project cleanup MLflow CRUD behind seam verbs

**Files:**
- Modify: `automl/mlflow/client.py`
- Modify: `automl/project/cleanup.py`
- Modify: `tests/unit/project/test_cleanup.py`
- Modify: `tests/contracts/test_architecture.py`

- [ ] **Step 1: Add seam verbs in `automl/mlflow/client.py`**

Add small wrappers that are deliberately MLflow-specific and raise
`StorageError` from inside the seam:

```python
def get_experiment_by_name(name: str):
    try:
        return raw().get_experiment_by_name(name)
    except Exception as exc:
        raise StorageError(f"Failed to get MLflow experiment {name!r}") from exc


def delete_experiment(experiment_id: str) -> None:
    try:
        raw().delete_experiment(experiment_id)
    except Exception as exc:
        raise StorageError(f"Failed to delete MLflow experiment {experiment_id!r}") from exc


def delete_run(run_id: str) -> None:
    try:
        raw().delete_run(run_id)
    except Exception as exc:
        raise StorageError(f"Failed to delete MLflow run {run_id!r}") from exc


def run_start_time(run_id: str) -> int | None:
    try:
        return getattr(raw().get_run(run_id).info, "start_time", None)
    except Exception as exc:
        raise StorageError(f"Failed to read MLflow run {run_id!r}") from exc
```

Export them from `__all__`.

- [ ] **Step 2: Replace cleanup raw calls**

In `automl/project/cleanup.py`, replace:

```python
mlflow = mlflow_client.raw()
found = mlflow.get_experiment_by_name(name)
mlflow.delete_experiment(experiment_id)
mlflow.delete_run(run_id)
run = mlflow_client.raw().get_run(run_id)
```

with the new seam verbs. Preserve the current continue-and-record behavior:

```python
try:
    found = mlflow_client.get_experiment_by_name(name)
    ...
    mlflow_client.delete_experiment(experiment_id)
except Exception as exc:
    mlflow_experiments[name] = f"failed: {exc}"
```

For trial GCS partition time, keep the existing fallback-to-now behavior:

```python
try:
    start_time = mlflow_client.run_start_time(run_id)
    if start_time:
        partition_time = datetime.fromtimestamp(start_time / 1000, UTC)
except Exception:
    pass
```

- [ ] **Step 3: Remove the contract offender**

In `tests/contracts/test_architecture.py`, change:

```python
KNOWN_DOMAIN_MLFLOW_RAW_OFFENDERS = set()
```

- [ ] **Step 4: Verify and commit**

```bash
uv run pytest tests/unit/project/test_cleanup.py tests/integration/cleanup tests/contracts/test_architecture.py -q
git add automl/mlflow/client.py automl/project/cleanup.py tests/unit/project/test_cleanup.py tests/contracts/test_architecture.py
git commit -m "refactor(cleanup): route mlflow crud through seam"
```

### Task C3: Move `ProjectOverview` to the project domain

**Files:**
- Create: `automl/project/overview.py`
- Modify: `automl/project/__init__.py`
- Modify: `automl/mlflow/project/overview.py`
- Modify: `automl/mlflow/project/__init__.py`
- Modify: `tests/unit/mlflow/test_client_and_routing.py` or
  `tests/unit/mlflow/test_project_overview.py`

- [ ] **Step 1: Create the domain type**

Create `automl/project/overview.py`:

```python
"""Project overview domain value objects."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProjectOverview:
    schema_version: int = 1
    project_name: str = ""
    created_at: str = ""
    current_experiment_id: str | None = None
    dataset_count: int = 0


__all__ = ["ProjectOverview"]
```

Export it from `automl/project/__init__.py`.

- [ ] **Step 2: Import the type from the domain**

In `automl/mlflow/project/overview.py`, delete the local dataclass and import:

```python
from automl.project.overview import ProjectOverview
```

Keep return payloads byte-for-byte equivalent.

- [ ] **Step 3: Verify and commit**

```bash
uv run pytest tests/unit/mlflow tests/unit/project -q
git add automl/project/overview.py automl/project/__init__.py automl/mlflow/project/overview.py automl/mlflow/project/__init__.py tests/unit/mlflow tests/unit/project
git commit -m "refactor(project): own ProjectOverview domain type"
```

### Task C4: Move eval dataset GCS reads/writes behind an experiment seam module

**Files:**
- Create: `automl/mlflow/experiment/eval_datasets.py`
- Modify: `automl/mlflow/experiment/__init__.py`
- Modify: `automl/eval/prepare.py`
- Modify: `automl/eval/_load.py`
- Modify: `automl/eval/registry.py`
- Test: `tests/unit/eval/test_eval_thin_path.py`
- Test: `tests/integration/eval/test_eval_dataset_persistence.py`
- Test: `tests/integration/eval/test_augmentation_integration.py`

- [ ] **Step 1: Create experiment-scoped eval persistence seam functions**

Create `automl/mlflow/experiment/eval_datasets.py` with wrappers around the
existing GCS operations:

```python
"""Experiment-scoped eval dataset persistence seam."""

from __future__ import annotations

from typing import Any

import pandas as pd

from automl.errors import StorageError
from automl.utils.io import gcs


def read_manifest(uri: str) -> dict[str, Any]:
    try:
        return gcs.read_json(uri)
    except Exception as exc:
        raise StorageError(f"Failed to read eval manifest {uri!r}") from exc


def write_manifest(uri: str, payload: dict[str, Any], *, overwrite: bool = False) -> None:
    try:
        gcs.write_json(uri, payload, overwrite=overwrite)
    except Exception as exc:
        raise StorageError(f"Failed to write eval manifest {uri!r}") from exc


def read_frame(uri: str) -> pd.DataFrame:
    try:
        return gcs.read_parquet(uri)
    except Exception as exc:
        raise StorageError(f"Failed to read eval frame {uri!r}") from exc


def write_frame(uri: str, frame: pd.DataFrame, *, overwrite: bool = False) -> None:
    try:
        gcs.write_parquet(uri, frame, overwrite=overwrite)
    except Exception as exc:
        raise StorageError(f"Failed to write eval frame {uri!r}") from exc


def blob_exists(uri: str) -> bool:
    return gcs.blob_exists(uri)


def list_blob_names(uri: str) -> list[str]:
    return gcs.list_blob_names(uri)


def list_prefixes(uri: str) -> list[str]:
    return gcs.list_prefixes(uri)


__all__ = [
    "blob_exists",
    "list_blob_names",
    "list_prefixes",
    "read_frame",
    "read_manifest",
    "write_frame",
    "write_manifest",
]
```

- [ ] **Step 2: Replace eval direct GCS calls**

In eval domain files, import
`automl.mlflow.experiment.eval_datasets as experiment_eval_datasets` and replace:

```python
gcs.read_json(...)       -> experiment_eval_datasets.read_manifest(...)
gcs.write_json(...)      -> experiment_eval_datasets.write_manifest(...)
gcs.read_parquet(...)    -> experiment_eval_datasets.read_frame(...)
gcs.write_parquet(...)   -> experiment_eval_datasets.write_frame(...)
gcs.blob_exists(...)     -> experiment_eval_datasets.blob_exists(...)
gcs.list_blob_names(...) -> experiment_eval_datasets.list_blob_names(...)
gcs.list_prefixes(...)   -> experiment_eval_datasets.list_prefixes(...)
```

This matches the current pattern: project data artifacts live under
`automl/mlflow/project/artifacts.py`, trial artifacts live under
`automl/mlflow/trial/artifacts/*`, and eval datasets are experiment-scoped
durable objects. Do not change URI construction. If any test shows changed URI
strings, stop.

- [ ] **Step 3: Verify and commit**

```bash
uv run pytest tests/unit/eval tests/integration/eval -q
git add automl/mlflow/experiment/eval_datasets.py automl/mlflow/experiment/__init__.py automl/eval/prepare.py automl/eval/_load.py automl/eval/registry.py tests/unit/eval tests/integration/eval
git commit -m "refactor(eval): route dataset persistence through experiment seam"
```

### Task C5: Deduplicate MLflow JSON artifact path normalization

**Files:**
- Create: `automl/mlflow/artifact_paths.py`
- Modify: `automl/mlflow/experiment/logging.py`
- Modify: `automl/mlflow/trial/logging.py`
- Modify: `automl/mlflow/project/artifacts.py`
- Test: `tests/unit/mlflow/test_artifact_paths.py`

- [ ] **Step 1: Add shared helper**

Create `automl/mlflow/artifact_paths.py`:

```python
"""MLflow artifact path normalization helpers."""

from __future__ import annotations


def json_artifact_path(name: str) -> str:
    path = name.strip("/")
    return path if path.endswith(".json") else f"{path}.json"


__all__ = ["json_artifact_path"]
```

- [ ] **Step 2: Replace the three private helpers**

Delete `_json_artifact_path` from:
- `automl/mlflow/experiment/logging.py`
- `automl/mlflow/trial/logging.py`
- `automl/mlflow/project/artifacts.py`

Import and call:

```python
from automl.mlflow.artifact_paths import json_artifact_path
```

- [ ] **Step 3: Verify and commit**

```bash
uv run pytest tests/unit/mlflow -q
git add automl/mlflow/artifact_paths.py automl/mlflow/experiment/logging.py automl/mlflow/trial/logging.py automl/mlflow/project/artifacts.py tests/unit/mlflow/test_artifact_paths.py
git commit -m "refactor(mlflow): share json artifact path helper"
```

### Task C6: Split runner timing out of `runner/artifacts.py`

**Files:**
- Create: `automl/runner/timing.py`
- Modify: `automl/runner/artifacts.py`
- Modify: `automl/runner/trial.py`
- Modify: `automl/runner/__init__.py`
- Test: `tests/unit/runner/test_timing.py`
- Existing tests: `tests/integration/runner/test_one_trial_local.py`

- [ ] **Step 1: Move timing code**

Create `automl/runner/timing.py` with `TimingRecorder` and `timed_phase`:

```python
"""Runner timing helpers."""

from __future__ import annotations

import time
from contextlib import contextmanager


class TimingRecorder:
    def __init__(self) -> None:
        self.started = time.monotonic()
        self.phases: dict[str, float] = {}
        self.last_phase: str | None = None

    @contextmanager
    def phase(self, name: str):
        self.last_phase = name
        started = time.monotonic()
        try:
            yield
        finally:
            self.phases[name] = self.phases.get(name, 0.0) + (
                time.monotonic() - started
            )

    def snapshot(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "unit": "seconds",
            "total_seconds": max(0.0, time.monotonic() - self.started),
            "phases": dict(self.phases),
        }


@contextmanager
def timed_phase(timing: TimingRecorder | None, name: str):
    if timing is None:
        yield
        return
    with timing.phase(name):
        yield
```

In `runner/artifacts.py`, import `TimingRecorder` and `timed_phase`; replace
`_timed_phase(...)` calls with `timed_phase(...)`; remove local timing code.

- [ ] **Step 2: Keep public imports stable**

In `runner/artifacts.py`, keep:

```python
from automl.runner.timing import TimingRecorder
```

so existing `from automl.runner.artifacts import TimingRecorder` continues to
work during the split.

- [ ] **Step 3: Verify and commit**

```bash
uv run pytest tests/unit/runner/test_timing.py tests/unit/runner tests/integration/runner/test_one_trial_local.py -q
git add automl/runner/timing.py automl/runner/artifacts.py automl/runner/trial.py automl/runner/__init__.py tests/unit/runner/test_timing.py
git commit -m "refactor(runner): split timing helpers"
```

### Task C7: Split serving validation out of `runner/artifacts.py`

**Files:**
- Create: `automl/runner/serving_validation.py`
- Modify: `automl/runner/artifacts.py`
- Test: `tests/unit/runner/test_validation_errors.py`
- Test: `tests/integration/runner/test_one_trial_local.py`

- [ ] **Step 1: Move validation subsystem**

Move these symbols from `runner/artifacts.py` into
`runner/serving_validation.py`:

```python
log_validation_artifacts
_validation_input_frame
_predict_model
_series_values
_input_schema_from_frame
_run_pyfunc_validation
_validation_report_document
_validation_status
_log_validation_tags_and_metrics
```

Keep imports explicit:

```python
from automl.runner.timing import TimingRecorder, timed_phase
from automl.errors import ValidationError
from automl.mlflow import trial as mlflow_trial
```

In `runner/artifacts.py`, re-export the moved public function:

```python
from automl.runner.serving_validation import log_validation_artifacts
```

- [ ] **Step 2: Preserve validation artifacts**

Run the existing integration checks and confirm the same artifacts still exist:

```python
"validation/data/input.csv"
"validation/data/input.parquet"
"validation/data/expected.parquet"
"validation/latency_detail.json"
"validation/report.json"
```

- [ ] **Step 3: Verify and commit**

```bash
uv run pytest tests/unit/runner/test_validation_errors.py tests/integration/runner/test_one_trial_local.py -q
git add automl/runner/serving_validation.py automl/runner/artifacts.py tests/unit/runner/test_validation_errors.py
git commit -m "refactor(runner): split serving validation"
```

### Task C7.5: Correct trial/runner ownership boundary

**Files:**
- Create/move: `automl/trial/paths.py`
- Create/move: `automl/trial/template.py`
- Delete: `automl/runner/paths.py`
- Delete: `automl/runner/template.py`
- Modify: `automl/trial/create.py`
- Modify: `automl/trial/fork.py`
- Modify: `automl/trial/promote.py`
- Modify: `automl/runner/trial.py`
- Modify: `automl/cli/trial.py`
- Test: `tests/contracts/test_architecture.py`
- Test: affected `tests/unit/trial`, `tests/unit/runner`, and `tests/unit/cli`

This task corrects the stale pre-C8 design. Trial owns authoring artifacts and
pure trial schemas/read types; runner owns execution. Surface/CLI code may
compose both. Core trial modules must not import runner.

- [ ] **Step 1: Move draft path and template ownership to trial**

Create or move `automl.trial.paths` and `automl.trial.template` as the only homes
for draft trial path helpers, `verify_trial_dir`, and the generated `run.py`
template. Delete `automl.runner.paths` and `automl.runner.template`; do not leave
re-export or import compatibility shims under `automl.runner`.

- [ ] **Step 2: Point trial authoring at trial-owned helpers**

Update `trial.create` and `trial.fork` to use `automl.trial.paths` and
`automl.trial.template` directly. These modules own draft path construction,
template rendering, metadata/schema creation, and per-trial authoring behavior.

- [ ] **Step 3: Keep runner execution-only**

Update `runner.trial` to consume `automl.trial.paths.verify_trial_dir` only as a
pure schema/path check before execution. Runner may import approved pure trial
leaves such as metadata/manifest/path verification, but must not import trial
workflow orchestration or template-authoring behavior.

- [ ] **Step 4: Remove trial-core create+run orchestration**

Remove or relocate `trial.promote` core create+run orchestration so the trial
domain no longer imports `automl.runner` or calls `runner.run_trial` as trial
domain behavior. If the existing CLI command remains, make the CLI/workflow
surface compose "create/fork/promote authoring" and then "run one trial" by
calling trial + runner from that surface layer.

- [ ] **Step 5: Replace architecture contracts with the approved boundary**

Add/update architecture tests to assert:
- no `automl.trial.*` module imports `automl.runner`;
- `automl.runner` may import only approved pure trial leaves, including path
  verification and schemas/read types needed for execution;
- `automl.runner.paths` and `automl.runner.template` do not exist;
- no runner-owned path/template compatibility shims remain.

- [ ] **Step 6: Verify and commit**

```bash
uv run pytest tests/unit/trial tests/unit/runner tests/unit/cli tests/contracts -q
git add automl/trial/paths.py automl/trial/template.py automl/trial/create.py automl/trial/fork.py automl/trial/promote.py automl/runner/trial.py automl/runner/paths.py automl/runner/template.py automl/cli/trial.py tests/contracts/test_architecture.py tests/unit/trial tests/unit/runner tests/unit/cli
git commit -m "refactor(trial): own draft paths and templates"
```

### Task C8: Move runner artifact writes to trial seam verbs and type current manifest

**Files:**
- Create: `automl/mlflow/trial/artifacts/runner.py`
- Create: `automl/trial/manifest.py`
- Modify: `automl/mlflow/trial/artifacts/__init__.py`
- Modify: `automl/runner/artifacts.py`
- Modify: `automl/runner/serving_validation.py`
- Modify: `automl/trial/metadata.py`
- Test: `tests/unit/mlflow/test_runner_artifacts.py`
- Test: `tests/unit/trial/test_manifest.py`
- Test: `tests/integration/runner/test_one_trial_local.py`

- [ ] **Step 1: Add typed seam writer functions**

Create `automl/mlflow/trial/artifacts/runner.py`:

```python
"""Runner-owned trial artifact writers."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from automl.mlflow import client
from automl.mlflow.trial.logging import log_json
from automl.trial.metadata import TimingReport


def write_local_file(run_id: str, artifact_path: str, local_path: Path) -> None:
    client.log_artifact_file(run_id, artifact_path, local_path)


def write_timing(run_id: str, timing: Mapping[str, object]) -> None:
    report = TimingReport.from_dict(timing).to_dict()
    log_json(run_id, "timing/summary", report)
```

Export the module from `automl/mlflow/trial/artifacts/__init__.py`.

- [ ] **Step 2: Move the trial run manifest schema to the trial domain**

Create `automl/trial/manifest.py` as the schema for the existing runner
`manifest.json` artifact. This schema must model the current payload shape, not
the older minimal `TrialManifest` currently parked in `automl/trial/metadata.py`.
This follows the revised C7.5 boundary: runner may import this pure trial-owned
schema, but trial must not import runner to build or execute workflows.

Use the current payload sections from `runner/artifacts.py::log_manifest`:

```python
"""Trial run manifest artifact schema."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class TrialRunManifest:
    run: Mapping[str, Any]
    data: Mapping[str, Any]
    model: Mapping[str, Any]
    evaluation: Mapping[str, Any]
    validation: Mapping[str, Any]
    timing: Mapping[str, Any]
    artifacts: tuple[Mapping[str, Any], ...]
    schema_version: int = 1

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TrialRunManifest":
        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            run=dict(payload.get("run", {})),
            data=dict(payload.get("data", {})),
            model=dict(payload.get("model", {})),
            evaluation=dict(payload.get("evaluation", {})),
            validation=dict(payload.get("validation", {})),
            timing=dict(payload.get("timing", {})),
            artifacts=tuple(dict(item) for item in payload.get("artifacts", ())),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run": dict(self.run),
            "data": dict(self.data),
            "model": dict(self.model),
            "evaluation": dict(self.evaluation),
            "validation": dict(self.validation),
            "timing": dict(self.timing),
            "artifacts": [dict(item) for item in self.artifacts],
        }


__all__ = ["TrialRunManifest"]
```

Remove the mismatched `ManifestEntry` and `TrialManifest` classes from
`automl/trial/metadata.py`; they do not describe the logged runner manifest.

Add `tests/unit/trial/test_manifest.py` with a fixture matching the current
runner `manifest.json` payload. Assert:

```python
assert TrialRunManifest.from_dict(payload).to_dict() == payload
```

- [ ] **Step 3: Wire runner wrappers to seam verbs**

In `runner/artifacts.py` and `runner/serving_validation.py`, replace direct calls
to `mlflow_client.log_artifact_file` with:

```python
from automl.mlflow.trial.artifacts import runner as runner_artifacts

runner_artifacts.write_local_file(run_id, artifact_path, local_path)
```

For timing, normalize through `TimingReport.from_dict(timing).to_dict()` before
writing `timing/summary.json` through `runner_artifacts.write_timing(...)`. If
that changes the JSON payload compared to the characterization tests, stop and
ask before changing logged timing JSON.

- [ ] **Step 4: Type the manifest without changing the artifact**

In `runner/artifacts.py::log_manifest`, after building the existing payload,
normalize through:

```python
from automl.trial.manifest import TrialRunManifest

payload = TrialRunManifest.from_dict(payload).to_dict()
```

Then keep:

```python
trial_artifacts.write_manifest(run_id, payload)
```

The characterization tests from Task C1 must prove the logged `manifest.json`
still contains the same top-level keys and nested paths. If the typed schema
changes the payload shape, stop and leave the manifest free-form.

- [ ] **Step 5: Verify and commit**

```bash
uv run pytest tests/unit/mlflow/test_runner_artifacts.py tests/unit/trial/test_manifest.py tests/integration/runner/test_one_trial_local.py -q
git add automl/mlflow/trial/artifacts/runner.py automl/mlflow/trial/artifacts/__init__.py automl/runner/artifacts.py automl/runner/serving_validation.py automl/trial/manifest.py automl/trial/metadata.py tests/unit/mlflow/test_runner_artifacts.py tests/unit/trial/test_manifest.py
git commit -m "refactor(runner): type manifest and use trial artifact seam"
```

### Task C9: Replace runner hand-rolled metadata parsing with `TrialMetadata.read`

**Files:**
- Modify: `automl/runner/trial.py`
- Test: `tests/unit/runner/test_trial_metadata.py`
- Existing tests: `tests/integration/runner/test_one_trial_local.py`

- [ ] **Step 1: Replace `_RunnerTrialMetadata`**

In `runner/trial.py`, delete the private `_RunnerTrialMetadata` dataclass and
JSON parser. Import:

```python
from automl.trial.metadata import TrialMetadata
```

Where the runner currently builds `_RunnerTrialMetadata`, call:

```python
metadata = TrialMetadata.read(metadata_path)
```

Keep all existing fallback behavior for missing optional fields.
This is allowed by the C7.5 pure-trial-leaf rule: `TrialMetadata` is a
trial-owned read type/schema, not trial workflow orchestration. Do not reintroduce
a blanket "runner cannot import trial" rule; instead enforce the narrower
approved import allowlist in the architecture test.

- [ ] **Step 2: Verify and commit**

```bash
uv run pytest tests/unit/runner/test_trial_metadata.py tests/integration/runner/test_one_trial_local.py -q
git add automl/runner/trial.py tests/unit/runner/test_trial_metadata.py
git commit -m "refactor(runner): read trial metadata via domain schema"
```

### Task C10: Split `agent/timeline.py` into a package

**Files:**
- Delete: `automl/agent/timeline.py`
- Create: `automl/agent/timeline/__init__.py`
- Create: `automl/agent/timeline/paths.py`
- Create: `automl/agent/timeline/ingest.py`
- Create: `automl/agent/timeline/reconcile.py`
- Create: `automl/agent/timeline/publish.py`
- Test: `tests/unit/agent/test_timeline.py`
- Test: `tests/e2e/test_phase5_agent_hooks.py`

- [ ] **Step 1: Create package boundaries**

Move code by responsibility:
- `paths.py`: `_route_segment`, `_timeline_dir`, `_timeline_path`,
  `_session_dir`, `_trial_dir`
- `ingest.py`: `handle_event`, `_event_from_hook_payload`,
  `_now_event_fields`, `_append_event`, `_should_publish_trial_on_hook_event`
- `reconcile.py`: `_read_events`, `_latest_session_id_from_events`,
  `_events_for_session`, `_summarize_events`, matching/tool-count helpers
- `publish.py`: `publish`, staging, GCS upload, MLflow publishing helpers
- `__init__.py`: re-export only `handle_event` and `publish`

Keep the public import stable:

```python
from automl.agent.timeline import handle_event, publish
```

- [ ] **Step 2: Preserve timeline file paths**

After the move, the path formula must still be:

```python
project_root / ".cache" / "automl" / "tmp" / "timelines" / route_segments / "agent_timeline.jsonl"
```

Run `tests/unit/agent/test_timeline.py::test_handle_event_appends_route_scoped_hook_event`
before committing.

- [ ] **Step 3: Verify and commit**

```bash
uv run pytest tests/unit/agent/test_timeline.py tests/e2e/test_phase5_agent_hooks.py -q
git add automl/agent/timeline.py automl/agent/timeline tests/unit/agent/test_timeline.py tests/e2e/test_phase5_agent_hooks.py
git commit -m "refactor(agent): split timeline package"
```

### Task C11: Wave C gate, review, status update

- [ ] **Step 1: Run full Wave C gate**

```bash
uv run pytest tests/unit tests/integration tests/contracts -q
```

Expected: all tests pass.

- [ ] **Step 2: Run structural acceptance checks**

```bash
wc -l automl/runner/artifacts.py
test "$(wc -l < automl/runner/artifacts.py)" -lt 250
test -d automl/agent/timeline
rg -n "client\.raw\(|mlflow_client\.raw\(" automl | grep -v "automl/mlflow" || true
uv run automl --help >/tmp/automl-help.txt
uv run python hooks/agent_timeline.py publish --help >/tmp/automl-hook-publish-help.txt
```

Expected:
- `runner/artifacts.py` is below 250 lines.
- `automl/agent/timeline/` exists and public imports still work.
- no domain `client.raw()` calls outside `automl/mlflow/`.
- no `automl agent` CLI noun was added.
- existing `hooks/agent_timeline.py publish --help` exits 0.

- [ ] **Step 3: Request review**

Use `superpowers:requesting-code-review`. Review focus:
- MLflow/GCS artifact path preservation;
- no raw MLflow use outside the seam;
- `runner/artifacts.py` and timeline package boundaries are real, not cosmetic;
- no new CLI nouns or role registry slipped into the wave;
- existing hook publish compatibility is preserved for current skill workflows.

- [ ] **Step 4: Update STATUS and commit**

After review fixes:

```bash
git add docs/execution/STATUS.md
git commit -m "docs(execution): mark Wave C complete"
```

## Wave D — CLI discipline + validation uniformity — DETAILED

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking. Wave D is not approved until
> the user explicitly says "go".

**Goal:** Make the CLI a thin, predictable surface and make validation targets
delegate uniformly to domain-owned checks.

**Architecture:** Reserve `--json` for the one command that actually branches on
it (`experiment run`); all other CLI verbs keep JSON output but drop the dead
flag. Keep command coordination in domain modules (`agent`, `trial`, `runner`,
`project`, `model`) and leave `automl/cli/*` as parser/adapter code. Validation
remains a framework orchestrator that lazily calls per-domain `checks.py`
functions wrapped by `_safe()`.

**Tech Stack:** Python 3.13, argparse, dataclasses, pytest, MLflow seam already
covered by Waves B/C.

**Design-alignment risk to approve before execution:** this plan treats the
`--json` flag as output-format syntax. That means `validate proposal --json
<path>` becomes `validate proposal --proposal-json <path>`, and skill-facing
safe commands are updated at the same time. Approving Wave D approves that
rename and removal of the old dead flags.

---

### Task D0: Baseline and current-surface inventory

**Files:** none

- [ ] **Step 1: Run the wave baseline**

```bash
uv run pytest tests/unit tests/contracts -q
```

Expected before editing: pass. If this fails, stop and debug before touching
Wave D code.

- [ ] **Step 2: Record the current parser surface before edits**

Run:

```bash
rg -n 'add_argument\("--json"' automl/cli tests skills references
uv run automl --help >/tmp/wave-d-automl-help.txt
uv run automl validate proposal --help >/tmp/wave-d-validate-proposal-help.txt
uv run automl experiment run --help >/tmp/wave-d-experiment-run-help.txt
```

Expected before editing: the `rg` command shows current `--json` parser uses;
the help commands exit 0. Do not commit D0. The command output is a local
baseline for D1.

### Task D1: Remove dead `--json` flags and update skill commands

**Files:**
- Modify: `automl/cli/project.py`
- Modify: `automl/cli/experiment.py`
- Modify: `automl/cli/trial.py`
- Modify: `automl/cli/data.py`
- Modify: `automl/cli/eval.py`
- Modify: `automl/cli/validate.py`
- Modify: `skills/automl/scripts/render_context.py`
- Modify: `skills/propose/SKILL.md`
- Modify: `references/loop/mlflow-context.md`
- Modify: `tests/unit/cli/test_phase6_cli_catalog.py`
- Modify: `tests/unit/cli/test_agent_phase_cli.py`
- Modify: `tests/e2e/test_phase6_surface_isolation.py`
- Modify: `tests/contracts/test_phase6_skill_commands.py`

- [ ] **Step 1: Add failing parser-surface tests**

Append these helpers and tests to `tests/unit/cli/test_phase6_cli_catalog.py`:

```python
def _subparser(parser, *path):
    current = parser
    for name in path:
        actions = [item for item in current._actions if hasattr(item, "choices")]
        choices = {}
        for action in actions:
            choices.update(getattr(action, "choices", {}) or {})
        current = choices[name]
    return current


def _option_strings(parser):
    return {
        option
        for action in parser._actions
        for option in getattr(action, "option_strings", ())
    }


def test_json_flag_is_reserved_for_experiment_run_output():
    from automl.cli import build_parser

    parser = build_parser()
    commands = {
        ("project", "list"),
        ("project", "deps"),
        ("project", "init"),
        ("project", "delete"),
        ("experiment", "list"),
        ("experiment", "delete"),
        ("experiment", "leaderboard"),
        ("experiment", "compare"),
        ("experiment", "summary"),
        ("experiment", "proposer-context"),
        ("trial", "list"),
        ("trial", "create"),
        ("trial", "fork"),
        ("trial", "promote"),
        ("trial", "run"),
        ("trial", "show"),
        ("trial", "delete"),
        ("trial", "lock", "acquire"),
        ("trial", "lock", "release"),
        ("data", "list"),
        ("data", "profile"),
        ("data", "materialize"),
        ("eval", "list"),
        ("eval", "compute"),
        ("validate", "project"),
        ("validate", "model"),
        ("validate", "proposal"),
    }

    assert "--json" in _option_strings(_subparser(parser, "experiment", "run"))
    offenders = [
        command
        for command in sorted(commands)
        if "--json" in _option_strings(_subparser(parser, *command))
    ]
    assert offenders == []


def test_validate_proposal_uses_named_input_path_not_output_json_flag():
    from automl.cli import build_parser

    proposal = _subparser(build_parser(), "validate", "proposal")

    assert "--proposal-json" in _option_strings(proposal)
    assert "--json" not in _option_strings(proposal)
```

- [ ] **Step 2: Add failing skill command tests**

Update `tests/contracts/test_phase6_skill_commands.py` so
`test_automl_render_context_safe_commands_use_phase6_cli_surface` asserts the
new safe-command shape:

```python
    assert " --json" not in safe_commands["loop_context"]
    assert " --json" not in safe_commands["prepare_snapshot"]
    assert " --json" not in safe_commands["persist_proposal"]
    assert " --json" not in safe_commands["validate_proposal"]
    assert " --proposal-json -" in safe_commands["persist_proposal"]
    assert " --proposal-json '<proposal.json>'" in safe_commands["validate_proposal"]
```

- [ ] **Step 3: Verify red**

```bash
uv run pytest \
  tests/unit/cli/test_phase6_cli_catalog.py::test_json_flag_is_reserved_for_experiment_run_output \
  tests/unit/cli/test_phase6_cli_catalog.py::test_validate_proposal_uses_named_input_path_not_output_json_flag \
  tests/contracts/test_phase6_skill_commands.py::test_automl_render_context_safe_commands_use_phase6_cli_surface \
  -q
```

Expected: fail on the current dead `--json` flags and current
`validate proposal --json`.

- [ ] **Step 4: Remove parser flags**

Remove every `add_argument("--json", ...)` except
`automl/cli/experiment.py`'s `experiment run` parser. In
`automl/cli/validate.py`, replace:

```python
proposal.add_argument("--json", required=True, help="Path to proposal JSON, or '-' for stdin")
```

with:

```python
proposal.add_argument(
    "--proposal-json",
    required=True,
    help="Path to proposal JSON, or '-' for stdin",
)
```

and change `_proposal` to read `args.proposal_json`.

- [ ] **Step 5: Update active skill command generation**

In `skills/automl/scripts/render_context.py`, remove `--json` from
`loop_context_args` and `prepare_args`. In the `persist_proposal` and
`validate_proposal` safe commands, replace `--json` with `--proposal-json`.

The command snippets must become:

```python
loop_context_args = [
    "uv",
    "run",
    "automl",
    *session_args,
    "experiment",
    "proposer-context",
]
prepare_args = [
    "uv",
    "run",
    "automl",
    *session_args,
    "data",
    "materialize",
]
```

and:

```python
"validate",
"proposal",
"--proposal-json",
"-",
```

- [ ] **Step 6: Update active prose that is executable**

In `skills/propose/SKILL.md`, change:

```text
uv run automl --project <project_name> --project-root <project_root> project deps --json
uv run automl --project <project_name> --project-root <project_root> experiment proposer-context --json
```

to:

```text
uv run automl --project <project_name> --project-root <project_root> project deps
uv run automl --project <project_name> --project-root <project_root> experiment proposer-context
```

In `references/loop/mlflow-context.md`, change both
`experiment proposer-context --json` examples to `experiment proposer-context`.

- [ ] **Step 7: Update CLI tests to use the new surface**

In `tests/unit/cli/test_phase6_cli_catalog.py`, remove `--json` from every
command except `experiment run`, and change `validate proposal --json -` to
`validate proposal --proposal-json -`. In `tests/unit/cli/test_agent_phase_cli.py`
and `tests/e2e/test_phase6_surface_isolation.py`, change
`validate proposal --json` to `validate proposal --proposal-json`; remove dead
`--json` flags from non-`experiment run` commands.

- [ ] **Step 8: Verify green for command surface**

```bash
uv run pytest \
  tests/unit/cli/test_phase6_cli_catalog.py \
  tests/unit/cli/test_agent_phase_cli.py \
  tests/contracts/test_phase6_skill_commands.py \
  tests/e2e/test_phase6_surface_isolation.py \
  -q
```

Expected: pass.

- [ ] **Step 9: Commit**

```bash
git add \
  automl/cli/project.py automl/cli/experiment.py automl/cli/trial.py \
  automl/cli/data.py automl/cli/eval.py automl/cli/validate.py \
  skills/automl/scripts/render_context.py skills/propose/SKILL.md \
  references/loop/mlflow-context.md \
  tests/unit/cli/test_phase6_cli_catalog.py tests/unit/cli/test_agent_phase_cli.py \
  tests/e2e/test_phase6_surface_isolation.py tests/contracts/test_phase6_skill_commands.py
git commit -m "refactor(cli): reserve json flag for experiment run"
```

### Task D2: Make `experiment run` own and forward its loop options

**Files:**
- Create: `automl/agent/run_options.py`
- Modify: `automl/cli/experiment.py`
- Modify: `skills/automl/scripts/preflight.py`
- Test: `tests/unit/agent/test_launch.py`
- Test: `tests/unit/cli/test_phase6_cli_catalog.py`

- [ ] **Step 1: Write failing forwarding tests**

Append to `tests/unit/cli/test_phase6_cli_catalog.py`:

```python
def test_experiment_run_forwards_loop_options_to_skill_command(monkeypatch, tmp_path):
    from automl.agent.launch import LaunchSpec
    from automl.cli import main

    active = types.SimpleNamespace(
        active_experiment_id="phase6",
        dry_run=False,
        namespace="",
        project_name="demo",
    )
    _patch_use_project(monkeypatch, active)
    launch_calls = []

    def fake_build_launch(**kwargs):
        launch_calls.append(kwargs)
        return LaunchSpec(command=["claude-test"], env={}, cwd=tmp_path)

    _patch_attr(monkeypatch, "automl.cli.experiment", "build_launch", fake_build_launch)
    monkeypatch.setattr(
        "automl.cli.experiment.subprocess.run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 0),
    )

    assert main([
        "--project", "demo",
        "--project-root", str(tmp_path),
        "experiment", "run",
        "--max-iter", "7",
        "--time-budget", "1.5",
        "--instruction", "prefer linear models",
    ]) == 0

    assert launch_calls[0]["automl_args"] == [
        "experiment", "run",
        "--project", "demo",
        "--max-iter", "7",
        "--time-budget", "1.5",
        "--instruction", "prefer linear models",
    ]
```

Append to `tests/unit/agent/test_launch.py`:

```python
def test_launch_command_preserves_explicit_loop_options(tmp_path, monkeypatch):
    active = _session(tmp_path)
    launch = build_launch(
        session=active,
        automl_args=[
            "experiment", "run",
            "--project", "demo",
            "--max-iter", "7",
            "--time-budget", "1.5",
            "--instruction", "prefer linear models",
        ],
        claude_bin="claude-test",
    )

    prompt = launch.command[-1]
    assert prompt == (
        "/brigit-automl:automl experiment run --project demo "
        "--max-iter 7 --time-budget 1.5 --instruction 'prefer linear models'"
    )
```

- [ ] **Step 2: Verify red**

```bash
uv run pytest \
  tests/unit/cli/test_phase6_cli_catalog.py::test_experiment_run_forwards_loop_options_to_skill_command \
  tests/unit/agent/test_launch.py::test_launch_command_preserves_explicit_loop_options \
  -q
```

Expected: first test fails because `experiment run` has no explicit
`--max-iter`, `--time-budget`, or `--instruction` parser options.

- [ ] **Step 3: Add one option owner**

Create `automl/agent/run_options.py`:

```python
"""Canonical AutoML experiment-run option parsing helpers."""

from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentRunOptions:
    project: str = ""
    dry_run: bool = False
    namespace: str = ""
    max_iter: int | None = None
    time_budget: float | None = None
    refresh_data: bool = False
    refresh_source: bool = False
    auto_confirm: bool = False
    instructions: tuple[str, ...] = ()


def add_experiment_run_options(
    parser: argparse.ArgumentParser,
    *,
    include_project_flags: bool = False,
    include_confirmation: bool = False,
) -> None:
    if include_project_flags:
        parser.add_argument("--project", default="")
        parser.add_argument("--dry-run", action="store_true")
        parser.add_argument("--namespace", default="")
    parser.add_argument("--max-iter", type=int, default=None)
    parser.add_argument("--time-budget", type=float, default=None)
    if include_confirmation:
        parser.add_argument("--auto-confirm", action="store_true")
    parser.add_argument("--refresh-data", action="store_true")
    parser.add_argument("--refresh-source", action="store_true")
    parser.add_argument("--instruction", "--constraint", action="append", default=[])


def options_from_namespace(args: argparse.Namespace) -> ExperimentRunOptions:
    return ExperimentRunOptions(
        project=str(getattr(args, "project", "") or ""),
        dry_run=bool(getattr(args, "dry_run", False)),
        namespace=str(getattr(args, "namespace", "") or ""),
        max_iter=getattr(args, "max_iter", None),
        time_budget=getattr(args, "time_budget", None),
        refresh_data=bool(getattr(args, "refresh_data", False)),
        refresh_source=bool(getattr(args, "refresh_source", False)),
        auto_confirm=bool(getattr(args, "auto_confirm", False)),
        instructions=tuple(
            item.strip()
            for item in getattr(args, "instruction", ()) or ()
            if str(item).strip()
        ),
    )


def skill_command_args(options: ExperimentRunOptions, *, project: str) -> list[str]:
    args = ["experiment", "run", "--project", project]
    if options.dry_run:
        args.append("--dry-run")
    if options.namespace:
        args.extend(["--namespace", options.namespace])
    if options.max_iter is not None:
        args.extend(["--max-iter", str(options.max_iter)])
    if options.time_budget is not None:
        args.extend(["--time-budget", str(options.time_budget)])
    if options.refresh_data:
        args.append("--refresh-data")
    if options.refresh_source:
        args.append("--refresh-source")
    if options.auto_confirm:
        args.append("--auto-confirm")
    for instruction in options.instructions:
        args.extend(["--instruction", instruction])
    return args


__all__ = [
    "ExperimentRunOptions",
    "add_experiment_run_options",
    "options_from_namespace",
    "skill_command_args",
]
```

- [ ] **Step 4: Wire CLI and preflight to the shared owner**

In `automl/cli/experiment.py`, import the helpers, remove the
`argparse.REMAINDER` positional, and add explicit run options:

```python
from automl.agent.run_options import add_experiment_run_options
from automl.agent.run_options import options_from_namespace, skill_command_args
```

Then change the run parser:

```python
run = experiment_sub.add_parser("run")
run.add_argument("experiment_id_arg", nargs="?")
run.add_argument("--max-budget-usd", default="5")
run.add_argument("--output-format", choices=["text", "json", "stream-json"], default="text")
run.add_argument("--claude-bin", default="claude")
run.add_argument("--json", action="store_true")
add_experiment_run_options(run)
run.set_defaults(func=_run)
```

Change `_run` to build the slash-command argument vector:

```python
options = options_from_namespace(args)
automl_args = skill_command_args(options, project=active.project_name)
launch = build_launch(
    session=active,
    automl_args=automl_args,
    max_budget_usd=args.max_budget_usd,
    output_format=args.output_format,
    claude_bin=args.claude_bin,
)
```

In `skills/automl/scripts/preflight.py`, import
`add_experiment_run_options` and `options_from_namespace`, use
`add_experiment_run_options(parser, include_project_flags=True,
include_confirmation=True)`, and build the payload from
`options_from_namespace(parsed)`.

- [ ] **Step 5: Verify green**

```bash
uv run pytest tests/unit/agent/test_launch.py tests/unit/cli/test_phase6_cli_catalog.py tests/unit/skills/test_render_context_routes.py tests/contracts/test_phase6_skill_commands.py -q
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add automl/agent/run_options.py automl/cli/experiment.py \
  skills/automl/scripts/preflight.py tests/unit/agent/test_launch.py \
  tests/unit/cli/test_phase6_cli_catalog.py tests/unit/skills/test_render_context_routes.py \
  tests/contracts/test_phase6_skill_commands.py
git commit -m "fix(agent): forward experiment run loop options"
```

### Task D3: Make data materialization return shape explicit

**Files:**
- Modify: `automl/data/pipeline.py`
- Modify: `automl/cli/data.py`
- Test: `tests/unit/data/test_materialize_return_shape.py`
- Test: `tests/unit/cli/test_phase6_cli_catalog.py`

- [ ] **Step 1: Write failing core return-shape test**

Create `tests/unit/data/test_materialize_return_shape.py`:

```python
from __future__ import annotations

from contextlib import nullcontext

import pandas as pd

from automl.data import ComponentHashes, Dataset, FeatureRegistry, LoadedDataset


def _dataset() -> Dataset:
    return Dataset(
        id="ds_001",
        identity_hash="hash",
        component_hashes=ComponentHashes(
            source_identity="s",
            feature_registry="f",
            data_content="d",
            schema="sc",
        ),
        gcs_bucket="bucket",
        gcs_prefix="root/demo",
        project_name="demo",
        created_at="2026-05-29T00:00:00+00:00",
        source_identity={"source": "unit"},
        n_rows=2,
        n_columns=2,
        target_column="target",
        split_id_col="SPLITID",
        hash_key=("id",),
    )


def test_materialize_can_return_dataset_metadata_without_rows(monkeypatch):
    from automl.data import pipeline

    dataset = _dataset()
    loaded = LoadedDataset(
        dataset=dataset,
        df=pd.DataFrame({"secret_row_value": ["do-not-print", "do-not-print"]}),
        registry=FeatureRegistry(),
    )
    active = type("Active", (), {"active_experiment_id": "exp"})()
    monkeypatch.setattr(pipeline, "_session", lambda explicit: active)
    monkeypatch.setattr(pipeline.mlflow_client, "bound_for", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(pipeline, "_materialize_bound", lambda **kwargs: loaded)

    assert pipeline.materialize(session=active, include_rows=False) == dataset
    assert pipeline.materialize(session=active, include_rows=True) == loaded
```

- [ ] **Step 2: Write failing CLI no-row-output test**

Append to `tests/unit/cli/test_phase6_cli_catalog.py`:

```python
def test_data_materialize_prints_dataset_manifest_not_loaded_rows(monkeypatch, tmp_path, capsys):
    from automl.cli import main
    from automl.data import ComponentHashes, Dataset

    active = types.SimpleNamespace(active_experiment_id="exp")
    _patch_use_project(monkeypatch, active)
    dataset = Dataset(
        id="ds_001",
        identity_hash="hash",
        component_hashes=ComponentHashes(
            source_identity="s",
            feature_registry="f",
            data_content="d",
            schema="sc",
        ),
        gcs_bucket="bucket",
        gcs_prefix="root/demo",
        project_name="demo",
        created_at="2026-05-29T00:00:00+00:00",
        source_identity={"source": "unit"},
        n_rows=2,
        n_columns=2,
        target_column="target",
        split_id_col="SPLITID",
        hash_key=("id",),
    )

    def fake_materialize(**kwargs):
        assert kwargs["include_rows"] is False
        return dataset

    _patch_attr(monkeypatch, "automl.cli.data", "materialize", fake_materialize)

    assert main(["--project", "demo", "--project-root", str(tmp_path), "data", "materialize"]) == 0

    output = capsys.readouterr().out
    payload = json.loads(output)
    assert payload["id"] == "ds_001"
    assert payload["n_rows"] == 2
    assert "secret_row_value" not in output
```

- [ ] **Step 3: Verify red**

```bash
uv run pytest \
  tests/unit/data/test_materialize_return_shape.py::test_materialize_can_return_dataset_metadata_without_rows \
  tests/unit/cli/test_phase6_cli_catalog.py::test_data_materialize_prints_dataset_manifest_not_loaded_rows \
  -q
```

Expected: fail because `materialize()` does not accept `include_rows` yet and
the CLI does not pass it.

- [ ] **Step 4: Add the core return-shape parameter**

Change `automl/data/pipeline.py::materialize` to preserve the existing default
and make metadata-only return explicit:

```python
def materialize(
    *,
    refresh_source: bool = False,
    include_rows: bool = True,
    session: Session | None = None,
) -> LoadedDataset | Dataset:
    """Persist the active dataset.

    Set `include_rows=False` to return only the Dataset manifest metadata after
    persistence. This controls the returned object shape; it does not change the
    materialization pipeline's persistence behavior.
    """
    active = _session(session)
    with mlflow_client.bound_for(active, experiment_id=active.active_experiment_id):
        loaded = _materialize_bound(active=active, refresh_source=refresh_source)
    return loaded if include_rows else loaded.dataset
```

- [ ] **Step 5: Make the CLI request metadata-only output**

Change `automl/cli/data.py::_materialize`:

```python
def _materialize(args: argparse.Namespace) -> int:
    dataset = materialize(
        refresh_source=args.refresh_source,
        include_rows=False,
        session=session_from_args(args),
    )
    print_json(dataset)
    return 0
```

- [ ] **Step 6: Verify green**

```bash
uv run pytest \
  tests/unit/data/test_materialize_return_shape.py \
  tests/unit/cli/test_phase6_cli_catalog.py::test_data_materialize_prints_dataset_manifest_not_loaded_rows \
  tests/unit/cli/test_phase6_cli_catalog.py \
  -q
```

Expected: pass.

- [ ] **Step 7: Commit**

```bash
git add automl/data/pipeline.py automl/cli/data.py \
  tests/unit/data/test_materialize_return_shape.py tests/unit/cli/test_phase6_cli_catalog.py
git commit -m "fix(data): allow metadata-only materialize return"
```

### Task D4: Move trial CLI policy and coordination into trial/runner domains

**Files:**
- Create: `automl/runner/results.py`
- Modify: `automl/runner/session_lock.py`
- Modify: `automl/trial/create.py`
- Modify: `automl/cli/trial.py`
- Test: `tests/unit/cli/test_trial_status_policy.py`
- Test: `tests/unit/cli/test_phase6_cli_catalog.py`
- Test: `tests/unit/runner/test_session_lock.py`
- Test: `tests/unit/trial/test_authoring.py`

- [ ] **Step 1: Write failing domain-policy tests**

Append to `tests/unit/cli/test_trial_status_policy.py`:

```python
def test_runner_result_exit_code_owns_finished_policy():
    from automl.runner.results import trial_result_exit_code

    assert trial_result_exit_code({"status": "success"}) == 1
    assert trial_result_exit_code({"status": TrialStatus.FINISHED.value}) == 0


def test_trial_cli_uses_runner_exit_policy(monkeypatch):
    active = object()
    calls = []
    monkeypatch.setattr(trial_cli, "session_from_args", lambda args: active)
    monkeypatch.setattr(trial_cli, "run_trial", lambda path, **kwargs: {"status": "whatever"})
    monkeypatch.setattr(trial_cli, "trial_result_exit_code", lambda result: calls.append(result) or 7)
    monkeypatch.setattr(trial_cli, "print_json", lambda value: None)

    assert trial_cli._run(argparse.Namespace(path="trial-one")) == 7
    assert calls == [{"status": "whatever"}]
```

Append to `tests/unit/runner/test_session_lock.py`:

```python
def test_session_lock_acquire_for_session_builds_route_payload(tmp_path):
    from automl.runner import session_lock

    active = type(
        "Active",
        (),
        {
            "project_name": "demo",
            "active_experiment_id": "exp",
            "namespace": "qa",
            "dry_run": True,
            "config": type("Config", (), {"repo_root": tmp_path})(),
        },
    )()

    payload = session_lock.acquire_for_session(active, session_id="session-1")

    assert payload["status"] == "acquired"
    assert payload["session_id"] == "session-1"
    assert payload["route"] == "qa/dry_run/demo/exp"
    assert payload["lock_id"]
```

Append to `tests/unit/trial/test_authoring.py`:

```python
def test_create_resolves_proposal_defaults(monkeypatch, tmp_path):
    from importlib import import_module

    create_module = import_module("automl.trial.create")
    calls = []
    monkeypatch.setattr(
        create_module,
        "_create_resolved",
        lambda **kwargs: calls.append(kwargs) or (tmp_path / "trial-one"),
    )
    proposal = {"slug": "trial_one", "strategy": "baseline", "hypothesis": "Use proposal.", "seed_hint": "best"}

    result = create_module.create(proposal=proposal, session=object())

    assert result == tmp_path / "trial-one"
    assert calls[0]["slug"] == "trial_one"
    assert calls[0]["strategy"] == "baseline"
    assert calls[0]["hypothesis"] == "Use proposal."
    assert calls[0]["seed"] == "best"


def test_trial_domain_facade_does_not_export_create_request_helper():
    import automl.trial as trial

    assert "create_from_request" not in trial.__all__
    assert not hasattr(trial, "create_from_request")
```

- [ ] **Step 2: Verify red**

```bash
uv run pytest \
  tests/unit/cli/test_trial_status_policy.py::test_runner_result_exit_code_owns_finished_policy \
  tests/unit/cli/test_trial_status_policy.py::test_trial_cli_uses_runner_exit_policy \
  tests/unit/runner/test_session_lock.py::test_session_lock_acquire_for_session_builds_route_payload \
  tests/unit/trial/test_authoring.py::test_create_resolves_proposal_defaults \
  -q
```

Expected: fail because the domain helpers do not exist.

- [ ] **Step 3: Add runner result policy**

Create `automl/runner/results.py`:

```python
"""Runner result policy helpers."""

from __future__ import annotations

from automl.trial.types import TrialStatus


def trial_status_value(status: object) -> str:
    return str(getattr(status, "value", status))


def trial_result_exit_code(result: object) -> int:
    status = result.get("status") if isinstance(result, dict) else getattr(result, "status", "")
    return 0 if trial_status_value(status) == TrialStatus.FINISHED.value else 1


__all__ = ["trial_result_exit_code", "trial_status_value"]
```

Do not export these helpers from `automl/runner/__init__.py`; they are internal
CLI/domain policy helpers, not new top-level runner public API.

- [ ] **Step 4: Add runner session-lock coordination helpers**

In `automl/runner/session_lock.py`, add:

```python
def acquire_for_session(active: Any, *, session_id: str) -> dict[str, str]:
    route = route_for_session(active)
    lock_id = acquire(
        project_root=active.config.repo_root,
        route=route,
        session_id=session_id,
    )
    return {
        "status": "acquired",
        "session_id": session_id,
        "route": route,
        "lock_id": lock_id,
    }


def release_for_session(active: Any, *, session_id: str, lock_id: str) -> dict[str, str]:
    release(project_root=active.config.repo_root, session_id=session_id, lock_id=lock_id)
    return {"status": "released", "session_id": session_id, "lock_id": lock_id}
```

Add both names to `__all__`.

- [ ] **Step 5: Move proposal-default resolution into the existing trial create verb**

In `automl/trial/create.py`, keep the existing public verb name and make its
input shape flexible enough for proposal-backed creation. Change the signature
to accept omitted `slug` and `strategy` when `proposal` carries them:

```python
def create(
    slug: str | None = None,
    strategy: str | None = None,
    *,
    hypothesis: str = "",
    seed: str | None = None,
    model_source: Path | None = None,
    training_origin: str = "automl",
    proposal: dict[str, Any] | None = None,
    session: Session | None = None,
) -> Path:
    resolved_slug = slug or _proposal_value(proposal, "slug")
    resolved_strategy = strategy or _proposal_value(proposal, "strategy")
    resolved_hypothesis = hypothesis or _proposal_value(proposal, "hypothesis", default="")
    resolved_seed = seed or _proposal_value(proposal, "seed_hint", default=None)
    if not resolved_slug:
        raise ValueError("trial create requires slug or proposal_json.slug")
    if not resolved_strategy:
        raise ValueError("trial create requires --strategy or proposal_json.strategy")
    return _create_resolved(
        slug=resolved_slug,
        strategy=resolved_strategy,
        hypothesis=resolved_hypothesis or "",
        seed=resolved_seed,
        model_source=model_source,
        training_origin=training_origin,
        proposal=proposal,
        session=session,
    )


def _proposal_value(
    proposal: dict[str, object] | None,
    key: str,
    *,
    default: str | None = "",
) -> str | None:
    if proposal is None:
        return default
    value = proposal.get(key)
    if value is None:
        return default
    text = str(value).strip()
    return text or default
```

Move the previous body of `create()` into a private `_create_resolved(...)`
helper with non-optional `slug` and `strategy`. Keep `automl/trial/__init__.py`
unchanged; `automl.trial.create` already lazily exports the existing `create`
verb and no new facade export is needed.

- [ ] **Step 6: Thin the CLI adapters**

In `automl/cli/trial.py`, import:

```python
from automl.runner.results import trial_result_exit_code
```

Continue importing the existing trial `create` function as `create_trial`. Call
it from `_create` with `slug=args.slug`, `strategy=args.strategy`, and
`proposal=proposal`; the domain function resolves proposal defaults. Use
`trial_result_exit_code(result)` in `_run` and `_promote`; use
`trial_lock.acquire_for_session` and `trial_lock.release_for_session` in lock
handlers. Remove `_proposal_value` and `_trial_status_value` from the CLI file.

- [ ] **Step 7: Verify green**

```bash
uv run pytest tests/unit/cli/test_trial_status_policy.py tests/unit/cli/test_phase6_cli_catalog.py tests/unit/runner/test_session_lock.py tests/unit/trial/test_authoring.py tests/contracts/test_architecture.py -q
```

Expected: pass, including the approved runner-to-`automl.trial.types` boundary,
with no new `automl.runner` or `automl.trial` top-level facade exports.

- [ ] **Step 8: Commit**

```bash
git add automl/runner/results.py automl/runner/session_lock.py \
  automl/trial/create.py automl/cli/trial.py \
  tests/unit/cli/test_trial_status_policy.py tests/unit/cli/test_phase6_cli_catalog.py \
  tests/unit/runner/test_session_lock.py tests/unit/trial/test_authoring.py
git commit -m "refactor(cli): move trial policies into domains"
```

### Task D5: Split over-budget CLI verb files into parser facades

**Files:**
- Create: `automl/cli/_experiment_actions.py`
- Create: `automl/cli/_trial_actions.py`
- Create: `automl/cli/_validate_actions.py`
- Modify: `automl/cli/experiment.py`
- Modify: `automl/cli/trial.py`
- Modify: `automl/cli/validate.py`
- Test: `tests/contracts/test_architecture.py`
- Test: `tests/unit/cli/test_phase6_cli_catalog.py`

- [ ] **Step 1: Add the CLI file budget contract**

Append to `tests/contracts/test_architecture.py`:

```python
def test_cli_verb_files_stay_thin():
    budgets = {
        "automl/cli/project.py": 80,
        "automl/cli/experiment.py": 80,
        "automl/cli/trial.py": 80,
        "automl/cli/data.py": 80,
        "automl/cli/eval.py": 80,
        "automl/cli/validate.py": 80,
    }
    offenders = []
    for relative, budget in budgets.items():
        line_count = len((REPO_ROOT / relative).read_text().splitlines())
        if line_count > budget:
            offenders.append((relative, line_count, budget))

    assert offenders == []
```

- [ ] **Step 2: Verify red**

```bash
uv run pytest tests/contracts/test_architecture.py::test_cli_verb_files_stay_thin -q
```

Expected: fail on at least the current over-budget CLI verb files.

- [ ] **Step 3: Move experiment action handlers**

Create `automl/cli/_experiment_actions.py` with the action functions currently
in `automl/cli/experiment.py` (`_session`, `_list`, `_run`, `_delete`,
`_leaderboard`, `_compare`, `_summary`, `_proposer_context`) and their imports.
Keep `automl/cli/experiment.py` responsible only for parser construction and
binding `func`.

The parser file should import handlers like:

```python
from automl.cli._experiment_actions import (
    _compare,
    _delete,
    _leaderboard,
    _list,
    _proposer_context,
    _run,
    _summary,
)
```

- [ ] **Step 4: Move trial action handlers**

Create `automl/cli/_trial_actions.py` with `_list`, `_create`, `_fork`,
`_promote`, `_run`, `_show`, `_delete`, `_lock_acquire`, `_lock_release`,
`_load_proposal`, and their imports. Keep `automl/cli/trial.py` responsible only
for parser construction and binding `func`.

- [ ] **Step 5: Move validate action handlers**

Create `automl/cli/_validate_actions.py` with `_project`, `_model`,
`_proposal`, `_optional_session`, and `_read_json_arg`. Keep
`automl/cli/validate.py` responsible only for parser construction and binding
`func`.

- [ ] **Step 6: Update CLI tests to patch the new action homes**

In `tests/unit/cli/test_phase6_cli_catalog.py`, change monkeypatch targets for
moved action dependencies from the parser modules to the action modules. Keep
the patched symbol names and fake functions unchanged; replace only these module
strings:

```text
automl.cli.experiment -> automl.cli._experiment_actions
automl.cli.experiment.subprocess.run -> automl.cli._experiment_actions.subprocess.run
automl.cli.trial -> automl.cli._trial_actions
automl.cli.validate -> automl.cli._validate_actions
```

Keep data and eval monkeypatch targets unchanged because Wave D does not split
those parser files.

- [ ] **Step 7: Verify parser facade line budgets**

```bash
uv run pytest tests/contracts/test_architecture.py::test_cli_verb_files_stay_thin tests/unit/cli/test_phase6_cli_catalog.py -q
```

Expected: pass, with each `automl/cli/{project,experiment,trial,data,eval,validate}.py`
at or below 80 lines.

- [ ] **Step 8: Verify no public API drift**

```bash
uv run pytest tests/unit/cli/test_cli_public_api.py tests/unit/cli/test_phase6_cli_catalog.py -q
```

Expected: pass.

- [ ] **Step 9: Commit**

```bash
git add automl/cli/_experiment_actions.py automl/cli/_trial_actions.py automl/cli/_validate_actions.py \
  automl/cli/experiment.py automl/cli/trial.py automl/cli/validate.py \
  tests/contracts/test_architecture.py tests/unit/cli/test_phase6_cli_catalog.py
git commit -m "refactor(cli): keep verb parser files thin"
```

### Task D6: Move project validation checks into the project domain

**Files:**
- Create: `automl/project/checks.py`
- Modify: `automl/validate/targets.py`
- Test: `tests/unit/validate/test_project_validation.py`
- Test: `tests/contracts/test_architecture.py`

- [ ] **Step 1: Write failing `_safe()` coverage for project checks**

Append to `tests/unit/validate/test_project_validation.py`:

```python
def test_validate_project_wraps_crashed_domain_checks(monkeypatch, tmp_path):
    from automl import validate
    from automl.project import checks as project_checks

    def boom(**kwargs):
        raise RuntimeError("project check exploded")

    monkeypatch.setattr(project_checks, "config_required_fields", boom)
    report = validate.project(session=_session(tmp_path))

    assert report.passed is False
    assert report.issues[0].check == "project.config_required_fields.crashed"
    assert "project check exploded" in report.issues[0].message
```

- [ ] **Step 2: Verify red**

```bash
uv run pytest tests/unit/validate/test_project_validation.py::test_validate_project_wraps_crashed_domain_checks -q
```

Expected: fail because `automl.project.checks` does not exist and project
validation is inline in `validate.targets`.

- [ ] **Step 3: Add the validation-direction architecture guard**

Append to `tests/contracts/test_architecture.py`:

```python
def test_domains_do_not_import_validate_target_orchestrators():
    offenders = []
    for domain in AUTOML_DOMAINS:
        if domain == "validate":
            continue
        for file_path in sorted((AUTOML_ROOT / domain).rglob("*.py")):
            for imported in _imports_in(file_path):
                if imported in {"automl.validate", "automl.validate.targets"}:
                    offenders.append((_relative(file_path), imported))

    assert offenders == []
```

Run:

```bash
uv run pytest tests/contracts/test_architecture.py::test_domains_do_not_import_validate_target_orchestrators -q
```

Expected: pass. This is a direction guard, not a red test: domains may import
`automl.validate.base` value types, but they must not import the validate target
orchestrator.

- [ ] **Step 4: Create project-owned checks**

Create `automl/project/checks.py`:

```python
"""Project-domain validation checks."""

from __future__ import annotations

from collections.abc import Iterable

from automl.validate.base import Issue


def config_required_fields(*, config) -> Iterable[Issue]:
    required_fields = {
        "task": "TASK",
        "data_spec": "DATA",
        "eval_spec": "EVAL",
        "run_config": "RUN_CONFIG",
    }
    issues: list[Issue] = []
    for attr, public_name in required_fields.items():
        if getattr(config, attr) is None:
            issues.append(
                Issue(
                    level="error",
                    check=f"project.config.{attr}",
                    message=f"{public_name} is missing from project config",
                    location=str(config.config_path) if config.config_path else None,
                )
            )
    return issues


def environment_fields(*, config) -> Iterable[Issue]:
    env_fields = {
        "gcs_bucket": "GCS_BUCKET",
        "gcs_prefix": "GCS_PREFIX",
        "mlflow_tracking_uri": "MLFLOW_TRACKING_URI",
    }
    issues: list[Issue] = []
    for attr, env_name in env_fields.items():
        if not getattr(config, attr):
            issues.append(
                Issue(
                    level="error",
                    check=f"project.env.{attr}",
                    message=f"{env_name} is required",
                )
            )
    return issues


__all__ = ["config_required_fields", "environment_fields"]
```

- [ ] **Step 5: Route validate.project through domain checks**

In `automl/validate/targets.py`, change the module docstring to:

```python
"""Validate orchestrators.

Direction rule: this framework lazily imports per-domain checks and wraps them
with `_safe()`. Domain check modules may import `automl.validate.base` value
types, but must not import this orchestrator module.
"""
```

Replace inline project validation with:

```python
def project(*args, **kwargs) -> ValidationReport:
    del args
    from automl.project import checks as project_checks

    active = kwargs.get("session")
    if active is None:
        from automl.project import session as active_project_session

        active = active_project_session()
    config = active.config
    issues: list[Issue] = []
    issues.extend(
        _safe("project.config_required_fields", project_checks.config_required_fields, config=config)
    )
    issues.extend(_safe("project.environment_fields", project_checks.environment_fields, config=config))
    return ValidationReport(issues=issues)
```

- [ ] **Step 6: Verify green**

```bash
uv run pytest tests/unit/validate/test_project_validation.py tests/contracts/test_architecture.py::test_domains_do_not_import_validate_target_orchestrators -q
```

Expected: pass.

- [ ] **Step 7: Commit**

```bash
git add automl/project/checks.py automl/validate/targets.py tests/unit/validate/test_project_validation.py tests/contracts/test_architecture.py
git commit -m "refactor(validate): route project checks through domain"
```

### Task D7: Make model validation checks single-pass and domain-owned

**Files:**
- Modify: `automl/model/checks.py`
- Modify: `automl/validate/targets.py`
- Modify: `automl/cli/_validate_actions.py`
- Modify: `tests/unit/validate/test_model_validation.py`
- Modify: `tests/unit/validate/test_required_transformer_gate.py`

- [ ] **Step 1: Write failing double-fetch and predict-check tests**

Append to `tests/unit/validate/test_required_transformer_gate.py`:

```python
def test_required_transformer_gate_reads_requirements_once(monkeypatch):
    from automl.model import preprocessing

    calls = []
    original = preprocessing._requirements

    def counted(session):
        calls.append(session)
        return original(session)

    monkeypatch.setattr(preprocessing, "_requirements", counted)
    issues = _issues_for(
        CompliantModel,
        [
            RequiredTransformer(
                name="required_category",
                transformer=RequiredCategoryEncoder(),
                input_cols=["category"],
            )
        ],
    )

    assert _required_issue_checks(issues) == []
    assert len(calls) == 1
```

In `tests/unit/validate/test_model_validation.py`, update the predict-failure
assertion to expect a distinct domain check:

```python
assert any(issue.check == "model.predict_succeeds" for issue in predict_broken.issues)
assert any("predict exploded" in issue.message for issue in predict_broken.issues)
```

- [ ] **Step 2: Verify red**

```bash
uv run pytest \
  tests/unit/validate/test_required_transformer_gate.py::test_required_transformer_gate_reads_requirements_once \
  tests/unit/validate/test_model_validation.py::test_validate_model_reports_subclass_fit_predict_and_post_fit_failures \
  -q
```

Expected: fail because requirements are fetched twice and predict failures are
reported through `model.fit_succeeds`.

- [ ] **Step 3: Add domain-owned predict check**

In `automl/model/checks.py`, add:

```python
def predict_succeeds(*, instance: Any, df) -> Iterable[Issue]:
    try:
        instance.predict(context=None, model_input=df)
    except Exception as exc:  # noqa: BLE001
        return [
            Issue(
                level="error",
                check="model.predict_succeeds",
                message=f"model predict failed: {type(exc).__name__}: {exc}",
            )
        ]
    return []
```

Add `predict_succeeds` to `__all__`.

- [ ] **Step 4: Fetch required transformers once**

Change `check_required_transformers` to read typed requirements once and derive
the display payload from that same list:

```python
def check_required_transformers(*, instance: Any, session: Any | None = None) -> Iterable[Issue]:
    from sklearn.compose import ColumnTransformer

    from automl.model.preprocessing import _requirements

    declared = list(_requirements(session))
    if not declared:
        return []
    requirements = [
        {
            "name": requirement.name,
            "columns": list(requirement.input_cols),
        }
        for requirement in declared
    ]
    preprocessor = getattr(instance, "preprocessor", None)
    if not isinstance(preprocessor, ColumnTransformer):
        return [
            Issue(
                level="error",
                check="model.required_transformers",
                message=(
                    "required transformers require instance.preprocessor to be a "
                    "top-level sklearn.compose.ColumnTransformer"
                ),
            )
        ]
    fitted_entries = getattr(preprocessor, "transformers_", None)
    if not fitted_entries:
        return [
            Issue(
                level="error",
                check="model.required_transformers",
                message="required transformers require a fitted ColumnTransformer",
            )
        ]

    by_name = {str(name): (transformer, columns) for name, transformer, columns in fitted_entries}
    declared_by_name = {requirement.name: requirement for requirement in declared}
    issues: list[Issue] = []
    for requirement in requirements:
        name = requirement["name"]
        if name not in by_name:
            issues.append(
                Issue(
                    level="error",
                    check="model.required_transformers",
                    message=f"required transformer {name!r} is missing from preprocessor",
                )
            )
            continue
        fitted_transformer, fitted_columns = by_name[name]
        declared_requirement = declared_by_name[name]
        if not isinstance(fitted_transformer, type(declared_requirement.transformer)):
            issues.append(
                Issue(
                    level="error",
                    check="model.required_transformers",
                    message=(
                        f"required transformer {name!r} must be "
                        f"{type(declared_requirement.transformer).__name__}, "
                        f"got {type(fitted_transformer).__name__}"
                    ),
                )
            )
            continue
        fitted_column_set = _named_column_set(fitted_columns)
        required_columns = set(requirement["columns"])
        if not required_columns.issubset(fitted_column_set):
            issues.append(
                Issue(
                    level="error",
                    check="model.required_transformers",
                    message=(
                        f"required transformer {name!r} must include columns "
                        f"{sorted(required_columns)}, got {sorted(fitted_column_set)}"
                    ),
                )
            )
    return issues
```

- [ ] **Step 5: Remove predict from `_try_fit` and pass session into model checks**

In `automl/validate/targets.py`, import and call `predict_succeeds` after fit
passes:

```python
from automl.model.checks import (
    check_required_transformers,
    fit_succeeds,
    post_fit_attrs_set,
    predict_succeeds,
    subclass_basemodel,
)
```

Change the model signature and post-fit checks:

```python
def model(cls: type[Any], *, df, registry, session=None) -> ValidationReport:
    from automl.model.checks import (
        check_required_transformers,
        fit_succeeds,
        post_fit_attrs_set,
        predict_succeeds,
        subclass_basemodel,
    )

    issues: list[Issue] = []
    issues.extend(_safe("model.subclass_basemodel", subclass_basemodel, cls=cls))
    if any(issue.level == "error" for issue in issues):
        return ValidationReport(issues=issues)

    instance, error, error_stage = _try_fit(cls, df, registry, seed=0)
    issues.extend(
        _safe(
            "model.fit_succeeds",
            fit_succeeds,
            cls=cls,
            instance=instance,
            error=error,
            error_stage=error_stage,
        )
    )
    if error is None:
        issues.extend(_safe("model.predict_succeeds", predict_succeeds, instance=instance, df=df))
        if not any(issue.level == "error" for issue in issues):
            issues.extend(
                _safe("model.post_fit_attrs_set", post_fit_attrs_set, cls=cls, instance=instance)
            )
            issues.extend(
                _safe(
                    "model.required_transformers",
                    check_required_transformers,
                    instance=instance,
                    session=session,
                )
            )
    return ValidationReport(issues=issues)
```

Remove the `instance.predict(...)` block from `_try_fit`.

In `automl/cli/_validate_actions.py`, pass an optional active session to
`validate_model`:

```python
active = _optional_session(args)
report = validate_model(cls, df=df, registry=registry, session=active)
```

- [ ] **Step 6: Update tests that relied on ambient active session**

In `tests/unit/validate/test_required_transformer_gate.py`, change `_issues_for`
to call `validate_model(..., session=_session(requirements))` instead of using
the private `_ACTIVE_SESSION` context manager. Delete the `_active` helper if it
becomes unused.

- [ ] **Step 7: Verify green**

```bash
uv run pytest tests/unit/validate/test_model_validation.py tests/unit/validate/test_required_transformer_gate.py tests/unit/cli/test_phase6_cli_catalog.py -q
```

Expected: pass.

- [ ] **Step 8: Commit**

```bash
git add automl/model/checks.py automl/validate/targets.py automl/cli/_validate_actions.py \
  tests/unit/validate/test_model_validation.py tests/unit/validate/test_required_transformer_gate.py \
  tests/unit/cli/test_phase6_cli_catalog.py
git commit -m "refactor(validate): make model checks domain-owned"
```

### Task D8: Wave D gate, review, status update

**Files:**
- Modify: `docs/execution/STATUS.md`

- [ ] **Step 1: Run the Wave D acceptance gate**

```bash
uv run pytest tests/unit tests/integration tests/contracts -q
uv run pytest tests/unit/cli tests/unit/validate tests/contracts/test_architecture.py -q
uv run automl --help >/tmp/automl-help.txt
uv run python - <<'PY'
from automl.cli import build_parser

def subparser(parser, *path):
    current = parser
    for name in path:
        choices = {}
        for action in current._actions:
            choices.update(getattr(action, "choices", {}) or {})
        current = choices[name]
    return current

def options(parser):
    return {item for action in parser._actions for item in getattr(action, "option_strings", ())}

assert "--json" in options(subparser(build_parser(), "experiment", "run"))
for command in [
    ("project", "list"),
    ("experiment", "list"),
    ("trial", "run"),
    ("data", "materialize"),
    ("eval", "compute"),
    ("validate", "proposal"),
]:
    assert "--json" not in options(subparser(build_parser(), *command)), command
print("json flag surface ok")
PY
wc -l automl/cli/project.py automl/cli/experiment.py automl/cli/trial.py automl/cli/data.py automl/cli/eval.py automl/cli/validate.py
```

Expected:
- all pytest commands pass;
- parser smoke prints `json flag surface ok`;
- CLI verb files are at or below 80 lines;
- `data materialize` prints a dataset manifest only;
- `validate proposal --proposal-json` works and the old `--json` input flag is gone.

- [ ] **Step 2: Run final code review**

Dispatch a review subagent with:

```text
Review Wave D from the Wave C complete commit through HEAD.
Focus on CLI public-surface drift, skill-command compatibility, trial/runner
ownership, validation direction, and whether any removed --json flag was still
meaningfully consumed.
```

Fix any Critical/Important findings with targeted commits and rerun the relevant
gate.

- [ ] **Step 3: Mark Wave D complete**

Update `docs/execution/STATUS.md`:

```markdown
**Last updated:** 2026-05-29 (Wave D complete; Wave E plan next)
**Current wave:** E — **BLOCKED** pending detailed plan + user approval
**Overall:** 4 / 5 waves complete
```

Add a handoff entry with the gate evidence and mark Wave D checklist items
complete.

- [ ] **Step 4: Commit status only**

```bash
git add docs/execution/STATUS.md
git commit -m "docs(execution): mark Wave D complete"
```

- [ ] **Step 5: Stop for the Wave E plan gate**

After committing Wave D complete, author the detailed Wave E plan with
`superpowers:writing-plans`, post the plan path + task/risk summary, and wait for
explicit user approval before executing Wave E.

## Wave E — Docs/notebook truth + test-tier durability — DETAILED

> Steps use checkbox (`- [ ]`) syntax for tracking. Wave E is not approved until
> the user explicitly says "go." Intent guardrail: this wave aligns docs,
> notebooks, and tests to the **already-shipped** facade. Do not add new public
> API, new CLI nouns, new notebook-only shims, or artifact shape changes while
> executing this plan. E2E env flag names may change only to remove temporary
> `PHASE` terminology as part of the test-tier cleanup. If a notebook workflow
> appears to need a missing library feature, stop and raise that as a design
> concern.

**Goal:** Make user-facing docs/notebooks truthful against the final Python/CLI
surface, then make test tiers durable enough that future drift is caught by
contracts.

**Architecture:** The library remains source of truth. Notebook and skill prose
must call the domain facades that already exist (`automl.use_project`,
`automl.data`, `automl.experiment`, `automl.trial`, `automl.eval`,
`automl.utils.io.gcs`). Test-tier work is structural: markers, file names, and
contract tests; it must not change runtime semantics. Phase-derived e2e file,
function, and env-gate names are temporary and should be replaced with
domain/behavior names in this wave.

**Tech Stack:** Python requirement read directly from `pyproject.toml`, stdlib
`json` for notebook editing/static tests, `pytest` strict markers, existing
`uv` commands.

### Task E1: Remove stale `load_training_data` docs and document named split loading

**Files:**
- Create: `tests/contracts/test_data_docs_truth.py`
- Modify: `references/setup/data-pipeline.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Write the failing data-doc truth test**

Create `tests/contracts/test_data_docs_truth.py`:

```python
from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
ACTIVE_DOCS = [
    "CLAUDE.md",
    "references/setup/data-pipeline.md",
]


def _read(relative: str) -> str:
    return (REPO_ROOT / relative).read_text(encoding="utf-8")


def test_active_data_docs_do_not_advertise_load_training_data_hook():
    offenders = [
        relative
        for relative in ACTIVE_DOCS
        if "load_training_data" in _read(relative)
    ]

    assert offenders == []


def test_data_pipeline_docs_show_named_split_loading_surface():
    text = _read("references/setup/data-pipeline.md")

    assert "data.load_dataset(split_name=" in text
    assert "data.load_dataset_by_id(" in text
    assert "data.load_dataset_by_trial(" in text
    assert "run_config.train_split" in text
    assert "run_config.eval_split" in text
```

- [ ] **Step 2: Verify red**

```bash
uv run pytest tests/contracts/test_data_docs_truth.py -q
```

Expected: fail because active docs still mention `load_training_data` and the
data-pipeline reference does not clearly show the named split read surface.

- [ ] **Step 3: Reconcile docs with actual split/slice behavior**

In `references/setup/data-pipeline.md`:
- replace `load_training_data` references with the current split/slice read
  surface;
- remove the outdated waterfall list naming hooks that do not exist
  (`normalize_source_values`, `validate_loaded_data`, `infer_dtypes`,
  `apply_dtypes`, `dedupe`, `flag_features`, `split`);
- document the real sequence: load raw rows, standardize columns, normalize
  target/hash/metadata declarations, apply quality filters, add `SPLITID`, build
  `FeatureRegistry`, compute the immutable `Dataset`.
- add an explicit loading section with this prose and examples:

After materialization, training and evaluation slices are loaded through the
registry facade. The runner uses the project run config:

```python
run_config = active.config.require_run_config()
train = data.load_dataset(split_name=run_config.train_split, session=active)
eval_rows = data.load_dataset(split_name=run_config.eval_split, session=active)
```

For a specific materialized dataset or a replayed trial dataset, use:

```python
train = data.load_dataset_by_id(dataset_id, split_name="train", session=active)
trial_train = data.load_dataset_by_trial(run_id, split_name="train", session=active)
```

In `CLAUDE.md`, keep the design constraint wording but make it concrete:
raw source loading belongs to `DATA.source` and optional `DataPipeline`
subclasses wired through `DataSpec.pipeline_cls`; named train/eval slices belong
to the registry facade (`data.load_dataset(..., split_name=...)`) and the
project `RunConfig`.

- [ ] **Step 4: Verify and commit**

```bash
uv run pytest tests/contracts/test_data_docs_truth.py tests/integration/data_pipeline/test_materialize_load.py tests/integration/data_pipeline/test_trial_replay.py -q
git add tests/contracts/test_data_docs_truth.py references/setup/data-pipeline.md CLAUDE.md
git commit -m "docs(data): document named split loading"
```

### Task E2: Align README, skill prose, and project docs with final facade names

**Files:**
- Create: `tests/contracts/test_docs_truth.py`
- Modify: `README.md`
- Modify: `CLAUDE.md`
- Modify: `skills/automl-guide/SKILL.md`
- Modify: `skills/automl/SKILL.md`
- Modify: `skills/propose/SKILL.md`
- Modify: `skills/coder/SKILL.md`
- Modify: `agents/automl-proposer.md`
- Modify: `agents/automl-coder.md`
- Modify: `references/loop/protocol.md`
- Modify: `references/setup/model-contract.md`
- Modify: `references/setup/evaluation-metric.md`
- Modify: `projects/example_homecredit/config.py`
- Modify: `projects/example_homecredit/model/__init__.py`
- Modify: `projects/example_homecredit/PROJECT_INSTRUCTIONS.md`

- [ ] **Step 1: Add docs truth contracts**

Create `tests/contracts/test_docs_truth.py`:

```python
from __future__ import annotations

import re
import tomllib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _read(relative: str) -> str:
    return (REPO_ROOT / relative).read_text(encoding="utf-8")


def test_readme_python_requirement_matches_pyproject():
    pyproject = tomllib.loads(_read("pyproject.toml"))
    required = pyproject["project"]["requires-python"]
    readme = _read("README.md")

    assert f"Python {required}" in readme
    assert "Python >=3.13" not in readme
    assert "Python ≥3.13" not in readme


def test_active_guidance_uses_proposal_noun_not_trialproposal():
    roots = [
        REPO_ROOT / "README.md",
        REPO_ROOT / "skills",
        REPO_ROOT / "agents",
        REPO_ROOT / "references",
    ]
    offenders = []
    for root in roots:
        paths = [root] if root.is_file() else sorted(root.rglob("*.md"))
        for path in paths:
            text = path.read_text(encoding="utf-8")
            if re.search(r"\bTrialProposal\b", text):
                offenders.append(path.relative_to(REPO_ROOT).as_posix())

    assert offenders == []


def test_guide_paths_point_at_real_repo_files():
    guide = _read("skills/automl-guide/SKILL.md")

    assert "automl_dev/CLAUDE.md" not in guide
    assert "`CLAUDE.md`" in guide
    assert "`automl/data/sources/`" in guide
```

- [ ] **Step 2: Verify red**

```bash
uv run pytest tests/contracts/test_docs_truth.py -q
```

Expected: fail on README Python version, `TrialProposal` prose, and
`automl-guide` stale paths.

- [ ] **Step 3: Update README and contributor guide**

In `README.md`:
- replace `Python ≥3.13 is required` with `Python >=3.11 is required`, because
  the current `pyproject.toml` declares `requires-python = ">=3.11"`;
- add a short "Python facade" paragraph after "CLI verbs":

```markdown
For notebook and library use, start with `automl.use_project(...)`, then call
domain modules directly: `automl.data.materialize`, `automl.experiment.leaderboard`
and `compare`, `automl.trial.show_trial` and `load_model`, and
`automl.eval.evaluate`. Session-level route flags (`dry_run`, `namespace`,
`experiment_id`) belong on `use_project`, not repeated on each domain call.
```

- replace `TrialProposal` prose with `Proposal`.

In `CLAUDE.md`:
- update the repo layout comment from `projects/... (project.py, ...)` to
  `projects/... (config.py, ...)`;
- update "Python >=3.13/same pinned packages" wording if present to defer to
  `pyproject.toml`.

- [ ] **Step 4: Update skills and references**

Sweep only prose nouns. Replace `TrialProposal` with `Proposal` in:
`skills/automl/SKILL.md`, `skills/propose/SKILL.md`,
`skills/coder/SKILL.md`, `agents/automl-proposer.md`,
`agents/automl-coder.md`, and `references/loop/protocol.md`. Do **not** rename
artifact file paths like `proposal/trial_proposal.json` or schema fields.

In `skills/automl-guide/SKILL.md`:
- change `automl/data/ (pipeline.py + sources.py)` to
  `automl/data/ (pipeline.py + sources/)`;
- change `automl_dev/CLAUDE.md` references to `CLAUDE.md`;
- change `TrialProposal JSON` to `Proposal JSON`.

In `references/setup/model-contract.md`, add a point-of-use note:

```markdown
Each project package exposes its default model through `MODEL_CLASS` in
`projects/<project_name>/model/__init__.py`. The runner imports
`projects.<project_name>.model` and requires `MODEL_CLASS` to be a class.
```

In `references/setup/evaluation-metric.md`, keep custom metrics under
`projects/<project_name>/eval/metrics.py` only when the project actually needs
custom metric classes; otherwise `config.py` owns `EVAL`.

- [ ] **Step 5: Document `Session.config` and `MODEL_CLASS` where users see them**

In `projects/example_homecredit/config.py`, add a short comment above
`PROJECT_CONFIG`:

```python
# ProjectConfig is loaded into Session.config by automl.use_project(...).
# Domain calls should receive the Session, not re-read config globals.
```

In `projects/example_homecredit/model/__init__.py`, add a comment above
`MODEL_CLASS`:

```python
# The runner imports projects.example_homecredit.model.MODEL_CLASS for
# project-baseline trial execution.
```

In `projects/example_homecredit/PROJECT_INSTRUCTIONS.md`, update notebook
wording from "snapshot" to "dataset" where it refers to the current data noun,
and note that `config.py` owns the default `EVAL`; custom eval helpers should
live under a project-local `eval/` package only when needed.

- [ ] **Step 6: Verify and commit**

```bash
uv run pytest tests/contracts/test_docs_truth.py tests/contracts/test_phase6_skill_commands.py -q
git add tests/contracts/test_docs_truth.py README.md CLAUDE.md \
  skills/automl-guide/SKILL.md skills/automl/SKILL.md skills/propose/SKILL.md skills/coder/SKILL.md \
  agents/automl-proposer.md agents/automl-coder.md references/loop/protocol.md \
  references/setup/model-contract.md references/setup/evaluation-metric.md \
  projects/example_homecredit/config.py projects/example_homecredit/model/__init__.py \
  projects/example_homecredit/PROJECT_INSTRUCTIONS.md
git commit -m "docs: align user guidance with final facade"
```

### Task E3: Add notebook facade contracts before editing notebooks

**Files:**
- Create: `tests/contracts/test_notebook_surface.py`

- [ ] **Step 1: Add notebook smoke and retired-surface tests**

Create `tests/contracts/test_notebook_surface.py`:

```python
from __future__ import annotations

import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_DIR = REPO_ROOT / "projects" / "example_homecredit" / "notebooks"
NOTEBOOKS = sorted(NOTEBOOK_DIR.glob("*.ipynb"))

RETIRED_NOTEBOOK_PATTERNS = {
    "automl.load_project": re.compile(r"\bautoml\.load_project\b"),
    "top-level inspect facade": re.compile(r"from automl import .*inspect|automl\.inspect\b"),
    "top-level profile facade": re.compile(r"from automl import .*profile|automl\.profile\b"),
    "build_pipeline": re.compile(r"\bbuild_pipeline\b|\bdata\.build_pipeline\b"),
    "eval publish module": re.compile(r"\bautoml\.eval\.publish\b|from automl\.eval\.publish"),
    "old io facade": re.compile(r"\bautoml\.io\.gcs\b|from automl\.io\.gcs"),
    "domain per-call dry_run": re.compile(
        r"\b(?:data|experiment|trial|eval)\.[A-Za-z_]+\([^)]*\bdry_run\s*=",
        re.DOTALL,
    ),
}


def _notebook(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _source(cell: dict) -> str:
    return "".join(cell.get("source", []))


def _code_cells(path: Path) -> list[str]:
    return [
        _source(cell)
        for cell in _notebook(path).get("cells", [])
        if cell.get("cell_type") == "code"
    ]


def test_homecredit_notebook_first_code_cells_import_clean():
    assert NOTEBOOKS
    for path in NOTEBOOKS:
        first = _code_cells(path)[0]
        namespace = {"__name__": "__notebook_smoke__"}
        exec(compile(first, str(path), "exec"), namespace)


def test_homecredit_notebooks_use_final_facade_names():
    offenders = []
    for path in NOTEBOOKS:
        text = path.read_text(encoding="utf-8")
        for label, pattern in RETIRED_NOTEBOOK_PATTERNS.items():
            if pattern.search(text):
                offenders.append((path.relative_to(REPO_ROOT).as_posix(), label))

    assert offenders == []
```

- [ ] **Step 2: Verify red**

```bash
uv run pytest tests/contracts/test_notebook_surface.py -q
```

Expected: fail on current notebooks because they reference `automl.load_project`,
`inspect`, `build_pipeline`, `dry_run=`, or stale eval/io paths.

- [ ] **Step 3: Commit red contracts only**

```bash
git add tests/contracts/test_notebook_surface.py
git commit -m "test(notebooks): pin final facade surface"
```

### Task E4: Realign Home Credit notebooks to the shipped facade

**Files:**
- Modify: `projects/example_homecredit/notebooks/1_define_data_and_snapshot.ipynb`
- Modify: `projects/example_homecredit/notebooks/2_profile_data_snapshot.ipynb`
- Modify: `projects/example_homecredit/notebooks/2_run_agent_automl.ipynb`
- Modify: `projects/example_homecredit/notebooks/3_author_new_trial.ipynb`
- Modify: `projects/example_homecredit/notebooks/4_fork_existing_trial.ipynb`
- Modify: `projects/example_homecredit/notebooks/5_reevaluate_existing_model.ipynb`
- Modify: `projects/example_homecredit/notebooks/6_inspect_logged_runs_and_artifacts.ipynb`

- [ ] **Step 1: Normalize the first code cell in every notebook**

For each notebook, make the first code cell import-only or import+constants only
so `tests/contracts/test_notebook_surface.py` can execute it without project
services. Use this shape as the default:

```python
from __future__ import annotations

import pandas as pd
from IPython.display import display

import automl
from automl import data, eval, experiment, trial
from automl.utils.io import gcs

PROJECT_NAME = "example_homecredit"
DRY_RUN = True
```

Only include imports actually used in that notebook. Do not call
`automl.use_project(...)` in the first code cell.

- [ ] **Step 2: Replace project/session setup**

Replace every `automl.load_project(verbose=True)` and bare
`automl.use_project(verbose=True)` with an explicit session cell:

```python
active = automl.use_project(PROJECT_NAME, dry_run=DRY_RUN)
config = active.config
display(
    {
        "project": active.project_name,
        "repo_root": str(config.repo_root),
        "project_dir": str(config.project_dir),
        "experiment": active.active_experiment_id,
        "dry_run": active.dry_run,
    }
)
```

Use `active.config.<field>` for config details (`target_column`,
`raw_target_column`, `primary_metric`, paths). Do not read `experiment_id`
directly off the session except through `active.active_experiment_id`.

- [ ] **Step 3: Replace data notebook calls**

Use final data-domain calls:

```python
loaded = data.build_dataset(session=active)          # local preview, not persisted
dataset = data.materialize(session=active)           # persisted LoadedDataset
index = data.list_datasets(session=active)
run_config = active.config.require_run_config()
train = data.load_dataset(split_name=run_config.train_split, session=active)
holdout = data.load_dataset(split_name=run_config.eval_split, session=active)
profile_result = data.profile(session=active)
```

If a notebook only needs a manifest for display, call
`data.materialize(include_rows=False, session=active)`.

- [ ] **Step 4: Replace experiment and trial inspection calls**

Use final domain facades:

```python
leaderboard = experiment.leaderboard(training_origin="all", n=20, session=active)
rows = list(leaderboard.rows)
comparison = experiment.compare([row.run_id for row in rows[:2]], session=active)
details = trial.show_trial(run_id, session=active)
loaded_model = trial.load_model(run_id, session=active)
```

For human-authored trials, use:

```python
draft_dir = trial.create(
    "notebook_baseline",
    "human_baseline",
    hypothesis="Notebook-authored baseline.",
    training_origin="human",
    session=active,
)
```

For forks, use:

```python
fork_dir = trial.fork(
    "notebook_fork",
    seed="best",
    strategy="manual_fork",
    hypothesis="Manual notebook fork.",
    session=active,
)
```

For cleanup examples, call `trial.delete(run_id, apply=False, session=active)`.
Do not use old `trial.cleanup(trial_id=..., dry_run=...)`.

- [ ] **Step 5: Replace eval notebook calls**

Use the final eval-domain functions:

```python
eval_dataset, cached = eval.prepare_eval_dataset(
    session=active,
    dataset_id=loaded.dataset.id,
    split=active.config.require_run_config().eval_split,
)
result = eval.evaluate(
    session=active,
    model_run_id=run_id,
    eval_dataset_id=eval_dataset.id,
    label="notebook_eval",
    overwrite=False,
)
```

For external frames or augmentations, use `eval.prepare_eval_dataset(kind="external", ...)`
and `eval.prepare_eval_augmentation(session=active, ...)`. Do not import
`automl.eval.publish`.

- [ ] **Step 6: Replace GCS helper imports**

Replace `from automl.io.gcs import ...` with:

```python
from automl.utils.io import gcs
```

Use `gcs.parse_gcs_uri`, `gcs.blob_exists`, `gcs.list_blob_names`,
`gcs.read_parquet`, and `gcs.write_parquet` exactly as exported today.

- [ ] **Step 7: Verify and commit**

```bash
uv run pytest tests/contracts/test_notebook_surface.py tests/contracts/test_phase6_skill_commands.py -q
git add projects/example_homecredit/notebooks/*.ipynb
git commit -m "docs(notebooks): align homecredit notebooks with facade"
```

### Task E5: Add opt-in end-to-end notebook execution

**Files:**
- Create: `tests/e2e/test_homecredit_notebooks.py`
- Modify if execution finds drift: `projects/example_homecredit/notebooks/*.ipynb`

- [ ] **Step 1: Add the notebook e2e runner**

Create `tests/e2e/test_homecredit_notebooks.py`:

```python
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


pytestmark = [pytest.mark.e2e, pytest.mark.qa]

REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_DIR = REPO_ROOT / "projects" / "example_homecredit" / "notebooks"
NOTEBOOKS = [
    NOTEBOOK_DIR / "1_define_data_and_snapshot.ipynb",
    NOTEBOOK_DIR / "2_profile_data_snapshot.ipynb",
    NOTEBOOK_DIR / "2_run_agent_automl.ipynb",
    NOTEBOOK_DIR / "3_author_new_trial.ipynb",
    NOTEBOOK_DIR / "4_fork_existing_trial.ipynb",
    NOTEBOOK_DIR / "5_reevaluate_existing_model.ipynb",
    NOTEBOOK_DIR / "6_inspect_logged_runs_and_artifacts.ipynb",
]

REQUIRED_ENV = [
    "AUTOML_E2E_NOTEBOOKS",
    "GCS_BUCKET",
    "GCP_PROJECT",
    "MLFLOW_TRACKING_URI",
]


def _require_notebook_e2e_env() -> None:
    missing = [name for name in REQUIRED_ENV if not os.environ.get(name)]
    if missing:
        pytest.skip(
            "Home Credit notebook e2e requires "
            + ", ".join(REQUIRED_ENV)
        )


def _code_cells(path: Path) -> list[tuple[int, str]]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return [
        (index, "".join(cell.get("source", [])))
        for index, cell in enumerate(notebook.get("cells", []), start=1)
        if cell.get("cell_type") == "code"
    ]


def _execute_notebook(path: Path) -> None:
    namespace = {
        "__name__": f"__notebook_e2e_{path.stem}__",
        "__file__": str(path),
    }
    for index, source in _code_cells(path):
        stripped = source.lstrip()
        if stripped.startswith(("%", "!")):
            pytest.fail(f"{path.name} cell {index} uses notebook-only shell/magic syntax")
        exec(compile(source, f"{path}#cell-{index}", "exec"), namespace)


def test_homecredit_notebooks_execute_end_to_end(monkeypatch):
    _require_notebook_e2e_env()
    monkeypatch.chdir(REPO_ROOT)

    for path in NOTEBOOKS:
        _execute_notebook(path)
```

This uses stdlib `json` and Python `exec` instead of adding a new notebook
execution dependency. It is intentionally e2e/QA-gated because full notebook
execution can touch MLflow and GCS and may run the agent workflow.

- [ ] **Step 2: Verify skip behavior without service env**

```bash
uv run pytest tests/e2e/test_homecredit_notebooks.py -q
```

Expected without external env: one skipped test with a message naming
`AUTOML_E2E_NOTEBOOKS`, `GCS_BUCKET`, `GCP_PROJECT`, and
`MLFLOW_TRACKING_URI`.

- [ ] **Step 3: Run the notebooks end-to-end when e2e env is available**

```bash
AUTOML_E2E_NOTEBOOKS=1 uv run pytest tests/e2e/test_homecredit_notebooks.py -q
```

Expected with configured MLflow/GCS env: all seven notebooks execute without
retired facade calls, missing imports, or hidden notebook-only shell/magic
syntax.

- [ ] **Step 4: Fix notebook failures through the public facade only**

If the e2e runner fails, update the notebook cells to use existing domain
facades. Do not add notebook-only wrappers, new CLI nouns, or new library APIs.
If a notebook requires behavior that the library does not expose cleanly, stop
and raise it as a design concern before adding code.

- [ ] **Step 5: Verify and commit**

```bash
uv run pytest tests/contracts/test_notebook_surface.py tests/e2e/test_homecredit_notebooks.py -q
git add tests/e2e/test_homecredit_notebooks.py projects/example_homecredit/notebooks/*.ipynb
git commit -m "test(notebooks): add homecredit notebook e2e"
```

### Task E6: Make pytest tiers explicit and remove empty tiers

**Files:**
- Modify: `tests/contracts/test_pytest_structure.py`
- Modify: every `tests/unit/test_*.py` and `tests/unit/**/test_*.py`
- Modify: every `tests/integration/**/test_*.py`
- Modify: every `tests/contracts/test_*.py`
- Rename: `tests/e2e/test_phase1_walking_skeleton.py` →
  `tests/e2e/test_homecredit_walking_skeleton.py`
- Rename: `tests/e2e/test_phase2_data_model_breadth.py` →
  `tests/e2e/test_homecredit_data_model_breadth.py`
- Rename: `tests/e2e/test_phase3_eval_breadth.py` →
  `tests/e2e/test_eval_dataset_breadth.py`
- Rename: `tests/e2e/test_phase4_experiment_trial_cleanup.py` →
  `tests/e2e/test_experiment_trial_cleanup.py`
- Rename: `tests/e2e/test_phase5_agent_hooks.py` →
  `tests/e2e/test_agent_timeline_hooks.py`
- Rename: `tests/e2e/test_phase6_surface_isolation.py` →
  `tests/e2e/test_cli_route_isolation.py`
- Rename: `tests/e2e/test_phase7_cutover.py` →
  `tests/e2e/test_generated_trial_folder_loop.py`
- Delete: `tests/shared/.gitkeep`
- Delete: `tests/regression/.gitkeep`

- [ ] **Step 1: Add tier marker contracts**

Append to `tests/contracts/test_pytest_structure.py` and add `import ast` plus
`import re` at the top if absent:

```python
import ast
import re


TIER_MARKERS = {
    "unit": {"unit"},
    "integration": {"integration"},
    "contracts": {"contract"},
    "e2e": {"e2e", "qa"},
}
PHASE_TOKEN = re.compile(r"\b(?:phase[0-9]|AUTOML_PHASE[0-9]_E2E|Phase [0-9])\b", re.IGNORECASE)


def _test_files(tier: str) -> list[Path]:
    return sorted((REPO_ROOT / "tests" / tier).rglob("test_*.py"))


def _pytestmark_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "pytestmark" for target in node.targets):
            continue
        values = node.value.elts if isinstance(node.value, (ast.List, ast.Tuple)) else [node.value]
        for value in values:
            if (
                isinstance(value, ast.Attribute)
                and isinstance(value.value, ast.Attribute)
                and isinstance(value.value.value, ast.Name)
                and value.value.value.id == "pytest"
                and value.value.attr == "mark"
            ):
                names.add(value.attr)
    return names


def test_test_files_declare_their_tier_marker():
    offenders = []
    for tier, expected in TIER_MARKERS.items():
        for path in _test_files(tier):
            actual = _pytestmark_names(path)
            missing = expected - actual
            if missing:
                offenders.append((path.relative_to(REPO_ROOT).as_posix(), sorted(missing)))

    assert offenders == []


def test_e2e_tests_are_named_by_domain_not_phase():
    offenders = [
        path.relative_to(REPO_ROOT).as_posix()
        for path in _test_files("e2e")
        if "phase" in path.name.lower()
    ]

    assert offenders == []


def test_e2e_tests_do_not_keep_temporary_phase_tokens():
    offenders = [
        path.relative_to(REPO_ROOT).as_posix()
        for path in _test_files("e2e")
        if PHASE_TOKEN.search(path.read_text(encoding="utf-8"))
    ]

    assert offenders == []


def test_empty_uncollected_test_tiers_do_not_exist():
    offenders = [
        relative
        for relative in ("tests/shared", "tests/regression")
        if (REPO_ROOT / relative).exists()
    ]

    assert offenders == []
```

- [ ] **Step 2: Verify red**

```bash
uv run pytest tests/contracts/test_pytest_structure.py -q
```

Expected: fail because current files lack module-level tier markers, e2e files
and env gates use `phaseN` names, and empty `shared/`/`regression/`
directories exist.

- [ ] **Step 3: Apply module-level markers**

For each test file, add `import pytest` if absent and insert `pytestmark`
immediately after imports:

```python
pytestmark = pytest.mark.unit
```

Use `pytest.mark.integration` for `tests/integration`, `pytest.mark.contract`
for `tests/contracts`, and this list for `tests/e2e`:

```python
pytestmark = [pytest.mark.e2e, pytest.mark.qa]
```

Rename existing e2e environment variables away from temporary phase terminology
in this task:

```python
LIVE_E2E_ENV = "AUTOML_E2E"
NOTEBOOK_E2E_ENV = "AUTOML_E2E_NOTEBOOKS"
```

Update skip reasons, namespace strings, experiment ids, slugs, and provenance
strings inside these e2e files to domain/behavior names where they are just QA
fixtures. Do not rename real production artifact fields such as timing
`phases`.

- [ ] **Step 4: Rename e2e files and test functions**

Move the seven e2e files to the target names listed above. Rename test
functions inside them to behavior names matching the file, for example:

```python
def test_homecredit_walking_skeleton_external_gate():
    ...

def test_cli_surface_isolates_dry_run_and_namespace_universes(capsys):
    ...
```

Update skip conditions to the renamed env vars listed above.

- [ ] **Step 5: Remove empty tiers**

Delete the two empty uncollected tier placeholders:

```bash
git rm tests/shared/.gitkeep tests/regression/.gitkeep
rmdir tests/shared tests/regression
```

- [ ] **Step 6: Verify and commit**

```bash
uv run pytest tests/contracts/test_pytest_structure.py -q
uv run pytest -m "unit or contract" tests/unit tests/contracts -q
git add tests/contracts/test_pytest_structure.py tests/unit tests/integration tests/e2e
git commit -m "test: make pytest tiers explicit"
```

### Task E7: Parse skill-emitted CLI commands against the real parser

**Files:**
- Modify: `tests/contracts/test_phase6_skill_commands.py`
- Modify only if tests require it: `skills/automl/scripts/render_context.py`

- [ ] **Step 1: Add parser-backed command assertions**

In `tests/contracts/test_phase6_skill_commands.py`, import `shlex` and
`build_parser`:

```python
import shlex

from automl.cli import build_parser
```

Add helpers:

```python
def _automl_argv(command: str) -> list[str] | None:
    tokens = shlex.split(command)
    while tokens and "=" in tokens[0] and not tokens[0].startswith("-"):
        tokens = tokens[1:]
    if tokens[:2] == ["uv", "run"]:
        tokens = tokens[2:]
    if not tokens or tokens[0] != "automl":
        return None
    return tokens[1:]


def _assert_parses(command: str) -> None:
    argv = _automl_argv(command)
    if argv is None:
        return
    build_parser().parse_args(argv)
```

Update `test_automl_render_context_safe_commands_use_phase6_cli_surface()` to
parse every `safe_commands` value that starts with `uv run automl` or `automl`:

```python
for command in safe_commands.values():
    _assert_parses(command)
```

Keep the existing semantic assertions for `--proposal-json`, retired `--json`,
and retired verbs.

- [ ] **Step 2: Verify red or green intentionally**

```bash
uv run pytest tests/contracts/test_phase6_skill_commands.py -q
```

Expected: pass if skill commands already parse; fail only if a rendered command
is still syntactically stale.

- [ ] **Step 3: Fix render context only if parser validation fails**

If the new parser-backed test fails, update
`skills/automl/scripts/render_context.py` to render the current CLI surface. Do
not change core library APIs or add new CLI options in this task.

- [ ] **Step 4: Verify and commit**

```bash
uv run pytest tests/contracts/test_phase6_skill_commands.py -q
git add tests/contracts/test_phase6_skill_commands.py skills/automl/scripts/render_context.py
git commit -m "test(skills): parse rendered cli commands"
```

If `skills/automl/scripts/render_context.py` was unchanged, omit it from
`git add`.

### Task E8: Pin surface and layer contracts at the contract tier

**Files:**
- Modify: `tests/contracts/test_architecture.py`
- Modify: `tests/contracts/test_phase6_skill_commands.py`
- Modify: `tests/e2e/test_cli_route_isolation.py`

- [ ] **Step 1: Add a contract-level CLI surface smoke**

Move the JSON-flag surface smoke currently used manually at Wave D gates into
`tests/contracts/test_phase6_skill_commands.py`:

```python
def test_cli_json_flag_surface_is_contract_pinned():
    parser = build_parser()

    def subparser(*path: str):
        current = parser
        for name in path:
            choices = {}
            for action in current._actions:
                choices.update(getattr(action, "choices", {}) or {})
            current = choices[name]
        return current

    def options(parser):
        return {
            item
            for action in parser._actions
            for item in getattr(action, "option_strings", ())
        }

    assert "--json" in options(subparser("experiment", "run"))
    for command in [
        ("project", "list"),
        ("experiment", "list"),
        ("trial", "run"),
        ("data", "materialize"),
        ("eval", "compute"),
        ("validate", "proposal"),
    ]:
        assert "--json" not in options(subparser(*command)), command
```

- [ ] **Step 2: Add a layer-dependency contract index**

In `tests/contracts/test_architecture.py`, add a small ratchet that makes the
existing layer checks discoverable by name:

```python
def test_layer_dependency_contracts_are_present():
    required = {
        "test_leaf_utilities_do_not_import_automl_domains",
        "test_project_domain_does_not_import_downstream_runtime_domains",
        "test_trial_domain_does_not_import_runner_domain",
        "test_domains_do_not_import_validate_target_orchestrators",
        "test_runner_imports_only_approved_pure_trial_leaves",
        "test_domains_do_not_import_private_mlflow_routing",
    }
    current = {
        name
        for name, value in globals().items()
        if name.startswith("test_") and callable(value)
    }

    assert required.issubset(current)
```

- [ ] **Step 3: Keep live route isolation e2e named as e2e**

In the renamed `tests/e2e/test_cli_route_isolation.py`, keep the live GCS/MLflow
behavior exactly where it is. Do not move the live test into `tests/contracts`;
only the parser/surface contract belongs in contracts. This avoids making
contracts require external services.

- [ ] **Step 4: Verify and commit**

```bash
uv run pytest tests/contracts/test_architecture.py tests/contracts/test_phase6_skill_commands.py -q
git add tests/contracts/test_architecture.py tests/contracts/test_phase6_skill_commands.py tests/e2e/test_cli_route_isolation.py
git commit -m "test(contracts): pin cli surface and layer checks"
```

Omit `tests/e2e/test_cli_route_isolation.py` from `git add` if Task E6 already
renamed it and no content changes were needed here.

### Task E9: Wave E gate, review, and status update

**Files:**
- Modify: `docs/execution/STATUS.md`

- [ ] **Step 1: Run the Wave E acceptance gate**

```bash
uv run pytest tests/contracts -q
uv run pytest tests/unit tests/integration tests/contracts -q
uv run pytest -m "unit or contract" tests/unit tests/contracts -q
uv run pytest tests/e2e/test_homecredit_notebooks.py -q
uv run pytest tests/e2e -q
uv run python - <<'PY'
import tomllib
from pathlib import Path

root = Path.cwd()
pyproject = tomllib.loads((root / "pyproject.toml").read_text())
required = pyproject["project"]["requires-python"]
readme = (root / "README.md").read_text()
assert f"Python {required}" in readme
print("python version docs ok")
PY
```

Expected:
- `tests/contracts` green;
- unit/integration/contracts green;
- marker selection works without unknown-marker warnings;
- e2e tests collect under renamed domain/behavior names and skip cleanly without
  live-service env;
- when `AUTOML_E2E_NOTEBOOKS`, `GCS_BUCKET`, `GCP_PROJECT`, and
  `MLFLOW_TRACKING_URI` are configured, the notebook e2e test must be run and
  pass before Wave E is marked complete;
- README and `pyproject.toml` Python versions agree;
- notebook facade smoke/static contracts are part of `tests/contracts`;
- no active docs/prose still use `TrialProposal`;
- active docs describe train/eval loading through `data.load_dataset(...,
  split_name=...)` rather than a `DataPipeline.load_training_data` hook;
- no e2e test file, function, env gate, namespace, slug, or provenance fixture
  keeps temporary `phaseN` naming unless it is a real runtime artifact field.

- [ ] **Step 2: Run final Wave E review**

Dispatch a review subagent:

```text
Review Wave E from the Wave D complete commit through HEAD.
Focus on docs/notebook truth, accidental public surface additions, notebook
facade consistency, pytest marker durability, skill-command parser validation,
and whether any e2e/live-service behavior changed beyond renames/markers.
```

Fix any Critical/Important findings with targeted commits and rerun relevant
gate commands.

- [ ] **Step 3: Mark Wave E complete**

Update `docs/execution/STATUS.md`:

```markdown
**Last updated:** 2026-05-30 (Wave E complete; cleanup finished)
**Current wave:** COMPLETE
**Overall:** 5 / 5 waves complete
```

Add a handoff entry with gate evidence and mark Wave E checklist items
complete.

- [ ] **Step 4: Commit status**

```bash
git add docs/execution/STATUS.md
git commit -m "docs(execution): mark Wave E complete"
```

---

## Wave E Misalignment Risks To Review Before Approval

- **Notebook e2e depth:** Wave E now adds an opt-in full notebook e2e test
  instead of only first-code-cell smoke. This is intentionally `e2e`/`qa`
  gated because it can touch MLflow/GCS and may run the agent workflow. If the
  user expects this in default CI, that is a CI policy decision rather than a
  docs cleanup detail.
- **E2E environment variable rename:** Wave E now renames the e2e env gates from
  `AUTOML_PHASE*_E2E` to domain/behavior names. This will require CI/job-secret
  updates wherever those old env vars are configured.
- **`load_training_data` removal:** Wave E will not add
  `DataPipeline.load_training_data`. The code already models training/eval data
  as named materialized slices through `data.load_dataset(split_name=...)`, so
  adding a source-loading hook named "training data" would create a second,
  misleading concept.
- **Notebook cleanup examples:** The final cleanup API is run-id based
  (`trial.delete(run_id, ...)`), while old notebooks used slug/trial-id cleanup.
  The plan avoids inventing a notebook-only slug cleanup wrapper; notebooks
  should preview or delete by run id.
- **Project-local eval examples:** The plan documents where custom eval code
  belongs but does not add a new `projects/example_homecredit/eval/` package,
  because that would imply an example extension the project does not currently
  need.

---

## Self-review notes

- **Spec coverage:** every cluster (1–10) maps to a wave; the traceability section ties all 102 surviving findings to a cluster. Deferred items are listed explicitly in "Out of scope."
- **No placeholders in detailed waves:** Wave A, Wave B, and Wave C each have real tasks, code sketches, commands, and gates. Waves D–E remain scope+acceptance until their just-in-time detail pass.
- **Type/name consistency:** `TrialStatus` (Wave A) is the canonical `automl.trial.types.TrialStatus`; runner uses `str`. `bound_for` (Wave B) is the single public bind helper. Wave C C7.5 narrows the old runner/trial isolation rule: trial must not import runner; runner may import approved pure trial leaves only. These names and boundaries are referenced consistently in later waves' acceptance gates.
