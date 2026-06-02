# Project Entry Point Hard Cut-Over Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the strict `project.py` cut-over so active code, validation, docs, and tests no longer support or depend on `automl_config.yaml`, project-root `data.py`, or project-root `evaluation.py`.

**Architecture:** Make `ProjectContext` the single project discovery boundary and have it recognize only `projects/*/project.py`. Runtime commands resolve a `ProjectContext` and consume typed constants from `project.py`; validation verifies those constants explicitly. Test fixtures become typed projects instead of YAML-backed projects.

**Tech Stack:** Python 3.13, frozen dataclasses, pytest (`unit`, `contracts`, `integration`, `e2e` fixture checks), `uv`.

---

## File Map

**Runtime code**
- Modify `automl/core/project_context.py`: replace legacy config sentinel with `project.py`, expose `project_path`, remove YAML discovery.
- Modify `automl/data/prepare.py`: remove `ctx.config_path` gate.
- Modify `automl/trial/creation.py`: require `ctx.project_path`.
- Modify `automl/profile/snapshot.py`: remove YAML/project dual check.
- Modify `automl/cleanup.py`: discover cleanup projects by `project.py` only.
- Modify `automl/core/run_config.py`: validate `ModelRoute.effort`.
- Modify `automl/mlflow/store.py`: rename context metadata from `config_path` to `project_path`.
- Modify comments/docs in `automl/core/feature_registry.py`, `automl/data/pipeline.py`, `automl/eval/base.py`, `automl/cli/project.py`.

**Validation**
- Modify `automl/validate/builtin/contract_checks.py`: add `TASK` type check.
- Modify `automl/validate/builtin/config_checks.py`: keep `RUN_CONFIG` type check, update wording if needed.

**Tests and fixtures**
- Create `tests/shared/typed_project.py`: shared helper for writing minimal typed project fixtures.
- Modify `tests/unit/test_project_context.py`.
- Modify `tests/unit/core/test_run_config.py`.
- Modify `tests/unit/test_validate_contracts_check.py`.
- Modify `tests/unit/test_validate_project_aggregator.py`.
- Modify `tests/integration/test_prepare_data_snapshot_script.py`.
- Modify `tests/integration/test_claude_automl_launcher.py`.
- Modify `tests/integration/test_skill_render_context.py`.
- Modify `tests/e2e/fixtures/projects/test_homecredit/project.py` (new).
- Delete `tests/e2e/fixtures/projects/test_homecredit/automl_config.yaml`.
- Delete `tests/e2e/fixtures/projects/test_homecredit/data.py`.
- Delete `tests/e2e/fixtures/projects/test_homecredit/evaluation.py`.
- Modify `tests/e2e/fixtures/projects/test_homecredit/README.md`.
- Modify `tests/e2e/test_test_homecredit_project.py`.
- Modify `tests/shared/e2e_project_fixture.py`.
- Modify `tests/contracts/test_skill_plugin_contract.py`.
- Add or modify a contract ratchet for retired entry-point references.

---

## Pre-Flight

- [ ] **Step 1: Confirm current failure baseline.**

Run:

```bash
uv run python -m automl.data.prepare --project-root . --project example_homecredit --dry-run
```

Expected before the fix: FAIL with `projects/example_homecredit/automl_config.yaml not found`.

- [ ] **Step 2: Confirm integration baseline failure.**

Run:

```bash
uv run pytest tests/integration -q
```

Expected before the fix: FAIL in launcher/render-context tests that still create YAML-only projects.

---

## Task 1: Add a Typed Project Test Helper

**Files:**
- Create: `tests/shared/typed_project.py`

- [ ] **Step 1: Create the helper.**

Add:

```python
from __future__ import annotations

from pathlib import Path
from textwrap import dedent


def write_typed_project(
    root: Path,
    project_name: str = "payment_routing",
    *,
    experiment_id: str = "unit-exp",
    target: str = "TARGET",
    csv_path: str = "data.csv",
    hash_key: str = "ID",
    manager: tuple[str, str] = ("sonnet", "medium"),
    proposer: tuple[str, str] = ("sonnet", "medium"),
    coder: tuple[str, str] = ("sonnet", "medium"),
    data_source_expr: str | None = None,
    extra_imports: str = "",
    extra_body: str = "",
    include_task: bool = True,
    include_data: bool = True,
    include_eval: bool = True,
    include_run_config: bool = True,
) -> Path:
    """Write a minimal `projects/<name>/project.py` for tests."""
    projects_root = root / "projects"
    project_dir = projects_root / project_name
    project_dir.mkdir(parents=True, exist_ok=True)
    (projects_root / "__init__.py").write_text("", encoding="utf-8")
    (project_dir / "__init__.py").write_text("", encoding="utf-8")

    source_expr = data_source_expr or (
        f"LocalCSVSource(csv_path={csv_path!r}, hash_key={hash_key!r})"
    )
    sections: list[str] = [
        "from automl.core.run_config import RunConfig, Split, ModelsConfig, ModelRoute",
        "from automl.core.task import BinaryClassification",
        "from automl.data.spec import DataSpec",
        "from automl.data.sources import LocalCSVSource",
        "from automl.eval import EvalSpec",
        "from automl.eval.metrics import Auc",
    ]
    if extra_imports.strip():
        sections.append(extra_imports.strip())
    sections.append("")
    if extra_body.strip():
        sections.append(dedent(extra_body).strip())
        sections.append("")
    if include_task:
        sections.append(f"TASK = BinaryClassification(target={target!r})")
    if include_data:
        sections.append(f"DATA = DataSpec(source={source_expr})")
    if include_eval:
        sections.append("EVAL = EvalSpec(primary=Auc())")
    if include_run_config:
        sections.append(
            dedent(
                f"""
                RUN_CONFIG = RunConfig(
                    experiment_id={experiment_id!r},
                    split=Split(train=[(0, 80)], test=[(80, 100)]),
                    models=ModelsConfig(
                        manager=ModelRoute({manager[0]!r}, {manager[1]!r}),
                        proposer=ModelRoute({proposer[0]!r}, {proposer[1]!r}),
                        coder=ModelRoute({coder[0]!r}, {coder[1]!r}),
                    ),
                    per_trial_seconds=60,
                )
                """
            ).strip()
        )

    (project_dir / "project.py").write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    return project_dir
```

- [ ] **Step 2: Run a syntax check.**

Run:

```bash
uv run python -m py_compile tests/shared/typed_project.py
```

Expected: PASS.

- [ ] **Step 3: Commit.**

```bash
git add tests/shared/typed_project.py
git commit -m "test: add typed project fixture helper"
```

---

## Task 2: Make ProjectContext Project.py-Only

**Files:**
- Modify: `automl/core/project_context.py`
- Modify: `tests/unit/test_project_context.py`

- [ ] **Step 1: Rewrite focused ProjectContext tests to use typed projects.**

Use `tests.shared.typed_project.write_typed_project` in `tests/unit/test_project_context.py`.

Replace YAML setup in path/discovery tests with:

```python
from tests.shared.typed_project import write_typed_project
```

For `test_project_context_resolves_project_paths`, write the project with:

```python
project = write_typed_project(tmp_path, "payment_routing")
(project / "PROJECT_INSTRUCTIONS.md").write_text("Use this project.\n")

ctx = ProjectContext.from_name(tmp_path, "payment_routing")

assert ctx.repo_root == tmp_path.resolve()
assert ctx.project_name == "payment_routing"
assert ctx.project_dir == project.resolve()
assert ctx.project_package == "projects.payment_routing"
assert ctx.project_path == project.resolve() / "project.py"
assert ctx.instructions_path == project.resolve() / "PROJECT_INSTRUCTIONS.md"
assert not hasattr(ctx, "config_path")
```

Add a new test:

```python
def test_project_context_rejects_project_directory_without_project_py(tmp_path: Path) -> None:
    project = tmp_path / "projects" / "payment_routing"
    project.mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="projects/payment_routing/project.py"):
        ProjectContext.from_name(tmp_path, "payment_routing")
```

Add a new test:

```python
def test_find_repo_root_ignores_yaml_only_project(tmp_path: Path) -> None:
    project = tmp_path / "projects" / "legacy"
    project.mkdir(parents=True)
    (project / "automl_config.yaml").write_text("experiment_id: legacy\n")

    with pytest.raises(FileNotFoundError, match="projects/<project_name>/project.py"):
        find_repo_root(project)
```

Update metadata assertions to use `project_path`:

```python
assert metadata["project_path"] == str((project / "project.py").resolve())
assert "config_path" not in metadata
```

- [ ] **Step 2: Run focused tests and verify failure.**

Run:

```bash
uv run pytest tests/unit/test_project_context.py -q
```

Expected before implementation: FAIL because `ProjectContext` still exposes `config_path` and still discovers YAML.

- [ ] **Step 3: Implement project.py-only context.**

In `automl/core/project_context.py`:

Replace:

```python
LEGACY_CONFIG_FILENAME = "automl_config.yaml"
```

with:

```python
PROJECT_FILENAME = "project.py"
```

Change the dataclass field:

```python
project_path: Path
```

and remove:

```python
config_path: Path
```

In `from_name`, add an early file check:

```python
project_path = project_dir / PROJECT_FILENAME
if not project_path.is_file():
    raise FileNotFoundError(f"projects/{project_name}/{PROJECT_FILENAME} not found at {project_path}")
```

Return:

```python
return cls(
    repo_root=root,
    project_name=project_name,
    project_dir=project_dir,
    project_package=f"{PROJECTS_DIR}.{project_name}",
    project_path=project_path,
    instructions_path=project_dir / INSTRUCTIONS_FILENAME,
)
```

Change `find_repo_root` to:

```python
def find_repo_root(start: str | Path) -> Path:
    """Walk up from ``start`` until a typed project recipe is found."""
    current = Path(start).resolve()
    if current.is_file():
        current = current.parent
    while True:
        if any((current / PROJECTS_DIR).glob(f"*/{PROJECT_FILENAME}")):
            return current
        if current.parent == current:
            raise FileNotFoundError(
                f"{PROJECTS_DIR}/<project_name>/{PROJECT_FILENAME} not found above {start}"
            )
        current = current.parent
```

Change `_configured_project_names` to scan only `project.py`:

```python
for path in projects_root.glob(f"*/{PROJECT_FILENAME}"):
    if path.parent.is_dir():
        names.add(path.parent.name)
```

Update metadata payloads:

```python
"project_path": str(ctx.project_path),
```

and remove all `config_path` keys from `_context_metadata` and `_empty_metadata`.

- [ ] **Step 4: Update remaining code references to `ctx.config_path`.**

Run:

```bash
rg -n "ctx\\.config_path|config_path" automl tests/unit/test_project_context.py tests/unit/test_mlflow_store.py
```

Replace active metadata keys with `project_path`, including `automl/mlflow/store.py` and `tests/unit/test_mlflow_store.py`. Keep only historical text in tests that explicitly asserts retired surfaces are absent.

- [ ] **Step 5: Run focused tests.**

Run:

```bash
uv run pytest tests/unit/test_project_context.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit.**

```bash
git add automl/core/project_context.py tests/unit/test_project_context.py tests/unit/test_mlflow_store.py
git commit -m "refactor(core): resolve projects only from project.py"
```

---

## Task 3: Remove Runtime YAML Gates

**Files:**
- Modify: `automl/data/prepare.py`
- Modify: `automl/trial/creation.py`
- Modify: `automl/profile/snapshot.py`
- Modify: `automl/cleanup.py`
- Modify: `automl/cli/project.py`
- Modify: `tests/integration/test_prepare_data_snapshot_script.py`

- [ ] **Step 1: Migrate data-prep integration tests away from YAML.**

In `tests/integration/test_prepare_data_snapshot_script.py`, delete `_project_config` and `_write_project_config`.

Remove calls to `_write_project_config(tmp_path)`.

For the ambiguous-project test, create typed projects:

```python
for name in ("payment_routing", "risk_modeling"):
    _write_project_py(
        tmp_path,
        "class _MockPipeline(DataPipeline):\n"
        "    def load_data_snapshot(self):\n"
        "        raise AssertionError('not reached')\n",
        project_name=name,
    )
```

Change the missing project test to expect project resolution failure:

```python
def test_prepare_data_snapshot_reports_missing_project_py(tmp_path: Path) -> None:
    project = tmp_path / "projects" / "payment_routing"
    project.mkdir(parents=True)
    (tmp_path / "projects" / "__init__.py").write_text("")
    (project / "__init__.py").write_text("")

    result = subprocess.run(
        [
            sys.executable,
            *MODULE_ARGS,
            "--project-root",
            str(tmp_path),
            "--project",
            "payment_routing",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "projects/payment_routing/project.py" in result.stderr
```

- [ ] **Step 2: Run focused tests and verify failure.**

Run:

```bash
uv run pytest tests/integration/test_prepare_data_snapshot_script.py -q
```

Expected before implementation: FAIL because `automl.data.prepare` still checks `ctx.config_path`.

- [ ] **Step 3: Remove the data-prep YAML gate.**

In `automl/data/prepare.py`, delete:

```python
if not ctx.config_path.exists():
    print(f"{ctx.config_path.relative_to(ctx.repo_root)} not found", file=sys.stderr)
    return 2
```

`resolve_project_context` now owns missing-project errors.

- [ ] **Step 4: Simplify other runtime config checks.**

In `automl/trial/creation.py`, replace `_ensure_config_exists` with:

```python
def _ensure_config_exists(ctx: ProjectContext) -> None:
    if not ctx.project_path.is_file():
        raise ValueError(f"projects/{ctx.project_name}/project.py not found")
```

In `automl/profile/snapshot.py`, delete the `has_typed`/`ctx.config_path` block. `resolve_project_context` already ensures `project.py`.

In `automl/cleanup.py`, change `_project_dirs` to scan only `project.py`:

```python
if all_projects:
    return sorted(
        path.parent
        for path in projects_root.glob("*/project.py")
        if path.parent.is_dir()
    )
```

and change the explicit-project check to:

```python
if not (project_dir / "project.py").is_file():
    raise FileNotFoundError(f"missing project config: {project_dir / 'project.py'} does not exist")
```

In `automl/cli/project.py`, change the generated SQL comment from:

```sql
--   1. Emit the raw target column configured in data.py.
```

to:

```sql
--   1. Emit the raw target column configured in project.py TASK.
```

- [ ] **Step 5: Run focused runtime tests.**

Run:

```bash
uv run pytest tests/integration/test_prepare_data_snapshot_script.py tests/unit/test_cli_project.py tests/unit/test_cli_trial.py -q
```

Expected: PASS.

- [ ] **Step 6: Run the original data-prep smoke command.**

Run:

```bash
uv run python -m automl.data.prepare --project-root . --project example_homecredit --dry-run
```

Expected: it must not fail with `automl_config.yaml not found`. If it fails because local data/MLflow/GCS setup is missing, capture the new error in the commit notes and continue to the test suite.

- [ ] **Step 7: Commit.**

```bash
git add automl/data/prepare.py automl/trial/creation.py automl/profile/snapshot.py automl/cleanup.py automl/cli/project.py tests/integration/test_prepare_data_snapshot_script.py
git commit -m "refactor(runtime): remove YAML project gates"
```

---

## Task 4: Validate All Required Project.py Constants

**Files:**
- Modify: `automl/core/run_config.py`
- Modify: `automl/validate/builtin/contract_checks.py`
- Modify: `tests/unit/core/test_run_config.py`
- Modify: `tests/unit/test_validate_contracts_check.py`
- Modify: `tests/unit/test_validate_project_aggregator.py`

- [ ] **Step 1: Add failing tests for effort validation.**

Append to `TestModelRoute` in `tests/unit/core/test_run_config.py`:

```python
@pytest.mark.parametrize("effort", ["low", "medium", "high"])
def test_effort_allowed_values(self, effort):
    assert ModelRoute(model="sonnet", effort=effort).effort == effort

@pytest.mark.parametrize("bad", ["", "  ", "LOW", "extreme", None])
def test_effort_must_be_allowed_value(self, bad):
    with pytest.raises((ValueError, TypeError), match="effort"):
        ModelRoute(model="sonnet", effort=bad)  # type: ignore[arg-type]
```

Run:

```bash
uv run pytest tests/unit/core/test_run_config.py::TestModelRoute -q
```

Expected before implementation: FAIL for invalid non-empty efforts such as `extreme`.

- [ ] **Step 2: Implement effort validation.**

In `automl/core/run_config.py`, add:

```python
ALLOWED_EFFORTS = frozenset({"low", "medium", "high"})
```

Update `ModelRoute.__post_init__`:

```python
if not isinstance(self.effort, str) or not self.effort.strip():
    raise ValueError(f"effort must be a non-empty string, got {self.effort!r}")
if self.effort not in ALLOWED_EFFORTS:
    raise ValueError(
        f"effort must be one of {sorted(ALLOWED_EFFORTS)!r}, got {self.effort!r}"
    )
```

- [ ] **Step 3: Add a TASK contract check.**

In `tests/unit/test_validate_contracts_check.py`, add:

```python
def test_contracts_flag_missing_task_export(tmp_path) -> None:
    project_root = _scaffold_project(
        tmp_path,
        project_body="""
            from automl.data.sources import LocalCSVSource
            from automl.data.spec import DataSpec
            from automl.eval import EvalSpec
            from automl.eval.metrics import Auc
            DATA = DataSpec(source=LocalCSVSource(csv_path='data.csv', hash_key='ID'))
            EVAL = EvalSpec(primary=Auc())
        """,
    )
    from automl.validate.builtin.contract_checks import check_task_module_exports

    issues = list(check_task_module_exports(project="demo", project_root=project_root))
    assert [issue.check for issue in issues] == ["contracts.task_missing_export"]
```

Add a good-path assertion in `test_contracts_pass_for_well_formed_project`:

```python
from automl.validate.builtin.contract_checks import check_task_module_exports
assert list(check_task_module_exports(project="demo", project_root=project_root)) == []
```

Make the well-formed project body include:

```python
from automl.core.task import BinaryClassification
TASK = BinaryClassification(target='TARGET')
```

- [ ] **Step 4: Implement the TASK contract check.**

In `automl/validate/builtin/contract_checks.py`, add:

```python
@register(
    target="contracts",
    name="contracts.task_module",
    description="projects/<p>/project.py exports TASK as a typed task",
)
def check_task_module_exports(*, project: str, project_root: Path) -> Iterable[Issue]:
    try:
        module = _load_project_module(project, project_root, "project")
    except Exception as exc:  # noqa: BLE001
        return [
            Issue(
                level="error",
                check="contracts.task_import_failed",
                message=f"projects/{project}/project.py failed to import: {exc!r}",
            )
        ]
    if not hasattr(module, "TASK"):
        return [
            Issue(
                level="error",
                check="contracts.task_missing_export",
                message=f"projects/{project}/project.py does not export TASK",
                location="TASK",
            )
        ]

    from automl.core.task import BinaryClassification, Multiclass, Regression

    task = module.TASK
    if not isinstance(task, (BinaryClassification, Regression, Multiclass)):
        return [
            Issue(
                level="error",
                check="contracts.task_spec_type",
                message=f"projects/{project}/project.py TASK must be a typed task",
                location=f"projects/{project}/project.py:TASK",
            )
        ]
    return []
```

- [ ] **Step 5: Remove legacy setup from validation aggregator tests.**

In `tests/unit/test_validate_project_aggregator.py`, remove writes to `automl_config.yaml` and root `data.py` in `_write_project`.

Add to the generated project body:

```python
from automl.core.task import BinaryClassification
TASK = BinaryClassification(target='TARGET')
```

Update tests that create a project for `validate.config` to write `project.py`, not `automl_config.yaml`.

- [ ] **Step 6: Run focused validation tests.**

Run:

```bash
uv run pytest tests/unit/core/test_run_config.py tests/unit/test_validate_contracts_check.py tests/unit/test_validate_project_aggregator.py tests/unit/test_validate_config_check.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit.**

```bash
git add automl/core/run_config.py automl/validate/builtin/contract_checks.py tests/unit/core/test_run_config.py tests/unit/test_validate_contracts_check.py tests/unit/test_validate_project_aggregator.py tests/unit/test_validate_config_check.py
git commit -m "feat(validate): require typed TASK and validate route effort"
```

---

## Task 5: Migrate Launcher and Render-Context Integration Tests

**Files:**
- Modify: `tests/integration/test_claude_automl_launcher.py`
- Modify: `tests/integration/test_skill_render_context.py`

- [ ] **Step 1: Rewrite launcher fixture helper to write typed project.py.**

In `tests/integration/test_claude_automl_launcher.py`, remove `DEFAULT_MODEL_ROUTES` and `ConfigError` import.

Replace `_write_project(root: Path, config: str)` with:

```python
def _write_project(
    root: Path,
    *,
    experiment_id: str = "qa-e2e-smoke",
    manager: tuple[str, str] = ("sonnet", "medium"),
    proposer: tuple[str, str] = ("sonnet", "medium"),
    coder: tuple[str, str] = ("sonnet", "medium"),
) -> None:
    from tests.shared.typed_project import write_typed_project

    write_typed_project(
        root,
        "test_homecredit",
        experiment_id=experiment_id,
        manager=manager,
        proposer=proposer,
        coder=coder,
    )
    agents_dir = root / "agents"
    agents_dir.mkdir()
    for name in ("automl-proposer", "automl-coder"):
        (agents_dir / f"{name}.md").write_text(
            "---\n"
            f"name: {name}\n"
            f"description: {name} test agent\n"
            "tools: Read, Grep\n"
            "model: inherit\n"
            "effort: high\n"
            "---\n\n"
            "Return exactly one JSON object.\n",
            encoding="utf-8",
        )
```

Update call sites:

```python
_write_project(tmp_path, manager=("haiku", "low"), proposer=("haiku", "low"), coder=("haiku", "low"))
```

Replace `test_launcher_rejects_legacy_flat_model_keys` with:

```python
def test_launcher_rejects_invalid_effort(tmp_path: Path) -> None:
    module = _load_module()
    _write_project(tmp_path, manager=("sonnet", "extreme"))

    with pytest.raises(ValueError, match="effort"):
        module.build_launch(
            project_root=tmp_path,
            project="test_homecredit",
            automl_args=["run", "--project", "test_homecredit"],
            max_budget_usd="5",
            output_format="text",
        )
```

For ambiguous-project tests, call `write_typed_project(tmp_path, "one")` and `write_typed_project(tmp_path, "two")`.

- [ ] **Step 2: Run launcher tests.**

Run:

```bash
uv run pytest tests/integration/test_claude_automl_launcher.py -q
```

Expected: PASS.

- [ ] **Step 3: Rewrite render-context fixture helper to write typed project.py.**

In `tests/integration/test_skill_render_context.py`, replace `write_project_config` with:

```python
def write_project_py(
    project_root: Path,
    *,
    project_name: str = "generic",
    experiment_id: str = "2026-Q2-example",
    tracking_uri: str = "file:///tmp/mlruns",
    manager: tuple[str, str] = ("sonnet", "medium"),
    proposer: tuple[str, str] = ("sonnet", "medium"),
    coder: tuple[str, str] = ("sonnet", "medium"),
) -> None:
    from tests.shared.typed_project import write_typed_project

    write_typed_project(
        project_root,
        project_name,
        experiment_id=experiment_id,
        manager=manager,
        proposer=proposer,
        coder=coder,
    )
    (project_root / ".env").write_text(
        "GCS_BUCKET=unused\n"
        "GCS_PREFIX=automl\n"
        f"MLFLOW_TRACKING_URI={tracking_uri}\n",
        encoding="utf-8",
    )
    os.environ["GCS_BUCKET"] = "unused"
    os.environ["GCS_PREFIX"] = "automl"
    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
```

Update all call sites from YAML text to structured arguments. Example:

```python
write_project_py(
    project_root,
    project_name="generic",
    experiment_id="2026-Q2-example",
    tracking_uri="http://127.0.0.1:5000",
)
```

Replace YAML-specific tests:

- Delete `test_automl_render_context_has_minimal_config_fallback_without_pyyaml`.
- Delete `test_automl_render_context_tolerates_config_in_progress`.
- Delete `test_automl_render_context_converts_typed_yaml_to_json_safe_text`.
- Replace `test_automl_render_context_rejects_legacy_flat_model_keys` with an invalid-effort test:

```python
def test_automl_render_context_reports_invalid_project_py(tmp_path: Path) -> None:
    script = ROOT / "skills" / "automl" / "scripts" / "render_context.py"
    project_root = tmp_path / "project"
    project_root.mkdir()
    write_project_py(project_root, manager=("sonnet", "extreme"))

    payload = run_json(
        script,
        [
            "--project-root",
            str(project_root),
            "--arguments",
            "run --project generic --dry-run --max-iter 1",
        ],
    )

    assert payload["invocation"]["mode"] == "error"
    assert "effort" in payload["invocation"]["error"]
    assert "models" not in payload
```

For multi-project tests, write two typed projects:

```python
for name in ("payment_routing", "example_homecredit"):
    write_project_py(tmp_path, project_name=name, experiment_id="2026-Q2")
```

- [ ] **Step 4: Run render-context tests.**

Run:

```bash
uv run pytest tests/integration/test_skill_render_context.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add tests/integration/test_claude_automl_launcher.py tests/integration/test_skill_render_context.py
git commit -m "test(integration): migrate launcher context fixtures to project.py"
```

---

## Task 6: Migrate the E2E Smoke Fixture

**Files:**
- Create: `tests/e2e/fixtures/projects/test_homecredit/project.py`
- Delete: `tests/e2e/fixtures/projects/test_homecredit/automl_config.yaml`
- Delete: `tests/e2e/fixtures/projects/test_homecredit/data.py`
- Delete: `tests/e2e/fixtures/projects/test_homecredit/evaluation.py`
- Modify: `tests/e2e/fixtures/projects/test_homecredit/README.md`
- Modify: `tests/e2e/test_test_homecredit_project.py`
- Modify: `tests/shared/e2e_project_fixture.py`
- Modify QA tests if they expect YAML-shaped config dictionaries.

- [ ] **Step 1: Add typed fixture project.py.**

Create `tests/e2e/fixtures/projects/test_homecredit/project.py`:

```python
"""Typed Home Credit smoke project for committed AutoML E2E tests."""
from __future__ import annotations

from pathlib import Path

from automl.core import BinaryClassification, ModelRoute, ModelsConfig, RunConfig, Split
from automl.data import DataSpec
from automl.data.sources import LocalCSVSource
from automl.eval import EvalSpec
from automl.eval.metrics import Auc


TASK = BinaryClassification(target="TARGET")

DATA = DataSpec(
    source=LocalCSVSource(
        csv_path=Path(__file__).parent / "data" / "application_train_small.csv",
        hash_key="SK_ID_CURR",
    ),
    exclude_cols=[],
    metadata_cols=["SK_ID_CURR"],
    dry_run_rows=100,
)

EVAL = EvalSpec(primary=Auc())

RUN_CONFIG = RunConfig(
    experiment_id="qa-e2e-smoke",
    split=Split(train=[(0, 80)], test=[(80, 100)]),
    models=ModelsConfig(
        manager=ModelRoute("haiku", "low"),
        proposer=ModelRoute("haiku", "low"),
        coder=ModelRoute("haiku", "low"),
    ),
    per_trial_seconds=300,
)
```

- [ ] **Step 2: Delete retired fixture files.**

Run:

```bash
rm tests/e2e/fixtures/projects/test_homecredit/automl_config.yaml
rm tests/e2e/fixtures/projects/test_homecredit/data.py
rm tests/e2e/fixtures/projects/test_homecredit/evaluation.py
```

- [ ] **Step 3: Update fixture loader.**

In `tests/shared/e2e_project_fixture.py`, remove `yaml` import and `FIXTURE_CONFIG`.

Replace `load_fixture_config` with:

```python
def load_fixture_config() -> dict:
    from tests.e2e.fixtures.projects.test_homecredit import project

    return {
        "experiment_id": project.RUN_CONFIG.experiment_id,
        "models": {
            "manager": {
                "model": project.RUN_CONFIG.models.manager.model,
                "effort": project.RUN_CONFIG.models.manager.effort,
            },
            "proposer": {
                "model": project.RUN_CONFIG.models.proposer.model,
                "effort": project.RUN_CONFIG.models.proposer.effort,
            },
            "coder": {
                "model": project.RUN_CONFIG.models.coder.model,
                "effort": project.RUN_CONFIG.models.coder.effort,
            },
        },
        "per_trial_seconds": project.RUN_CONFIG.per_trial_seconds,
    }
```

- [ ] **Step 4: Update e2e fixture assertions.**

In `tests/e2e/test_test_homecredit_project.py`:

Remove `yaml` import.

Change tracked-file test:

```python
assert (PROJECT / "project.py").is_file()
assert not (PROJECT / "automl_config.yaml").exists()
assert not (PROJECT / "data.py").exists()
assert not (PROJECT / "evaluation.py").exists()
```

Replace config test with:

```python
def test_test_homecredit_project_uses_typed_routing_and_haiku() -> None:
    sys.modules.pop("projects", None)
    sys.modules.pop("projects.test_homecredit", None)
    sys.modules.pop("projects.test_homecredit.project", None)
    importlib.invalidate_caches()
    old_path = list(sys.path)
    try:
        fixture_root = str(FIXTURE_ROOT)
        sys.path[:] = [fixture_root, *[item for item in sys.path if item != fixture_root]]
        project = importlib.import_module("projects.test_homecredit.project")
    finally:
        sys.path[:] = old_path

    assert PROJECT.name == PROJECT_NAME
    assert project.RUN_CONFIG.experiment_id == "qa-e2e-smoke"
    assert project.RUN_CONFIG.models.manager.model == "haiku"
    assert project.RUN_CONFIG.models.manager.effort == "low"
    assert project.RUN_CONFIG.models.proposer.model == "haiku"
    assert project.RUN_CONFIG.models.coder.model == "haiku"
    assert project.RUN_CONFIG.per_trial_seconds == 300
    assert project.TASK.target == "TARGET"
```

Replace pipeline test with `build_pipeline(ctx)`:

```python
def test_test_homecredit_pipeline_uses_smaller_dry_run_than_full_run() -> None:
    from automl.core.project_context import ProjectContext
    from automl.data.loader import build_pipeline

    ctx = ProjectContext.from_name(FIXTURE_ROOT, "test_homecredit")
    dry_pipeline = build_pipeline(ctx, dry_run=True)
    full_pipeline = build_pipeline(ctx, dry_run=False)

    assert dry_pipeline.dry_run_row_limit() == 100
    assert full_pipeline.dry_run_row_limit() is None
    assert len(dry_pipeline.load_training_data()) == 100
    assert len(full_pipeline.load_training_data()) == 1000
```

- [ ] **Step 5: Update fixture README.**

In `tests/e2e/fixtures/projects/test_homecredit/README.md`, replace `automl_config.yaml` with `project.py RUN_CONFIG` and describe `TASK`, `DATA`, `EVAL`, `RUN_CONFIG`.

- [ ] **Step 6: Run fixture tests.**

Run:

```bash
uv run pytest tests/e2e/test_test_homecredit_project.py tests/qa/test_configured_service_namespaces.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit.**

```bash
git add tests/e2e/fixtures/projects/test_homecredit/project.py tests/e2e/fixtures/projects/test_homecredit/README.md tests/e2e/test_test_homecredit_project.py tests/shared/e2e_project_fixture.py tests/qa/test_configured_service_namespaces.py tests/qa/test_configured_services_live.py tests/e2e/test_test_homecredit_live_smoke.py
git rm tests/e2e/fixtures/projects/test_homecredit/automl_config.yaml tests/e2e/fixtures/projects/test_homecredit/data.py tests/e2e/fixtures/projects/test_homecredit/evaluation.py
git commit -m "test(e2e): migrate smoke fixture to typed project.py"
```

---

## Task 7: Update Docs, Comments, and Ratchets

**Files:**
- Modify: `automl/core/feature_registry.py`
- Modify: `automl/data/pipeline.py`
- Modify: `automl/eval/base.py`
- Modify: `automl/cli/project.py`
- Modify: `references/setup/model-contract.md`
- Modify: `references/setup/snowflake.md`
- Modify: `tests/contracts/test_skill_plugin_contract.py`
- Modify or create: `tests/contracts/test_project_entrypoint_cutover.py`

- [ ] **Step 1: Update stale comments and docs.**

Replace active references as follows:

```text
projects/<project_name>/data.py -> projects/<project_name>/project.py DATA
projects/<project_name>/evaluation.py -> projects/<project_name>/project.py EVAL
data.py -> project.py DATA
evaluation.py -> project.py EVAL
automl_config.yaml -> project.py RUN_CONFIG
```

Do not edit historical design docs under `docs/to-do/` unless a contract reads them as active guidance.

- [ ] **Step 2: Fix the contradictory skill contract test.**

In `tests/contracts/test_skill_plugin_contract.py`, update `test_agent_docs_use_project_based_context_paths` so it asserts:

```python
assert "projects/<project_name>/PROJECT_INSTRUCTIONS.md" in combined
assert "projects/<project_name>/project.py" in combined
assert "projects/<project_name>/automl_config.yaml" not in combined
assert "projects/<project_name>/data.py" not in combined
assert "projects/<project_name>/evaluation.py" not in combined
```

- [ ] **Step 3: Add a retired entry-point ratchet.**

Create `tests/contracts/test_project_entrypoint_cutover.py`:

```python
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

CHECKED_ROOTS = [
    ROOT / "automl",
    ROOT / "skills",
    ROOT / "agents",
    ROOT / "references",
    ROOT / "tests" / "unit",
    ROOT / "tests" / "integration",
    ROOT / "tests" / "e2e",
    ROOT / "tests" / "shared",
    ROOT / "tests" / "qa",
]

FORBIDDEN = [
    "automl_config.yaml",
    "from automl.eval import EvaluationSpec",
    "projects/<project_name>/evaluation.py",
    "projects/<project_name>/data.py",
]

ALLOWLIST = {
    "tests/contracts/test_project_entrypoint_cutover.py",
}


def test_active_surfaces_do_not_reference_retired_project_entrypoints() -> None:
    failures: list[str] = []
    for root in CHECKED_ROOTS:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix not in {".py", ".md"}:
                continue
            rel = path.relative_to(ROOT).as_posix()
            if rel in ALLOWLIST:
                continue
            text = path.read_text(encoding="utf-8")
            for forbidden in FORBIDDEN:
                if forbidden in text:
                    failures.append(f"{rel}: contains {forbidden!r}")
    assert failures == []
```

- [ ] **Step 4: Run the ratchets and fix reported active references.**

Run:

```bash
uv run pytest tests/contracts/test_project_entrypoint_cutover.py tests/contracts/test_skill_plugin_contract.py -q
```

Expected before cleanup: FAIL listing stale references. Update only active docs/tests/runtime comments, then rerun until PASS.

- [ ] **Step 5: Commit.**

```bash
git add automl/core/feature_registry.py automl/data/pipeline.py automl/eval/base.py automl/cli/project.py references/setup/model-contract.md references/setup/snowflake.md tests/contracts/test_skill_plugin_contract.py tests/contracts/test_project_entrypoint_cutover.py
git commit -m "test(contracts): ratchet strict project.py entrypoint"
```

---

## Task 8: Full Verification and Final Cleanup

**Files:**
- Any files reported by verification commands.

- [ ] **Step 1: Remove stray untracked file if it is still present.**

Run:

```bash
git status --short
```

If `projects/payment_routing/Untitled` is still present and untracked, delete it:

```bash
rm projects/payment_routing/Untitled
```

- [ ] **Step 2: Search for retired active references.**

Run:

```bash
rg -n "automl_config\\.yaml|EvaluationSpec|projects/<project_name>/evaluation\\.py|projects/<project_name>/data\\.py|ctx\\.config_path|config_path" automl skills agents references tests --glob '!**/__pycache__/**'
```

Expected: no active references. If results remain, either remove them or add a narrow contract allowlist only when the reference is intentionally testing retired text.

- [ ] **Step 3: Run unit and contract suites.**

Run:

```bash
uv run pytest tests/unit tests/contracts -q
```

Expected: PASS.

- [ ] **Step 4: Run integration suite.**

Run:

```bash
uv run pytest tests/integration -q
```

Expected: PASS.

- [ ] **Step 5: Run e2e fixture shape tests.**

Run:

```bash
uv run pytest tests/e2e/test_test_homecredit_project.py -q
```

Expected: PASS.

- [ ] **Step 6: Re-run the data-prep command.**

Run:

```bash
uv run python -m automl.data.prepare --project-root . --project example_homecredit --dry-run
```

Expected: no `automl_config.yaml not found` failure. If the command fails for local service credentials or missing active snapshot setup, record the exact non-YAML failure in the final handoff.

- [ ] **Step 7: Commit any verification fixes.**

```bash
git status --short
git add -A
git commit -m "fix: complete strict project.py cutover"
```

Use this commit only if verification found additional cleanup after Tasks 1-7.

---

## Final Acceptance Criteria

- `ProjectContext` discovers only `projects/*/project.py`.
- `ProjectContext` exposes `project_path`, not `config_path`.
- Runtime paths do not check `automl_config.yaml`.
- Validation requires `TASK`, `DATA`, `EVAL`, and `RUN_CONFIG`.
- `ModelRoute.effort` accepts only `low`, `medium`, and `high`.
- Integration tests no longer create YAML-only projects.
- The e2e smoke fixture is a typed `project.py` project.
- Active docs and ratchets reject retired entry-point references.
- `uv run pytest tests/unit tests/contracts -q` passes.
- `uv run pytest tests/integration -q` passes.
