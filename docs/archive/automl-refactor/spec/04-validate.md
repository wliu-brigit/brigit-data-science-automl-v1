# 04 — Validate Framework + Registry

**Status:** DONE (2026-05-23, second review applied) — two rounds of
three-agent review run; all findings integrated.
**Parent:** `00-structural-design.md` §9.2, §13.2, §15.2 (deferred items)
**Workflow:** see `README.md` "Sub-spec workflow"

This sub-spec settles the interfaces for the **validate** framework: what
lives in `validate/`, what shape check functions take, and how
project-side custom checks plug in.

---

## 1. Scope

Validate is a **runtime check suite** invoked at three concrete moments:

| Moment | Trigger | Cost of failure |
|---|---|---|
| **DS pre-flight** | `automl validate project` (DS runs before kicking off trials; also via `/validate` skill) | Bad config / env / TBD placeholder caught before any compute |
| **Runner pre-fit** | Automatic, every trial — runner loads a 200-row sample and calls `validate.model(...)` before opening any MLflow run | "Model won't even fit on 200 rows" caught before burning a trial slot + GCS / MLflow state |
| **Proposer → coder** | Automatic, between agent turns (via `automl validate proposal`) | Malformed proposal caught before a coder agent runs |

Validate is **not** a type checker, a linter, or anything that runs at
editor / import time. It is a small library of check functions that
produce a uniform `ValidationReport`.

## 2. What we have today

- **`validate/types.py`** — `Issue`, `ValidationReport`, `Severity`
  (`Literal["error","warning","info"]`), `Target`
  (`Literal["model","config","contracts","proposal","project"]`),
  `CheckSpec`.
- **`validate/registry.py`** — `@register(target=...)` decorator +
  module-level `_CHECKS` dict + `discover_project_checks(...)` for
  lazy-importing `projects/<name>/validators.py` + `_RESET_FOR_TESTS`.
- **`validate/targets.py`** — five orchestrators (`project`, `config`,
  `contracts`, `model`, `proposal`); each calls `register_all()`
  defensively, then `get_checks(target=...)`, then `_run()` with
  `inspect.signature`-based kwarg filtering + **per-check
  exception-wrapping** (one bad check emits a `*.crashed` Issue rather
  than crashing the whole report).
- **`validate/builtin/*_checks.py`** — five files declaring built-in
  checks via `@register`. Auto-imported by `builtin/__init__.py`.

Callers + parallel systems:

- **Runner** (`runner/_execute.py:312`) — pre-fit happens BEFORE any
  MLflow run is opened. Today `validate.model()` internally loads
  `build_pipeline(ctx, dry_run=True).load_data_snapshot().df_train.head(200)`.
  Pre-fit failures go through `_finish_without_mlflow_run` (no MLflow
  record). The MLflow run is opened after pre-fit succeeds; the main
  data load happens after, so main-fit-time load failures get an MLflow
  run that captures them.
- **CLI** (`cli/validate.py`) — exposes 5 sub-verbs.
- **`cli/propose.py`** — separate `automl propose validate` verb with
  a `--output <path>` flag (writes validated JSON on pass). Used by
  `skills/automl/scripts/render_context.py::safe_commands.persist_proposal`.
- **`cli/trial.py:76`** — third quiet caller of `propose.validate()`.
- **DS skill** (`/validate`, `/setup`) — documents only
  `automl validate project`.
- **Parallel propose system** (`propose/__init__.py`, `propose/schema.py`)
  — defines its own `Issue` / `ValidationReport` (with a *constructor*
  `passed: bool` field, not a property) + its own `propose.validate()`
  function.

Three duplicate `Issue` / `ValidationReport` dataclasses today: in
`validate/types.py`, `propose/schema.py`, and `propose/__init__.py`.

## 3. Folder shape (final)

```
validate/
├── __init__.py          ← public surface (Issue, ValidationReport, Severity, Target, project, model, proposal)
├── base.py              ← Issue, ValidationReport, Severity = Literal["error","warning"], Target = Literal["project","model","proposal"]
├── targets.py           ← project(...), model(...), proposal(...) + _safe + _try_fit + _import_project_validators
└── synthetic.py         ← make_synthetic_fixture(rows=50)
```

Per the structural spec §9.2, **check logic lives in `<domain>/checks.py`**:

| File | Absorbs |
|---|---|
| `project/checks.py` | Today's `config_checks.py`, `env_checks.py`, and the project-side part of `contract_checks.py` (TASK export check). **`check_project_placeholders` is dropped** — sub-spec 01's None-semantics for unfilled config fields supersede the `<TBD>`-string scanner. |
| `data/checks.py` | `check_data_module_exports` (DATA / DataSpec checks). |
| `eval/checks.py` | `check_evaluation_module_exports` + the ~80-line `_probe_evaluation_shape` helper. |
| `model/checks.py` | `subclass_basemodel`, `fit_succeeds`, `post_fit_attrs_set`, the `REQUIRED_POST_FIT_ATTRS` constant. |
| `agent/checks.py` | `proposal_schema` — absorbs today's `propose.validate()` logic + the adapter in `proposal_checks.py`. |
| `project/_imports.py` | The `_load_project_module(session, module_name)` helper (used by data + eval + project checks to import the project's `config` module). |

This is a *file move + small absorption* of today's
`validate/builtin/*_checks.py` into the domains they describe — not a
rewrite of the check logic.

## 4. Decisions

### Q1 — Registry's role: direct calls, no decorator

**Decision (2026-05-22):** built-in check functions are imported and
invoked by name inside their orchestrator. No `@register` decorator for
built-ins. No `_CHECKS` global. No `inspect.signature` filtering. No
`register_all()` defensive helper. **`CheckSpec` dataclass deleted.**

**Per-check exception-wrapping is preserved** (revision 2026-05-23, per
review): orchestrators use a small `_safe` helper that wraps each call;
a crashing check emits `Issue(level="error", check="<name>.crashed",
message=...)` instead of taking down the whole orchestrator. Same
behavior as today's `_run()`, just made explicit at each call site.

**Shape (Q4-final signature, with `_safe` wrapping):**
```python
# validate/targets.py
def model(cls, *, df, registry) -> ValidationReport:
    from automl.model.checks import (
        subclass_basemodel, fit_succeeds, post_fit_attrs_set,
    )
    issues: list[Issue] = []
    issues.extend(_safe("model.subclass_basemodel", subclass_basemodel, cls=cls))
    if any(i.level == "error" for i in issues):
        return ValidationReport(issues=issues)
    instance, error, error_stage = _try_fit(cls, df, registry, seed=0)
    issues.extend(_safe("model.fit_succeeds", fit_succeeds,
                        cls=cls, instance=instance,
                        error=error, error_stage=error_stage))
    if error is None:
        issues.extend(_safe("model.post_fit_attrs_set", post_fit_attrs_set,
                            cls=cls, instance=instance))
    return ValidationReport(issues=issues)


def _safe(name: str, fn, **kwargs) -> list[Issue]:
    try:
        return list(fn(**kwargs))
    except Exception as exc:  # noqa: BLE001 — intentional broad catch
        return [Issue(level="error", check=f"{name}.crashed",
                      message=f"check {name!r} raised {type(exc).__name__}: {exc}")]
```

**Implication for structural-spec §15.2:** "Validate registry import
timing" item dissolves — each domain's `checks.py` is imported on first
orchestrator call by ordinary Python semantics.

### Q2 — Project-side custom checks: project target only

**Decision (2026-05-22, scoped down 2026-05-23 per B2 review):**
`projects/<name>/validators.py` declares a top-level
`PROJECT_CHECKS: dict[str, list[Callable]]` keyed by target. The
orchestrator imports the module on first call (cached per
`(repo_root, project_name)`), reads `PROJECT_CHECKS.get(target, [])`,
runs each function via the same `_safe` wrapper.

**Project-side checks support ONLY the `"project"` target.** Per
`feedback_extension_points_follow_demand`: zero real projects use
project-side model/proposal checks today (one test exercises the model
case; no `validators.py` exists in `projects/` or `kaggle_home_credit/`).
The simpler API beats supporting speculative use. Add back the day a
real project asks for it.

**Pinned signature:**

| Target | Required signature |
|---|---|
| `"project"` | `fn(*, session: Session) -> Iterable[Issue]` |

Authors who don't need `session` use `**_` to absorb it. Per sub-spec
01's parameter convention.

**Shape:**
```python
# projects/<name>/validators.py
from collections.abc import Iterable
from automl.validate import Issue
from automl.project import Session

def cardholder_id_present(*, session: Session) -> Iterable[Issue]:
    if "cardholder_id" not in session.config.DATA.required_columns:
        return [Issue(level="error", check="proj.card_id_required",
                      message="DATA must include cardholder_id for this project")]
    return []

PROJECT_CHECKS = {"project": [cardholder_id_present]}
```

```python
# validate/targets.py
_PROJECT_VALIDATORS_CACHE: dict[tuple[str, str], ModuleType | None] = {}

def _import_project_validators(session: Session) -> ModuleType | None:
    """Lazy import of projects/<name>/validators.py. Cached per (root, name)."""
    root = str(session.repo_root.resolve())
    key = (root, session.project_name)
    if key in _PROJECT_VALIDATORS_CACHE:
        return _PROJECT_VALIDATORS_CACHE[key]
    candidate = session.repo_root / "projects" / session.project_name / "validators.py"
    if not candidate.exists():
        _PROJECT_VALIDATORS_CACHE[key] = None
        return None
    digest = hashlib.sha1(str(candidate.resolve()).encode()).hexdigest()[:12]
    module_name = f"_automl_project_validators_{digest}_{session.project_name}"
    spec = importlib.util.spec_from_file_location(module_name, candidate)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load validators from {candidate}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    _PROJECT_VALIDATORS_CACHE[key] = module
    return module

def _run_project_checks(target: str, *, session: Session, **kwargs) -> list[Issue]:
    module = _import_project_validators(session)
    if module is None:
        return []
    fns = getattr(module, "PROJECT_CHECKS", {}).get(target, [])
    issues: list[Issue] = []
    for fn in fns:
        issues.extend(_safe(f"project_custom:{fn.__name__}", fn, session=session, **kwargs))
    return issues


def _RESET_FOR_TESTS() -> None:
    """Clear the project-validator cache for test isolation."""
    _PROJECT_VALIDATORS_CACHE.clear()
    for module_name in list(sys.modules):
        if module_name.startswith("_automl_project_validators_"):
            del sys.modules[module_name]
```

`_import_project_validators` preserves today's sha1-digested
`sys.modules` name (so two projects sharing a name across different
repo roots don't collide). `_RESET_FOR_TESTS` survives — same shape as
legacy, moved to `validate/targets.py` and rescoped to the project-
validator cache only.

### Q3 — Canonical target set: three orchestrators

**Decision (2026-05-22): three orchestrators — `project`, `model`,
`proposal`.** Map 1-to-1 to the three real caller intents. Drop
separate `config` and `contracts` orchestrators + CLI sub-verbs (their
checks still run inside `project`). Don't add `experiment` until a real
check appears.

**Coverage is unchanged from today** — same checks run; only the CLI
surface shrinks. The `Target` literal shrinks to
`Literal["project", "model", "proposal"]`.

**Future-target principle:** new orchestrator appears when (a) a new
concrete caller emerges with (b) at least one real check.

### Q4 — Sample-data ownership: runner pulls SMALL sample forward (REVISED 2026-05-23)

**Decision (2026-05-22, clarified 2026-05-23 per B1 review):**
`validate.model()`'s signature becomes `(cls, *, df, registry)` —
caller builds the sample, validate runs the checks.

**Critical clarification:** the runner pulls forward **only the 200-row
pre-fit sample**, NOT the full snapshot. This preserves today's
observability semantics:

```
phase 1: load_pre_fit_sample (~200 rows)    ← runner; no MLflow run yet
phase 2: pre_fit_validation                 ← validate.model(cls, df=, registry=)
         (fail → _finish_without_mlflow_run, same as today: NO MLflow record)
phase 3: open MLflow run                    ← same point in lifecycle as today
phase 4: load full snapshot + fit + eval + log  ← same as today
                                              (full-load failure → captured by MLflow run)
```

The earlier framing ("load snapshot once, reuse") would have moved the
**full** snapshot load before the MLflow run, losing today's MLflow
capture of full-load failures. The actual change is smaller: the small
pre-fit sample moves from inside validate to the runner; the full data
load stays exactly where it is.

**Runner pre-fit sample load:**
```python
# runner — phase 1
pipeline = build_pipeline(session.ctx, dry_run=session.dry_run)
pre_fit_snapshot = pipeline.load_data_snapshot()
df_pre_fit = pre_fit_snapshot.df_train.head(200)
registry = pre_fit_snapshot.registry
# (pipeline cache likely makes the phase-4 reload cheap; not relied on)
```

Note: today's pre-fit hardcoded `dry_run=True` for the sample. The new
ordering uses `session.dry_run` — aligns with sub-spec 03's "dry_run is
a session-wide container, never per-operation" principle. Small
intentional behavior improvement.

**Validate.model() final signature:**
```python
def model(cls, *, df, registry) -> ValidationReport
```

No `sample_kind` parameter. Error messages from `fit_succeeds` already
include the row count + exception type + class name — sufficient
diagnostic. "Synthetic vs real" is implicit from which CLI command the
user ran.

**Callers:**
- **Runner:** `validate.model(cls, df=df_pre_fit, registry=registry)`
- **CLI helper** (`cli/validate.py`): if `--sample-from` given, resolve
  session, call `build_pipeline(...).load_data_snapshot().df_train.head(200)`,
  then `validate.model(...)`. Otherwise use
  `validate.synthetic.make_synthetic_fixture(rows=50)`.
- **Tests:** build a tiny inline DataFrame + FeatureRegistry, or call
  `validate.synthetic.make_synthetic_fixture()`.

**`ModelProbe` dataclass is deleted.** The orchestrator's private
`_try_fit(cls, df, registry, *, seed=0)` helper fits once and returns
`(instance, error, error_stage)`; model checks accept these as keyword
args. `make_model_probe` and `sample_load_failed_probe` go with it.

**Why A1 over A2 (convenience helper) / A3 (synthetic-only):**
- A1 puts data-loading where it belongs — the runner is already in the
  data-loading business. Validate becomes a pure check runner.
- A2 (`validate.model_from_project` wrapper) preserves the
  `validate → data` import, which was the cause we wanted to remove.
- A3 is a real regression — real-sample pre-fit catches column-type
  mismatches synthetic can't.

### Q5 — Proposal target: collapse the parallel `propose.validate()`

**Decision (2026-05-22):** delete `propose.validate()` and its private
`Issue` / `ValidationReport` types. Schema check lives in
`agent/checks.py::proposal_schema(*, proposal, allowed_dependencies)
-> Iterable[Issue]` returning canonical `validate.Issue` directly — no
translation. `validate.proposal()` is a thin orchestrator that calls it
and bundles via `_safe`.

**Final placement:**
- `agent/proposal.py` — `Proposal` dataclass + schema constants
  (`REQUIRED_FIELDS`, `OPTIONAL_FIELDS`, `DISALLOWED_FIELDS`, `SLUG_RE`)
- `agent/checks.py::proposal_schema(...)` — the schema check
- `validate/targets.py::proposal(...)` — 3-line orchestrator

**Migration footprint:** `propose.validate()` has **three** callers
today:
1. `cli/propose.py` — separate `automl propose validate` verb. The
   verb collapses into `cli/validate.py`'s `proposal` sub-verb. **The
   `--output <path>` flag is preserved** (added to the `validate proposal`
   sub-verb) so `skills/automl/scripts/render_context.py::safe_commands.persist_proposal`
   keeps working.
2. `cli/trial.py:76` — third quiet caller. Switches to
   `validate.proposal(...)`.
3. `validate/builtin/proposal_checks.py` — adapter, deleted.

Plus `skills/propose/SKILL.md` (likely references the old verb) — the
implementation plan updates it.

**`ValidationReport` constructor change:** the legacy
`propose/__init__.py` uses `ValidationReport(passed=False, issues=...)`
~15 times. Canonical `ValidationReport` has `passed` as a `@property`,
not a constructor argument. The deletion of `propose.validate()` and
its private types makes those 15 call sites disappear in the same
change — but if any survive, they must rewrite to
`ValidationReport(issues=...)` and let `.passed` derive.

### Q6 — Closeout: three small loose ends

**Q6a — `synthetic.py` home: keep in `validate/`.** Used by tests +
the CLI helper's no-`--sample-from` sanity mode. `make_synthetic_fixture`'s
signature simplifies to `make_synthetic_fixture(*, rows: int = 50)` —
the legacy `n_numeric`/`n_categorical`/`target_col`/`seed` kwargs always
took defaults; drop them. Imports `FeatureRegistry` from
`automl.data.features` (sub-spec 05 confirms the exact path); single
test-fixture cross-domain import, layering-acceptable (same pattern as
`mlflow/` importing domain types).

**Q6b — `Severity = Literal["error", "warning"]`.** Drop `"info"` (no
emitter, no caller). `.passed` semantics unchanged.

**Q6c — delete `ValidationReport.raise_if_failed()` and `ValidationError`.**
No caller. Per `feedback_extension_points_follow_demand`.

---

## 5. Summary — the final shape

```
validate/
├── __init__.py          ← Issue, ValidationReport, Severity, Target, project, model, proposal
├── base.py              ← Issue, ValidationReport, Severity, Target
├── targets.py           ← project(...), model(...), proposal(...) + _safe + _try_fit + project-validators import
└── synthetic.py         ← make_synthetic_fixture(*, rows=50)

<domain>/checks.py       ← direct check functions, return Iterable[Issue]
project/_imports.py      ← _load_project_module(session, module_name) helper
projects/<n>/validators.py  ← optional, exports PROJECT_CHECKS = {"project": [fn]}
```

**Canonical types (in `validate/base.py`):**
```python
@dataclass
class Issue:
    level: Literal["error", "warning"]
    check: str
    message: str
    location: str | None = None

@dataclass
class ValidationReport:
    schema_version: int = 1
    issues: list[Issue] = field(default_factory=list)

    @property
    def passed(self) -> bool: return not any(i.level == "error" for i in self.issues)

    def to_json(self) -> dict[str, Any]: ...

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ValidationReport":
        """Strips unknown keys (additive-only schema per sub-spec 02)."""
        ...
```

`schema_version: int = 1` + `from_dict` follow sub-spec 02's pattern
(`ValidationReport.to_json()` is persisted as a trial artifact by the
runner; counts as a typed schema).

**Public orchestrator surfaces** (`session` per sub-spec 01 convention):
```python
validate.project(*, session: Session | None = None) -> ValidationReport
validate.model(cls, *, df, registry) -> ValidationReport
validate.proposal(*, proposal: dict, allowed_dependencies: list[str]) -> ValidationReport
```

`session` defaults to `automl.session()` (the active session), matching
sub-spec 01 §4.

**CLI verbs:**
```
automl validate project --project <name>
automl validate model --module <m> --class-name <c> [--sample-from <name>]
automl validate proposal --json <path> [--output <path>] [--allowed-deps-file <path> | --allowed-dependencies-json <json>]
```

**Deletions vs status quo:**
- `validate/registry.py` (decorator + `_CHECKS` global; `_RESET_FOR_TESTS` migrates to `validate/targets.py`)
- `validate/types.py::CheckSpec` (no callers post-refactor)
- `validate.config()` / `validate.contracts()` orchestrators + CLI sub-verbs
- `propose.validate()` + private Issue / ValidationReport types in `propose/`
- `validate/builtin/proposal_checks.py` adapter
- `ValidationError` (`errors.py`), `ValidationReport.raise_if_failed()`
- `Severity = "info"` level
- `Target` literal pruned from 5 entries to 3 (`"project"`, `"model"`, `"proposal"`)
- `ModelProbe` dataclass + `make_model_probe` + `sample_load_failed_probe`
- Project-side support for `"model"` / `"proposal"` targets (only `"project"` survives)
- Cross-domain data-loading from inside `validate.model()`
- `register_all()` defensive helper
- `inspect.signature`-based kwarg filtering
- `make_synthetic_fixture`'s `n_numeric`/`n_categorical`/`target_col`/`seed` kwargs
- `check_project_placeholders` (TBD-string scanner; superseded by sub-spec 01's None-semantics)

**Preserved:**
- All current check *coverage* (every error caught today still caught)
- The three caller intents and their public Python / CLI surfaces
- Project-side extension for `"project"` target via `PROJECT_CHECKS` dict
- Synthetic-fixture helper (simplified signature)
- Per-check exception-wrapping (now via `_safe` helper)
- `_RESET_FOR_TESTS` for unit-test isolation (moved + rescoped)
- `cli propose validate --output` semantics (carried into `validate proposal --output`)
- Today's observability: pre-fit failures → no MLflow record; full-load failures during fit → MLflow run captures
- `ValidationReport.to_json()` shape (additive — adds `schema_version: 1`; readers using `from_dict` are forward-compatible)

## 6. Carry-backs to parent specs

- **Structural spec §11.1 (CLI verb catalog):** `validate` entry lists
  only `project`, `model`, `proposal` as sub-verbs (drop `config`,
  `contracts`). `validate proposal` accepts `--output` (preserved from
  legacy `propose validate`).
- **Structural spec §13.1 (schema location table):** `CheckSpec` row
  removed. `Issue`, `ValidationReport`, `Severity`, `Target` move from
  `validate/types.py` to `validate/base.py` (rename). `ValidationReport`
  gains `schema_version: int = 1` + `from_dict` per sub-spec 02's pattern.
- **Structural spec §15.2:** "Validate registry import timing" item
  removed — Q1 resolves it.
- **Structural spec §17.8 (per-domain check growth):** prose currently
  says "registry behavior is unchanged" when discussing the future
  `<domain>/checks/` folder split. Update to: "the orchestrator's
  imports point at the new folder; no other behavior change."
- **Sub-spec 05 (Data):** confirm `FeatureRegistry`'s post-refactor
  module path (`automl.data.features`). `validate/synthetic.py` tracks it.
- **Sub-spec 11 (Agent):** confirm `agent/proposal.py`
  (Proposal dataclass + schema constants) + `agent/checks.py::
  proposal_schema` placement absorb the deleted `propose/` module.
  (Was "sub-spec 09" before the experiment domain was split into
  experiment/trial/agent; the Proposal contract lives in `agent/`.)

## 7. Open items

Closeout tasks (bookkeeping):
- Flip migration-checklist rows for symbols whose new home is now pinned
- Mark `open-questions.md` items resolved by this sub-spec
- Update `README.md` "What's done" section + sequence table row
- Apply the carry-backs above to parent docs

Implementation-plan work flagged here (out of design scope):
- Update three `propose.validate()` call sites + `skills/propose/SKILL.md`
- Reorder runner phases: small pre-fit sample → pre-fit → MLflow run open → full snapshot load + fit (no change to MLflow-open / data-load relative ordering for the main fit)
- Update/delete obsolete tests: `test_validate_registry.py`,
  `test_validate_proposal_adapter.py`, `test_errors_hierarchy.py`
  (ValidationError row), `test_validate_types.py` (info-level +
  raise_if_failed tests), `test_propose_schema.py` (import paths),
  **`test_validate_project_aggregator.py`** (uses `@register`; model-
  target project-check tests deleted per Q2 scope-down),
  **`test_runner_prefit_validate.py:88`** (monkeypatches old
  `validate.model` signature)
- Decide at implementation time whether `validation_report.json` gets a
  typed writer at `mlflow/artifacts/validation_report.py` (sub-spec 02
  pattern); sub-spec 02 doesn't require it for ad-hoc trial-result fields
- Update `cli/validate.py` exception handling — confirm
  `(ProjectError, FileNotFoundError, ValueError)` coverage matches the
  exceptions the new orchestrators can raise

## 8. Sub-spec status

**Status: DONE (2026-05-23).** Two rounds of three-agent review run,
all findings integrated. Pending workflow closeout: open-questions,
migration-checklist, README updates.
