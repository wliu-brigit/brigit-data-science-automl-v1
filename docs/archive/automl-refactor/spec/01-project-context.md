# Project Context Threading — Sub-Spec Design

**Date:** 2026-05-19
**Parent spec:** `00-structural-design.md` §13.5
**Status:** Design approved; ready to inform implementation plan.
**Scope:** How project identity, recipe, environment, and active session state thread through the refactored library.

This is the **highest-priority sub-spec** before implementation begins. The structural spec settled what code goes where; this sub-spec settles how every function in that code accesses the cross-cutting ambient state.

---

## 1. Why this sub-spec exists

The structural spec (§13.5) explicitly deferred the *mechanism* of project-context threading, having only settled that it's a cross-cutting concern every domain reads. That deferral existed because:

- "Project context" means different things at different call sites (some need only a name; some need the loaded recipe; some need env-derived values).
- An "exploration phase" exists where a project folder is present but `config.py` may be empty or partial — the framework must not crash here.
- CLI flags need to override recipe values at invocation time without mutating the recipe on disk.
- The threading rule (explicit param vs ambient contextvar) must be uniform across the codebase so contributors don't drift.

Past refactors have produced inconsistent answers to these questions and the friction has compounded. This sub-spec defines one answer.

## 2. The three names

```
   config.py            ProjectConfig           Session
   ─────────            ─────────────           ───────
   the FILE       →     the LOADED OBJECT  →    the ACTIVE STATE
   user edits           (eager + validated)     (for this process)
   on disk              (immutable)             (in contextvar)
```

| Name | What it IS | When it exists | Mutable? |
|---|---|---|---|
| `config.py` | the recipe **file** on disk: `projects/<name>/config.py` | after `automl project init` | yes (user edits) |
| `ProjectConfig` | the **loaded, validated Python object** — combines `config.py` contents + env values, eagerly resolved | constructed once by `use_project()` or `ProjectConfig.load()` | NO (frozen dataclass) |
| `Session` | the **active state** for this Python process — wraps a `ProjectConfig` + active experiment_id override + dry_run | for the lifetime of `use_project()`'s effect | mutable via `replace()`; held in contextvar |

Naming rules:
- The word "Settings" is not used. "Settings" was an arbitrary placeholder; "Config" matches the file (`config.py`) and the existing typed config classes (`RunConfig`, `DataSpec`).
- The word "context" is not used as a noun in this layer. `ProjectContext` from the legacy code is gone — its responsibilities split between `ProjectConfig` and `Session`.

## 3. The objects

### 3.1 `ProjectConfig` — immutable, loaded view

```python
# project/config.py
@dataclass(frozen=True)
class ProjectConfig:
    # ── Identity (always available after load) ──
    project_name: str
    repo_root: Path
    project_dir: Path
    project_package: str          # e.g. "projects.payment_routing"
    config_path: Path             # projects/<name>/config.py (may not exist)
    instructions_path: Path       # projects/<name>/PROJECT_INSTRUCTIONS.md (may not exist)

    # ── Recipe (from config.py; None when unfilled) ──
    task: Task | None
    data_spec: DataSpec | None
    eval_spec: EvalSpec | None
    run_config: RunConfig | None
    required_transformers: list[RequiredTransformer]   # from config.py's REQUIRED_TRANSFORMERS; [] when none declared (sub-spec 06). Type imported from model/preprocessing.py — load-time only, late-imported in load() (same acyclic pattern as eval_spec).

    # ── Environment (from env vars + .env file; empty string when missing) ──
    gcs_bucket: str
    gcs_prefix: str
    mlflow_tracking_uri: str
    mlflow_artifacts_destination: str       # MLFLOW_ARTIFACTS_DESTINATION; threaded into `mlflow gc` by cleanup (sub-spec 03 §6.4)

    # ── Readiness ──
    def is_complete(self) -> bool:
        """True when all four recipe fields are non-None."""
        return all(getattr(self, f) is not None for f in
                   ("task", "data_spec", "eval_spec", "run_config"))

    def missing_fields(self) -> list[str]:
        """List of recipe field names that are still None."""
        return [name.upper() for name in
                ("task", "data_spec", "eval_spec", "run_config")
                if getattr(self, name) is None]

    # ── Strict accessors — raise clearly if a needed field is None ──
    def require_task(self) -> Task: ...
    def require_data_spec(self) -> DataSpec: ...
    def require_eval_spec(self) -> EvalSpec: ...
    def require_run_config(self) -> RunConfig: ...

    # ── Derived convenience ──
    @property
    def target_column(self) -> str:
        """Standardized target column name. Raises if task is None."""
        ...

    @classmethod
    def load(cls, name: str, *, repo_root: Path | None = None) -> "ProjectConfig":
        """Single entry point for constructing a ProjectConfig from disk.
           See §7 for the step-by-step contract."""
        ...
```

**Why frozen:** the loaded view is the reproducible artifact every other piece of code is built on. If a `ProjectConfig` could mutate after construction, every cache-invalidation question gets harder. Frozen is the discipline that makes the rest of the system simple.

**Why None for unfilled:** see §5.

### 3.2 `Session` — active state, contextvar-held

```python
# project/session.py
@dataclass(frozen=True)
class Session:
    config: ProjectConfig

    # CLI / programmatic overrides for THIS process:
    dry_run: bool = False
    namespace: str = ""                    # isolation prefix (e.g. "qa"); "" = real. From the top-level --namespace flag / env. Segregates MLflow + GCS + local trial dirs as a full universe, orthogonal to (and composable with) dry_run. Renamed from legacy `route_namespace` (clean cut).
    experiment_id: str | None = None       # overrides config.run_config.experiment_id

    @property
    def active_experiment_id(self) -> str:
        """Effective experiment id: CLI override wins, falls back to recipe declaration."""
        if self.experiment_id is not None:
            return self.experiment_id
        if self.config.run_config is None:
            raise ProjectError(
                "no experiment_id set: RUN_CONFIG missing from "
                f"{self.config.config_path} and no --experiment override given"
            )
        return self.config.run_config.experiment_id

    # Convenience proxies (delegate to config; keep call sites concise):
    @property
    def project_name(self) -> str:
        return self.config.project_name


_ACTIVE_SESSION: ContextVar[Session | None] = ContextVar(
    "automl_active_session", default=None
)
```

**Why frozen:** changes happen via `dataclasses.replace()` returning a new `Session`. Mutation in place is a code smell; explicit replacement is auditable.

**Why contextvar:** matches the existing pattern, async-safe and thread-safe by construction.

## 4. Entry point and accessor

### 4.1 `use_project(name, **session_kwargs) -> Session`

Single entry point. The notebook / CLI / skill all call this **at process start**.

```python
def use_project(
    name: str,
    *,
    repo_root: Path | None = None,
    dry_run: bool = False,
    namespace: str = "",
    experiment_id: str | None = None,
) -> Session:
    """Load a project's ProjectConfig, build a Session, set it as active.

    Intended for top-level process setup (notebook cell, CLI verb entry, skill
    bootstrap) — call this once near the start of execution. Returns the
    Session so callers can also use it explicitly.

    Always succeeds if projects/<name>/ exists — partial configs are allowed
    (see §5). Raises ProjectError only when the folder itself is missing
    or config.py is structurally malformed.

    NOT for nested or async-scoped switching. The contextvar update is NOT
    token-tracked here; calling use_project() inside one coroutine permanently
    clobbers the active session for sibling coroutines. Use active_session()
    (§4.3) instead for any scoped or concurrent switching.

    Also propagates the new session state to the mlflow persistence layer
    via _bind_mlflow_for(session) — see §4.6.
    """
    config = ProjectConfig.load(name, repo_root=repo_root)
    session = Session(
        config=config,
        dry_run=dry_run,
        namespace=namespace,
        experiment_id=experiment_id,
    )
    _ACTIVE_SESSION.set(session)
    _bind_mlflow_for(session)        # propagate to mlflow framework
    return session
```

### 4.2 `automl.session() -> Session`

The single access helper. Every Tier 2 function calls it:

```python
def session() -> Session:
    """Return the active Session. Raises if no project active."""
    s = _ACTIVE_SESSION.get()
    if s is None:
        raise ProjectError(
            "no active project — call automl.use_project(name) first"
        )
    return s
```

### 4.3 `automl.active_session(name, ...) -> ContextManager[Session]`

For scoped temporary switching (tests, multi-project tools, comparisons):

```python
@contextmanager
def active_session(name: str, **session_kwargs) -> Iterator[Session]:
    """Temporarily set the active project for the duration of the with-block.

    Used for tests, multi-project tooling, and cross-project comparisons.
    Restores the previous active session on exit AND re-fires the mlflow
    bind so the persistence layer sees the restored state (§4.6).
    """
    config = ProjectConfig.load(name)
    session = Session(config=config, **session_kwargs)
    token = _ACTIVE_SESSION.set(session)
    _bind_mlflow_for(session)              # bind to the scoped session
    try:
        yield session
    finally:
        _ACTIVE_SESSION.reset(token)
        prior = _ACTIVE_SESSION.get()
        if prior is not None:
            _bind_mlflow_for(prior)        # restore mlflow bind to prior session
        # If prior is None, mlflow remains bound to the inner session's values;
        # but automl.session() raises in that state, so subsequent mlflow calls
        # would never happen via the normal path.
```

### 4.4 `automl.clear_session() -> None`

Reset to no-active-project state. Mostly useful for tests:

```python
def clear_session() -> None:
    """Clear the active session. Subsequent automl.session() calls will raise."""
    _ACTIVE_SESSION.set(None)
```

### 4.5 `automl.update_session(**kwargs) -> Session`

Apply changes to the active Session in-process without reloading config:

```python
def update_session(**kwargs: Any) -> Session:
    """Atomically replace the active Session with a copy that has the given
    fields overridden. Returns the new Session.

    Required because Session is frozen — `dataclasses.replace(session, ...)`
    alone produces a new object without updating the contextvar, which
    silently drifts. This helper does both atomically AND re-fires the
    mlflow bind (§4.6) so the persistence layer sees the new values.
    It is the ONLY sanctioned way to mutate active session state.

    Use cases:
      - The agent loop wants to flip dry_run after a preflight check
      - A notebook user wants to switch experiment_id without re-loading config
      - Tests want to tweak a session-level flag mid-test
    """
    current = session()                       # raises if no active session
    new = dataclasses.replace(current, **kwargs)
    _ACTIVE_SESSION.set(new)
    _bind_mlflow_for(new)                     # propagate to mlflow framework
    return new
```

This helper closes the "frozen Session + contextvar held" loophole — `replace()`-without-set drift is a real footgun. Library and skill code that needs to flip a session flag MUST call `update_session()`; never call `_ACTIVE_SESSION.set(...)` directly outside the four entry-point helpers.

### 4.6 `_bind_mlflow_for(session)` — single source of truth for Session → mlflow bind

The mlflow framework (`automl.mlflow`) needs to know the same connection state Session holds (tracking URI, GCS bucket, project name, active experiment id, dry_run, etc.). When Session changes, those values must propagate to mlflow's contextvar via `mlflow.bind(...)` (see sub-spec 02 §5).

Rather than having three different entry-points (`use_project`, `update_session`, `active_session`) each spell out the translation independently, **one private helper does it** and the three entry-points call it:

```python
# project/session.py
def _bind_mlflow_for(session: Session) -> None:
    """Propagate this session's state to the mlflow persistence layer.

    Single source of truth for Session → mlflow.bind() translation.
    Called by use_project, update_session, and active_session whenever
    the active session changes. If you add a new Session field that
    mlflow needs, you change ONE helper — not three entry-points.
    """
    from automl import mlflow             # late import — avoids circularity at module load
    mlflow.bind(
        tracking_uri=session.config.mlflow_tracking_uri,
        bucket=session.config.gcs_bucket,
        gcs_prefix=session.config.gcs_prefix,
        project_name=session.config.project_name,
        experiment_id=(
            session.experiment_id
            if session.experiment_id is not None
            else (
                session.config.run_config.experiment_id
                if session.config.run_config is not None
                else None
            )
        ),
        dry_run=session.dry_run,
        namespace=session.namespace,
    )
```

Notes:

- **Late import** of `automl.mlflow` inside the function body avoids an import-cycle headache at module load. Project domain code knows about mlflow framework; framework doesn't know about Session. No cycle, but the late import is cheap and idiomatic.
- **Single field-list to maintain.** Adding a new Session field that mlflow needs is a one-line change here; the three entry-points stay unchanged.
- **No mlflow types leak into Session.** Session is a plain dataclass; `_bind_mlflow_for` is the only project/session.py code that imports from `automl.mlflow`.

This is the carry-back from sub-spec 02 §12, applied directly. The contract: **anywhere the active session changes, `_bind_mlflow_for(session)` fires in lock-step.** Three call sites, one helper, zero opportunities to drift.

## 5. None-semantics for unfilled config fields

`config.py` for a freshly-scaffolded project declares unfilled fields as `None`:

```python
# projects/payment_routing/config.py — what `automl project init payment_routing` writes:

from automl.project import RunConfig, Splits, ModelsConfig, ModelRoute  # Splits replaces Split — free-form named ranges (sub-spec 05 Q8)
from automl.project.task import BinaryClassification
from automl.data import DataSpec
from automl.data.sources import SnowflakeSource
from automl.eval import EvalSpec
from automl.eval.metrics import Auc


TASK: BinaryClassification | None = None
DATA: DataSpec | None = None
EVAL: EvalSpec | None = None
RUN_CONFIG: RunConfig | None = None
```

`ProjectConfig.load()` always succeeds if the project folder exists. What's filled is what's filled. No mode flag, no `exploration=True`, no string-detection for `<TBD>` placeholders. Three signals tell the DS what state they're in:

| Need | Use |
|---|---|
| "Tell me everything that's wrong with my project" | CLI verb: `automl validate project` |
| "Programmatically check if my config is ready" | `session.config.is_complete()` / `.missing_fields()` |
| "Use this thing — fail clearly if a required field isn't set" | `session.config.require_task()` (etc.) |

Functions that need a recipe field call `require_*()`. The error message includes the file path and the field name:

```
ProjectError: TASK is not set in projects/payment_routing/config.py
```

So:
- **Notebook-1 exploration:** `automl.use_project("payment_routing")` works. The DS can list datasets, poke at sources, etc. — anything that doesn't need recipe fields. Calls that DO need recipe fields raise clearly, pointing at what to fill.
- **Mid-fill state:** `TASK` set but `DATA` still `None`. The DS can do task-shaped things; data-shaped calls raise.
- **Fully configured:** every recipe field non-None and well-formed. Everything works.

No magic; the truth is in the file.

## 6. Per-function signature convention

Every public Tier 2 function follows the same pattern:

```python
def materialize(*args, session: Session | None = None, **kwargs) -> Dataset:
    s = session if session is not None else automl.session()           # one idiom, used identically everywhere
    name = s.config.project_name
    spec = s.config.require_data_spec()
    ...
```

The three rules:

1. **`session` is a keyword-only argument**, never positional. Avoids accidental positional collisions and makes the parameter discoverable at the call site.
2. **Default is `None`, resolved via `session if session is not None else automl.session()`.** Use the explicit `is not None` check (NOT `or`) — `or` would also fall back on any falsy session value, which is a latent footgun. Single idiom across the library. Grep-able.
3. **`automl.session()` raises `ProjectError`** if no Session is active. No silent fallback, no anonymous-project mode.

Tier-by-tier:

| Tier | Convention |
|---|---|
| **Tier 1 (facade verbs like `use_project`, `clear_session`)** | These MANAGE the session; they don't take one. Obvious from naming. |
| **Tier 2 (domain functions in `automl.data`, `automl.experiment`, etc.)** | Keyword-only `session: Session | None = None`. Resolved via the idiom. |
| **Tier 3 (ABCs like `BaseModel.fit`, `DataSource.load`, `Metric.compute`)** | Do NOT take Session. The framework caller resolves session and passes explicit values (e.g., `data_spec`, `target_column`). Extension authors don't need to know about Session machinery. |
| **Private helpers (`_load_module`, `_compute_hash`)** | Take only what they need explicitly. Pure functions. |

## 7. `ProjectConfig.load(name)` step-by-step

The single point of disk-touching for project configuration. Step by step:

```
ProjectConfig.load(name, *, repo_root: Path | None = None) -> ProjectConfig

1. Validate `name`              → regex check (PACKAGE_COMPONENT_RE)
                                → raise ProjectError("invalid project name: …") if bad

2. Resolve repo_root            → if not provided, walk up from cwd looking for
                                   the workspace markers
                                → raise ProjectError if no projects/ found

3. Check project_dir exists     → projects/<name>/ must exist
                                → raise ProjectError("project 'foo' not found
                                   at projects/foo") if missing

4. Load .env (idempotent)       → reads project-local or workspace .env;
                                   does NOT overwrite existing process env

5. Read env values              → GCS_BUCKET, GCS_PREFIX, MLFLOW_TRACKING_URI,
                                   MLFLOW_ARTIFACTS_DESTINATION;
                                   missing values become empty strings (NOT an error)
                                → IF MLFLOW_TRACKING_URI is empty, emit
                                   warnings.warn("MLFLOW_TRACKING_URI not set;
                                   MLflow will silently fall back to ./mlruns")
                                   so CI/offline misconfigurations are noisy
                                   at load time, not at first MLflow call.

6. Try-import config.py         → if config.py does not exist:
                                     TASK/DATA/EVAL/RUN_CONFIG = None each
                                → if config.py exists and imports cleanly:
                                     extract attributes (each may be None or a typed object)
                                → if config.py exists but malformed (ImportError, SyntaxError):
                                     raise ProjectError wrapping the underlying cause

7. Type-check recipe fields     → for each non-None field, isinstance check
                                → raise TypeError("TASK must be a Task instance, got …")
                                   if wrong type

8. Construct and return         → frozen ProjectConfig with all the above
```

Two non-obvious behaviors worth flagging:

- **Missing env vars do not fail load.** They surface at point-of-use (e.g., `data.materialize()` raises "GCS_BUCKET not set in env"). Load remains permissive.
- **Missing `config.py` does not fail load.** Recipe fields are `None`; same point-of-use pattern via `require_*()`. The notebook-1 exploration case is supported without a flag.

## 8. CLI override layering — onto `Session`, not onto `ProjectConfig`

The rule: **`ProjectConfig` is deterministic from `config.py` + env at load time. CLI overrides go onto `Session`.**

```python
automl.use_project("foo", dry_run=True, experiment_id="alt-id")
   │
   ├─→ ProjectConfig.load("foo")        # NO CLI overrides here
   │       reads config.py + env, returns frozen ProjectConfig
   │
   └─→ Session(
           config=<that ProjectConfig>,
           dry_run=True,                 # CLI override goes here
           experiment_id="alt-id",       # CLI override goes here
       )
```

Why this split is load-bearing:

- **`ProjectConfig` stays reproducible.** Two people loading the same project in the same env get the same `ProjectConfig`. That's the property that lets MLflow / GCS reads be trustworthy.
- **`Session` is "right now in this process."** `dry_run` is per-invocation; `experiment_id` override is per-invocation. They belong here, not on a frozen config.
- **The override surface is narrow.** Only Session-level fields are CLI-overridable. Everything else (recipe contents, env values) flows in unchanged from disk.

For the rare case where ProjectConfig fields need overriding (test fixtures, what-if analyses), use `dataclasses.replace()`:

```python
cfg = ProjectConfig.load("foo")
test_cfg = dataclasses.replace(cfg, gcs_bucket="test-bucket")
session = Session(config=test_cfg)
```

No special API needed; frozen dataclass + `replace()` is sufficient.

## 9. Multi-project patterns

All three patterns are supported by the same machinery (contextvar + frozen dataclasses):

### 9.1 Sequential switching (most common)

```python
automl.use_project("foo")          # active = foo
materialize()
automl.use_project("bar")          # active = bar (foo's session is gone)
materialize()
```

### 9.2 Scoped temporary switch (tests, comparisons)

```python
automl.use_project("foo")
with automl.active_session("bar"):
    bar_lb = automl.experiment.leaderboard()      # uses bar's session
foo_lb = automl.experiment.leaderboard()          # back to foo
```

### 9.3 Explicit session objects (full control)

```python
s_foo = automl.use_project("foo")                                    # returns the Session
s_bar = Session(config=ProjectConfig.load("bar"))                    # direct construction

foo_lb = automl.experiment.leaderboard(session=s_foo)
bar_lb = automl.experiment.leaderboard(session=s_bar)
```

### 9.4 Concurrent process safety

Each subprocess (e.g. a trial subprocess spawned by the runner) gets its own contextvar. Sessions in different processes do not interact. Standard contextvar semantics.

### 9.5 Async safety — which helpers are safe in which contexts

Both the session contextvar AND the mlflow bind move together (via `_bind_mlflow_for` — §4.6), so the safety properties below apply to *both at once*. There is no scenario where session and mlflow bind can be out of sync.

| Helper | Safe in async / sibling coroutines? | Why |
|---|---|---|
| `use_project()` | **NO** — top-level / process-start only | Calls `_ACTIVE_SESSION.set(...)` + `_bind_mlflow_for(...)` without saving a Token; permanently clobbers both the session contextvar and the mlflow bind for sibling coroutines. |
| `active_session()` (context manager) | **YES** | Saves the session Token and calls `.reset(token)` on exit, then re-fires `_bind_mlflow_for(prior)`. Sibling coroutines see their own session AND their own mlflow bind. |
| `update_session()` | **NO** — same caveat as `use_project()` | Also uses `.set()` + `_bind_mlflow_for(...)` without token preservation; mutates active state for the whole process. Use only when you mean "this change applies process-wide from now on." |
| `clear_session()` | **NO** — same caveat | Top-level reset; never use inside async tasks. |
| `automl.session()` (read) | **YES** | Pure read of the contextvar; no mutation. |

The implication: `use_project()` and `update_session()` are **process-level setters**. They are fine inside a single notebook cell, a single CLI verb entry, or a single skill bootstrap — anywhere there is exactly one "current active project" for the whole process. For scoped or async switching, use `active_session()` as the context manager — it handles both session AND mlflow bind atomically via the Token pattern.

## 10. What's stored, what's cached, what's NOT stored

### Stored on `ProjectConfig` (immutable, lifetime of session)

- Identity (paths, name, package) — small strings
- Recipe (typed config objects: `Task`, `DataSpec`, `EvalSpec`, `RunConfig`)
- Env values (`GCS_BUCKET`, `GCS_PREFIX`, `MLFLOW_TRACKING_URI`, `MLFLOW_ARTIFACTS_DESTINATION`) — small strings

That's it. No data, no connections, no remote state.

### Stored on `Session` (mutable-by-replace, contextvar-held)

- Reference to `ProjectConfig`
- `dry_run: bool`
- `experiment_id: str | None` override

### NOT stored anywhere on `ProjectConfig` / `Session`

| What | Why not | Where it lives instead |
|---|---|---|
| Loaded Dataset bytes | Heavy, ephemeral; GCS is source of truth | GCS; `data.load_dataset(hash)` on demand |
| Trial records | MLflow is source of truth; staleness pain | MLflow; queried fresh per call |
| MLflow client object | Connection state is infrastructure | `mlflow/client.py` module (may reuse internally) |
| GCS client object | Same as above | `utils/io/gcs.py` |
| Credentials / tokens | Security; come from env at IO time | Process env / ADC at the moment of the IO call |
| Active dataset hash | Per-experiment state; lives in MLflow's experiment-overview run | Pulled on demand via `mlflow.experiment.active_dataset(...)` |
| Trial count, current best metric | Remote state; staleness pain | MLflow query each time |

### Caching rules

- **Module-level caching of `config.py` import is allowed.** Python's module cache handles this naturally; we don't fight it. Safe because `ProjectConfig` is frozen.
- **No caching on `Session`.** Session is small and explicit; nothing worth caching on it.
- **No "smart caching" of MLflow query results.** Reads go to MLflow each time. If performance ever requires caching, that's a separate decision behind a clear cache abstraction in `mlflow/queries.py` — not on `Session`.

### Hygiene rules

1. `ProjectConfig` is `@dataclass(frozen=True)`. Changes happen via `dataclasses.replace()`.
2. `Session` is also frozen; field changes go through `replace()`.
3. `ProjectConfig.load` is the ONLY place that touches disk for project configuration.
4. Neither object holds references to remote resources (bucket NAMES yes; client OBJECTS no).
5. No transitive lazy properties. Everything is computed at load time.

## 11. Validation surface

Three layers, each appropriate to a different question:

| Question | Use |
|---|---|
| "Tell me everything that's wrong with my project" | `automl validate project --project <name>` (CLI verb) — runs every registered check, returns a `ValidationReport` with all `Issue`s. |
| "Programmatically check if my config is ready" | `session.config.is_complete()` and `session.config.missing_fields()`. Lightweight; doesn't run heavy checks. |
| "Use this thing — fail clearly if a required field isn't set" | `session.config.require_task()` etc. Raises with file path + field name. |

`automl validate project` continues to be the explicit verb; it's the validation framework's job (see structural spec §9.2), not this sub-spec's. What this sub-spec settles is that `ProjectConfig` itself doesn't run validation at load time beyond type-checking — validation is a separate, explicit action.

## 12. Migration from current `ProjectContext`

| Current `ProjectContext` (in `automl/core/project_context.py`) | New shape |
|---|---|
| `repo_root`, `project_name`, `project_dir`, `project_package` | `ProjectConfig` identity fields |
| `config_path`, `instructions_path` (derived) | `ProjectConfig` identity fields |
| `task`, `task_object`, `data_spec`, `evaluation_spec`, `run_config` (lazy properties) | `ProjectConfig` recipe fields — eagerly loaded, not lazy |
| `primary_metric`, `target_column`, `raw_target_column`, `per_trial_seconds`, `models` (derived properties) | `ProjectConfig` derived properties; methods delegate to recipe fields |
| `gcs_bucket`, `gcs_prefix`, `mlflow_tracking_uri` (env-backed properties) | `ProjectConfig` env fields — eagerly read; plus new `mlflow_artifacts_destination` field added by sub-spec 03 carry-back (§3.1 / §7 step 5) |
| `dry_run` (constructor flag) | `Session.dry_run` (moves to active-state layer) |
| `import_module()`, `_project_module()` | `project/_import.py` (private helper for `ProjectConfig.load`) |
| `validate_config()` (eager validation method) | Folded into `ProjectConfig.load`'s type-check step (#7 in §7) |
| `require_config=True/False` constructor flag | REMOVED — None-semantics handles partial configs automatically |
| `_ACTIVE_PROJECT` contextvar of `ProjectContext` | `_ACTIVE_SESSION` contextvar of `Session` |
| `clear_active_project()`, `set_active_project()`, `active_project()` (cm) | `clear_session()`, `use_project()`, `active_session()` (cm) — renamed for clarity |
| (no equivalent) | **`Session.experiment_id` override + `Session.active_experiment_id` computed property** — new |
| legacy `route_namespace` (a seam param, always `""`, never wired to a source) | **`Session.namespace`** — promoted to a real, wired Session override fed by the top-level `--namespace` flag / env; full-universe isolation (MLflow + GCS + local trial dirs). Defaults `""` = real. |
| (no equivalent) | **`ProjectConfig.is_complete()`, `.missing_fields()`, `.require_*()`** — new |
| (no equivalent) | **`_bind_mlflow_for(session)` helper (§4.6)** — propagates session state to mlflow framework. Called by `use_project`, `update_session`, `active_session` in lock-step. |

Net: nothing is dropped; behavior splits cleanly between immutable (`ProjectConfig`) and active-state (`Session`). The lazy-property pattern is replaced by eager-load + frozen state. The mlflow seam now binds atomically with every session transition.

## 13. What this sub-spec defers

- **The `automl project init` CLI verb's exact behavior.** What template content does it write? What checks does it run? Handled by the structural spec's CLI verb sketch (§11.1) and the implementation plan.
- **Project-local subclass resolution.** Projects override orchestration by subclassing `DataPipeline` and pointing `DataSpec.pipeline_cls` at the subclass (resolved per sub-spec 05 Q2 + Q7 — `DataSpec` carries `pipeline_cls`; `_build_pipeline(session)` instantiates `session.config.require_data_spec().pipeline_cls`). The subclass lives in a normal project module (e.g. `projects/<name>/pipelines.py`) imported by `config.py`; no special `ProjectConfig.load` wiring needed beyond importing the config module (which `ProjectConfig.load` already does).
- **`project → eval` edge for the eval spec — RESOLVED here, no new surface.** Sub-spec 07 carried a question about whether the project domain should re-expose eval-spec loading. It does not need to: the legacy lazy `evaluation_spec` property is already replaced by the **eager `ProjectConfig.eval_spec` field** (§3.1 / §12), and `primary_metric` is a derived property reading it (§12). `load_evaluation_spec` keeps living in `eval/_load.py` (eval owns it); `ProjectConfig.load()` invokes it via a **late import** (the same acyclic pattern `_bind_mlflow_for` uses), so the `project → eval` edge is load-time only and creates no cycle. Callers read `session.config.eval_spec` / `session.config.primary_metric` — there is no separate `evaluation_spec` re-export property.

## 14. Concrete examples

### Example 1: Notebook-1 exploration

```python
import automl

automl.use_project("payment_routing")          # works even with empty config.py

s = automl.session()
print(s.config.project_name)                   # "payment_routing" — always available
print(s.config.gcs_bucket)                     # from env

print(s.config.is_complete())                  # False
print(s.config.missing_fields())               # ['TASK', 'DATA', 'EVAL', 'RUN_CONFIG']

automl.experiment.leaderboard()                # raises ProjectError pointing at RUN_CONFIG
```

### Example 2: Fully configured runtime

```python
import automl

# experiment_id passed at session-start; could also come from --experiment CLI flag
s = automl.use_project("payment_routing", experiment_id="baseline-sweep")

automl.experiment.create("baseline-sweep")
ds = automl.data.materialize()
trial = automl.trial.create(slug="lgbm_baseline", strategy="baseline")
result = automl.runner.run_trial(trial.path)
```

Note that `trial` operations live under `automl.trial.*` — Trial was promoted from a sub-concept inside Experiment to a **top-level domain** during sub-spec 09 (structural spec §8.7). Execution lives in `runner` (`runner.run_trial`).

### Example 3: CLI overrides

```bash
automl run --project payment_routing --dry-run --experiment alt-id
```

The CLI verb's main():

```python
def run_main(args):
    automl.use_project(
        args.project,
        dry_run=args.dry_run,
        experiment_id=args.experiment,
    )
    ...  # rest of run logic uses automl.session()
```

### Example 4: Test fixture (no disk, no env)

```python
import pytest
from automl.project import ProjectConfig, Session
from dataclasses import replace

@pytest.fixture
def test_session(tmp_path):
    cfg = ProjectConfig(
        project_name="test_project",
        repo_root=tmp_path,
        project_dir=tmp_path / "projects" / "test_project",
        project_package="projects.test_project",
        config_path=tmp_path / "projects" / "test_project" / "config.py",
        instructions_path=tmp_path / "projects" / "test_project" / "PROJECT_INSTRUCTIONS.md",
        task=BinaryClassification(target="y"),
        data_spec=DataSpec(...),
        eval_spec=EvalSpec(primary=Auc()),
        run_config=RunConfig(experiment_id="test-exp", ...),
        gcs_bucket="test-bucket",
        gcs_prefix="test/",
        mlflow_tracking_uri="file:///tmp/mlflow",
    )
    return Session(config=cfg)
```

### Example 5: Multi-project comparison tool

```python
def compare_projects(name_a: str, name_b: str) -> ComparisonResult:
    with automl.active_session(name_a) as s_a:
        lb_a = automl.experiment.leaderboard()
    with automl.active_session(name_b) as s_b:
        lb_b = automl.experiment.leaderboard()
    return ComparisonResult(a=lb_a, b=lb_b)
```

---

## Sub-spec status

This sub-spec is complete. It resolves every open question §13.5 of the structural spec listed except the two items explicitly noted as "defers" in §13 above. Implementation can proceed against this contract; the next sub-spec in priority order is **MLflow seam interfaces** (structural spec §15.1 Priority 2).
