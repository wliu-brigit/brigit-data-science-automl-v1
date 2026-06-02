# Sub-spec 06 — Model Domain

**Status:** APPROVED 2026-05-24 (design interview + three-agent review + fixes applied)
**Started:** 2026-05-24
**Parent:** `00-structural-design.md` §8.3 (`model/`)
**Anchor:** `BaseModel` (Tier 3) — but this sub-spec elevates **preprocessing**
to a co-equal, formally-contracted part of the model domain.

---

## Framing

The model domain has two parts: **preprocessing → estimator**. Today the
library formally contracts only the estimator side; preprocessing is an opaque,
free-flow `self.preprocessor` attribute. This sub-spec gives preprocessing a
first-class contract so that **a project can declare mandatory preprocessing**
(e.g. a Weight-of-Evidence encoder on a categorical risk column) that every
trial model — human- or AI-authored — **must** use, enforced in both the coder
prompt and validation.

## Current state (grounding)

`automl_legacy/core/base_model.py` — `BaseModel(mlflow.pyfunc.PythonModel, ABC)`:
- Post-fit contract attrs: `feature_registry`, `preprocessor`, `model`, `name`.
- Abstract: `fit(df_train, registry, seed)`, `transform(df) -> np.ndarray`,
  `_predict(X) -> np.ndarray`.
- Base-owned: `predict(context, model_input, params)` (PyFunc),
  `predict_transformed(X)`. Optional `feature_importances()`, `training_report()`.
- **`preprocessor` is opaque and free-flow.** The model builds whatever sklearn
  object it wants inside `fit()`; nothing inspects its contents.

`automl_legacy/validate/builtin/model_checks.py` — shape-only:
`model.subclass_basemodel`, `model.fit_succeeds`, `model.post_fit_attrs_set`
(the four attrs non-None). **No check asserts a specific transform was used.**

There is **zero project-side preprocessing contribution in the legacy package**.
`references/setup/model-contract.md` line 13: *"Projects do not carry a
project-local model contract file."*

### Prior art (read-only `automl/` snapshot — proves the use case)

`automl/project-template/src/predefined_transformers.py`:
- `predefined_transformer_entries() -> list[(name, transformer, input_cols)]`,
  a project-extended factory; default empty.
- Models spliced `*predefined_transformer_entries()` into a `ColumnTransformer`
  (v0.1 plan line 3499) so all trials shared the same domain preprocessing.
- Coder was told to **read** the file (`implement.md` step 2).
- **Gaps:** no formal interface (bare tuples + docstring), **no enforcement**
  (read-only prompt mention), no model-side awareness/hook.

Use case proven by the prior art: *"a WoE encoder for `BANKINSTITUTION` in
subscription_skip"* — a lookup-table encoder mapping a categorical to a risk
weight. The splice-into-`ColumnTransformer` pattern worked; what it lacked was
a contract + a gate.

---

## Decisions

### Q1 — Where does mandatory preprocessing live / how enforced?

**DECIDED: (C) — contract-level extension point, formally owned by the model domain.**

Not (A) framework-injected (the author must *actively* use it — auto-injection
would mean they don't "have to"), not (B) pure project-layer convention (too
free-flow; no standardized interface to build on). Instead: the model domain
defines a **formal contract interface** for required preprocessing; `model.py`
*knows the extension point exists* and loads the project's declared
requirements through it; validation enforces that they were used.

Rationale (user, 2026-05-24): *"model only has two parts, preprocessing and a
model. Right now we focus really on the model part, but this deserves to
formally support it as a contract — tier-one support."* The control plane (the
contract) stays in the model domain / `model.py`, not scattered as bridge code
on the project side.

This is the **real consumer** that re-opens the `"model"` project-check target
that sub-spec 04 deferred (`feedback_extension_points_follow_demand`: *"no real
consumer, can re-add"*). Now there is one.

### Q2 — the contract interface's unit + shape

**DECIDED: (A) — a typed `RequiredTransformer` spec.** Formalizes the prior
art's `(name, transformer, input_cols)` tuple into a frozen dataclass defined in
core (`model/preprocessing.py`):

- `name: str` — stable identifier; the key the validation gate matches on.
- `transformer` — any sklearn-compatible object with `fit`/`transform` (carries
  its fitted state, e.g. the WOE lookup table, into `self.preprocessor` via the
  existing cloudpickle/single-Docker contract — no new packaging work).
- `input_cols: list[str]` — the columns it operates on.

Column-scoped + sklearn-native: each entry splices straight into a
`ColumnTransformer`. Not (B) richer config object / (C) project-subclassed ABC —
both add configurability with no real consumer yet
(`feedback_extension_points_follow_demand`); the dataclass grows additively if a
real need (e.g. sequential ordering) appears. (`RequiredTransformer` MAY itself
be a tiny protocol/ABC so *project transformer authors* build against a standard
`fit`/`transform`/`name` base — that's a project-author-facing contract, not a
trial-model-facing one.)

### Q3 — enforcement architecture

**DECIDED: (B) — inspection gate, framework-owned.** Three pieces:

1. **Declaration — project's responsibility, expressed as data.** The list is
   declared in `config.py` as `REQUIRED_TRANSFORMERS: list[RequiredTransformer]`
   (default empty), importing the transformer *classes* from
   `projects/<name>/model/preprocessing.py`. (See §Q5 for the full
   location split — config.py declares; `model/preprocessing.py` defines the
   classes.) The project owns *what* is required; it writes **zero enforcement
   code**.

2. **Consumption — a concrete hook on BaseModel.**
   `BaseModel.required_transformer_entries(session=None) -> list[tuple]`
   (base-owned). Returns ColumnTransformer-ready `(name, clone, cols)` entries —
   `sklearn.clone`d from the active project's declaration. The author splices
   `*self.required_transformer_entries()` into their `ColumnTransformer`. The
   entries carry the **canonical declared `name` + columns**, so a model that
   uses the hook gets exact gate-matching for free. **Single method** — the gate
   reads requirements from `session.config` and the proposer uses
   `describe_required_transformers`, so the model needs no second
   `required_transformers()` accessor. **Concrete, not abstract:** an abstract
   method would add boilerplate without adding a guarantee.
   **Session resolution (S1):** `session` defaults to `None` and resolves the
   *ambient* session via `automl.session()` (sub-spec 01 convention). The runner
   runs a trial within the active project, so the contextvar is set at fit time.
   Standalone unit tests must wrap the call in `automl.active_session()`.
   The hook is **not** an integrity *guarantee* — it makes the correct instance
   the *easy path*; enforcement is the gate (piece 3).

3. **Enforcement — a framework-owned check inside `validate.model`.**
   Runs automatically whenever the project declares requirements, **only after a
   successful fit** (`instance is not None and error is None`, parallel to
   `check_post_fit_attrs_set`; a fit failure is already reported by
   `check_fit_succeeds`). Introspects the *fitted* `self.preprocessor` via
   `ColumnTransformer.transformers_` (the fitted `(name, transformer, columns)`
   triples — `named_transformers_` carries no column info, so the `input_cols`
   check must read `transformers_`). For each required entry it asserts: declared
   `name` present + `isinstance(type(declared))` + `input_cols ⊆ entry columns`
   (named columns only — positional integer indexing is disallowed when
   requirements exist, so name+column matching is unambiguous). Structural
   integrity (type + columns), **not** behavioral (the WOE math is the
   transformer's own project-side unit test, not the AutoML gate's job).
   **Session resolution (S1):** the gate reads the ambient session via
   `automl.session()` — sub-spec 04's `validate.model(cls, *, df, registry)`
   signature is unchanged (no `session` param added); see carry-back to 04.
   **Trust boundary (C7):** the gate proves the required transformer is present +
   fitted in `self.preprocessor`; it does not prove the author's `transform()`
   routes through it — the same author-trust boundary as the rest of the
   `BaseModel` contract. No extra guard (`feedback_no_redundant_guards`).

Rejected: (A) construction guarantee (base auto-splices — removes the author
agency Q1 established); (D) project-owned check (every project re-implements the
same boilerplate gate).

**This is the targeted re-add of sub-spec 04's deferred `"model"` check** — but
narrower than a generic `PROJECT_CHECKS` escape hatch: the project declares
*data*, the framework owns the *enforcement*. We add exactly the gate the real
consumer needs, framework-owned, not an arbitrary project-check seam.

#### Empty-default behavior (the load-bearing graceful case)

- Default `REQUIRED_TRANSFORMERS = []` (or file/symbol absent → empty). An
  unfilled requirement is empty, not an error (None-semantics, per sub-spec 01).
- `required_transformer_entries()` returns `[]`; the splice adds nothing; a model
  may always write `ColumnTransformer([*self.required_transformer_entries(), ...])`
  with no `if project has requirements` branching.
- The gate has nothing to enforce → no-op. A model that never references the
  hook still passes.
- **Today's free-flow models + the Home Credit harness are unaffected.** The
  framework path is always present and always runs, but is inert until a project
  opts in by declaring. Zero cost when unused.

### Q4 — structural constraint on `self.preprocessor` + how the gate finds the step

**DECIDED: (A) — `self.preprocessor` must BE a top-level `ColumnTransformer` when
requirements exist.** If `REQUIRED_TRANSFORMERS` is non-empty, `self.preprocessor`
must *be* a `ColumnTransformer` (S2: **not** a `Pipeline` that wraps one) whose
top-level entries include each required `name`. The gate inspects the fitted
`ColumnTransformer.transformers_` triples — one level, no tree-walking, no
column-provenance reasoning.

**Where downstream steps go (S2):** any post-column-split step the author wants
(feature selection, dimensionality reduction, etc.) lives in `self.model` — the
estimator may itself be a `Pipeline([..., estimator])`. It does **not** go by
wrapping the `ColumnTransformer` in an outer `Pipeline`, which would hide the CT
from the one-level gate and falsely fail. This keeps the gate a clean one-level
check and removes the "mandate CT" vs "allow nesting" ambiguity the review
flagged.

Constraint is opt-in (only when the project declares requirements); free-flow
projects and a plain `Pipeline` preprocessor (e.g. `logistic_baseline`) are
untouched.

Not (B) recursive introspection (validator complexity + ambiguous column
provenance under nesting) or (C) identity tracking (couples hook to validation
state; more machinery than needed). Trigger to revisit (B): a real model needs a
preprocessing step (e.g. scaling) *before* the column split — unusual, since
domain encoders like WOE consume raw categoricals.

### Q5 — declaration mechanism + file placement

**DECIDED:** the same single-level pattern `config.py` already uses for
`DATA`/`EVAL` — *custom code in a sibling module, declared in `config.py`*. Not
a pointer/discovery indirection (the "two-level contract" worry).

- **Core type:** `automl/model/preprocessing.py` → `RequiredTransformer`
  (frozen dataclass) + `SklearnTransformer` protocol.
- **Project transformer classes:** `projects/<name>/model/preprocessing.py` —
  mirrors core path per `feedback_project_mirrors_core` (the "required"
  semantics live in the *symbol* `RequiredTransformer`, so the filename mirrors
  core rather than inventing `required_transformers.py`). **Shipped as an empty
  stub in the project template** — the "override from here" file. Must be an
  importable module (NOT inline in `model.py`) so cloudpickle serializes the
  fitted transformer *by reference*; the bundled `projects/` tree
  (`stage_code_bundle` → `code_paths`) makes that import resolve at serving.
- **Declaration:** `config.py` exports `REQUIRED_TRANSFORMERS: list[RequiredTransformer]`
  (imports the classes; empty/omitted by default). One declarative surface,
  exactly like `DATA = DataSpec(...)`.

User-facing mental model: *write your encoder in `model/preprocessing.py`, list
it in `config.py`.* Hook (§Q3.2) + gate (§Q3.3) are automatic.

**Serialization reliability (verified, not assumed):** `mlflow/code_bundle.py`
copies the whole `automl/` + `projects/` trees into `code_bundle/`;
`mlflow.pyfunc.log_model(code_paths=[…/automl, …/projects, …/trial_model])`
(`runner/_execute.py:869-879`) puts them on `sys.path` at load time. A custom
transformer in an importable project module round-trips for free. **Guideline:**
define transformer classes in the project module, never inline in `model.py`.

### Q6 — integrity depth

**DECIDED: (i) — type + columns floor; do NOT pin hyperparameters.** The gate
checks `name` present + `isinstance(type(declared))` + `input_cols ⊆ entry cols`.
It does not assert `get_params()` equality. Rationale: the hook hands out the
canonical clone, so the easy path is already correct; and a project may
legitimately want trials to tune the mandated step. Pin later only if a real
project needs immutability (`feedback_extension_points_follow_demand`).

---

### Q7 — coder-prompt surfacing

**DECIDED: (A) — model domain exposes a description helper; the surface layer
renders it; the vehicle is the `TrialProposal` handoff.**

Surfacing chain (all four touchpoints derive from the one `config.py`
declaration → zero drift, requirement tied to provenance):
**proposer detects → writes onto `TrialProposal` → coder reads its handoff doc →
model gate enforces.**

Layering:
- **Sub-spec 06 (here) commits only to:** `automl/model/preprocessing.py::
  describe_required_transformers(session) -> list[dict]` (name, type, import
  path, columns). Generated from `session.config.required_transformers`.
- **Sub-spec 11 (owns `Proposal`, `propose/schema.py` →
  `agent/proposal.py`)** adds a dedicated optional field
  `required_preprocessing` (NOT overloaded into `constraints` — it's
  auto-populated from config, not proposer judgment, so it stays structurally
  distinct + machine-readable). Persisted in `trial_proposal.json`.
- **Plugin layer:** proposer agent populates the field via the helper; coder
  agent reads it.

**Single enforcement point** (per `feedback_no_redundant_guards`): the
`validate proposal` check (sub-spec 04 `agent/checks.py::proposal_schema`)
just *allows* `required_preprocessing`; it does NOT re-police the requirement.
The model gate (§Q3.3) is the only enforcement — one invariant, one gate. The
proposal field is informational/handoff + provenance only.

Not (B) static `PROJECT_INSTRUCTIONS.md` prose (drifts from config — the prior
art's fragility) or (C) both.

## Carry-backs

- **→ sub-spec 01 §3.1 (`ProjectConfig`):** add field
  `required_transformers: list[RequiredTransformer]`, loaded from `config.py`'s
  `REQUIRED_TRANSFORMERS` (empty default / None-semantics). The `validate.model`
  gate reads `session.config.required_transformers`.
- **→ structural §8.3 + §7 (`model/` folder):** add `model/preprocessing.py`
  (RequiredTransformer type + protocol + `describe_required_transformers`)
  alongside `base.py`, `packaging.py`, `checks.py` — in **both** the §8.3
  export/ownership text **and** the §7 folder tree (which currently lists only
  base/packaging/checks). Tier 2 exports gain `RequiredTransformer` +
  `describe_required_transformers`.
- **→ sub-spec 04 (`validate.model`):** the framework gate
  `check_required_transformers(instance, *, session=None)` lives in
  `model/checks.py` and is invoked by the `validate.model` orchestrator in its
  `error is None` branch. This is the targeted re-add of the deferred `"model"`
  check — framework-owned (project declares *data*), NOT a generic
  `PROJECT_CHECKS` escape hatch. **S1:** the gate resolves the ambient session
  via `automl.session()`; sub-spec 04's `validate.model(cls, *, df, registry)`
  signature is **unchanged** (no `session` parameter added — avoids reopening 04).

## Mechanical migration map (`model/` domain)

The substantive design is Q1–Q7. The rest is reshaping current code into the
new structure. The model domain's outbound deps are **`errors` only** (§8.3) —
this boundary drove two corrections to the migration checklist.

- **`model/base.py`** — today's `core/base_model.py::BaseModel`, unchanged
  except one new method `required_transformer_entries(session=None) -> list[tuple]`
  (returns ColumnTransformer-ready `(name, clone, cols)` for splicing; `[]` when
  no requirements). Single hook — the gate reads requirements from
  `session.config` and the proposer uses `describe_required_transformers`, so
  the model needs no second `required_transformers()` accessor.
- **`model/preprocessing.py`** (new) — `RequiredTransformer` (frozen dataclass),
  `SklearnTransformer` (protocol), `describe_required_transformers(session) ->
  list[dict]`.
- **`model/packaging.py`** — `save_model(model, path)`: the cloudpickle dump
  extracted from `runner/_execute.py:710` into one documented place beside the
  `BaseModel` it serializes. Pure cloudpickle + `errors`; **no mlflow**. A
  path-based `load_model(path)` is **deferred** — nothing loads from a raw path
  today (loads go through mlflow pyfunc); add on demand
  (`feedback_extension_points_follow_demand`).
- **`model/checks.py`** — `REQUIRED_POST_FIT_ATTRS` +
  `check_subclass_basemodel` / `check_fit_succeeds` / `check_post_fit_attrs_set`
  (sub-spec 04 signatures; `ModelProbe` family already dropped there) **+ new
  `check_required_transformers(instance, *, session)`** (the §Q3.3 gate).
- **`model/__init__.py` Tier 2 exports** — `BaseModel`, `RequiredTransformer`,
  `describe_required_transformers`.

### Corrections to migration-checklist (functions that do NOT belong in `model/`)

1. **`trial/packaging.py::package_model`** — notebook-class → `model.py` *source
   extraction* (a trial-authoring helper), not serialization. Home:
   `trial/packaging.py` (sub-spec 10), NOT `model/packaging.py`.
2. **`inspect/views.py::load_model(run_id)`** — MLflow PyFunc load by run id +
   project-context resolution; depends on the mlflow seam, which the model
   domain may not import (§8.3). Home: `trial/show.py` (sub-spec 10), NOT
   `model/packaging.py`.

## Carry-forwards (to later sub-specs)

- **→ sub-spec 11 (Agent / `Proposal`):** add optional field
  `required_preprocessing` to Proposal v2; proposer populates it from
  `model.describe_required_transformers(session)`; coder reads it; persisted in
  `trial_proposal.json`. The `proposal_schema` check *allows* the field but does
  NOT enforce it (single gate = model gate).
- **→ plugin layer (agents/skills):** proposer agent populates
  `required_preprocessing`; coder agent (`agents/automl-coder.md`) reads it and
  splices `*self.required_transformer_entries()`. Update
  `references/setup/model-contract.md` to document the required-preprocessing
  contract + the `projects/<name>/model/preprocessing.py` stub.

## Implementation-time deliverables

- **Project template:** ship `projects/<name>/model/preprocessing.py` as an
  empty stub (commented WOE example) + a commented `REQUIRED_TRANSFORMERS = []`
  block in the template `config.py`.
- **Home Credit example (testable end-to-end path):** ship a real `WOEEncoder`
  in `projects/example_homecredit/model/preprocessing.py` + declare it in that
  project's `config.py` — candidate column **`ORGANIZATION_TYPE`** (or
  `OCCUPATION_TYPE`): high-cardinality categorical risk driver, domain-authentic
  for credit-default, exercises the full hook→splice→gate path so users can see
  it work (and see validation fail when omitted).

## Implementation notes (from three-agent review)

- **Sequencing prerequisite:** the gate, the `required_transformer_entries()`
  hook, and `describe_required_transformers()` all read `session.config`
  (`Session`/`ProjectConfig`) — a **sub-spec 01 deliverable**. Today's code has
  `ProjectContext`, not `Session`. So sub-spec 01 must land before sub-spec 06's
  surface can be implemented. Record in the implementation plan's ordering.
- **Name ban:** `tests/contracts/test_no_removed_src_layout.py` bans the literal
  `predefined_transformers` anywhere in active surfaces. The new naming
  (`RequiredTransformer`, `model/preprocessing.py`) already complies — do not
  reintroduce the old name in stubs/comments/templates.
- **Baseline interaction:** once `example_homecredit` declares a requirement, the
  existing `logistic_baseline` trial (plain `Pipeline`, no WOE) will **fail the
  new gate**. That failure IS the "see validation fail when omitted"
  demonstration — expected, not a regression. (A compliant baseline that splices
  the hook can be added alongside it.)
- **`package_model` migration scope:** `trial/packaging.py` is ~109 lines —
  `package_model` + private source-extraction helpers (`_model_class_source`,
  `_source_defines_class`, `_class_source_from_method_code`, `_unwrap_function`,
  `_extract_class_block`). All move together to `trial/packaging.py` (sub-spec 10).
