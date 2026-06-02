# Sub-spec 10 — Trial domain

**STATUS: APPROVED 2026-05-26.** (Design interview Q1–Q10 + three-agent review +
fixes applied. Carry-backs to 00/02/09 + living docs applied at closeout.)

**Date:** 2026-05-25
**Sub-spec:** 10 of the AutoML refactor (see `README.md` + `00-structural-design.md`).

**Scope.** Settle the interface + internal shape of the **`trial/`** domain — the Trial
noun, promoted to a top-level peer of `experiment/` during the sub-spec 09 decomposition.
It owns: trial lifecycle (`create` / `fork` / `promote`), the trial-scope cleanup wrapper,
`package_model` (notebook→source, the 06 correction), the per-trial read paths
(`show_trial` / `load_model`), and the trial type system — the seam-returned read
contracts (`TrialSummary` / `TrialDetails` / `TrialStatus` / `ParentExperimentRef`) plus
the write-side schemas (`TrialMetadata` / `SeedSelection` / `TimingReport` /
`TrialManifest`). It does **not** cover trial *execution* (→ `runner/`, sub-spec 08), the
cross-trial views (→ `experiment/`, sub-spec 09), or the Proposal contract + agent loop
(→ `agent/`, sub-spec 11).

**Reshape, not invent.** The functionality already exists in
`automl_legacy/trial/{creation,fork,promotion,cleanup,packaging,run}.py`,
`automl_legacy/inspect/views.py` (`show_trial` / `load_model`),
`automl_legacy/loop_context/queries.py` (`show_trial`), and
`automl_legacy/mlflow/store.py` (`_run_to_trial_summary` /
`get_trial_summaries`). 10 puts each piece in its canonical home, threads it through the
locked seam (02) and sibling domains (01 / 03 / 06 / 07 / 08 / 09), and resolves the trial
type system that 02 named but left open.

---

## 1. Context — what `trial/` owns, and what it inherits

| Current location | What it does | New home |
|---|---|---|
| `trial/creation.py::create` | build the trial draft folder + metadata + seed | `trial/create.py::create` (08 Q1 path; session conv.) |
| `trial/creation.py::_SeedSelection` / `_resolve_seed*` | pick + fetch a warm-start seed | `SeedSelection` (public, `trial/metadata.py`) + private resolver in `create.py` (re-pointed to seam) |
| `trial/creation.py::_next_trial_number_from_mlflow` | exec-time trial number | **seam** `mlflow.experiment.next_trial_number` (08 Q2) |
| `trial/fork.py::fork` | human-authored seeded trial (no run) | `trial/fork.py::fork` |
| `trial/promotion.py::promote` | human trial from a file → run it | `trial/promote.py::promote` (calls `runner.run_trial`) |
| `trial/cleanup.py::cleanup` | delete one trial | `trial/cleanup.py::delete` (thin wrapper → `project.cleanup`, 03 §9) |
| `trial/packaging.py::package_model` | notebook class → `model.py` source | `trial/packaging.py::package_model` (06 correction) |
| `trial/run.py::run` | 1-line `run_trial` delegate | **DROP** — callers use `runner.run_trial` directly |
| `inspect/views.py::show_trial` | enriched single-trial read | `trial/show.py::show_trial` → `TrialDetails` |
| `inspect/views.py::load_model` | pyfunc load by run_id | `trial/show.py::load_model` (06 correction — needs seam) |
| `inspect/views.py::load_data_snapshot` | load a run's Dataset | **LEAVES trial** → `data.load_dataset_by_trial` (Appendix A) |
| `loop_context/queries.py::show_trial` | raw 5-key run dict | superseded by `mlflow.trial.get_details` + `trial.show_trial` |
| `mlflow/store.py::_run_to_trial_summary` | run → 19-key summary dict | seam `mlflow.experiment.*` returns typed `TrialSummary` (type owned here) |

**Inherited (settled by prior specs — pulled up, not re-derived):**
- **08 Q1** — `trial.create` builds the **mode-segregated** folder
  (`projects/<project>/experiments/[dry_run/]<project>/<experiment_id>/<slug>/`) using a
  **path helper owned by `runner/`**; the runner *verifies* the path as a universe-isolation
  guard. `trial/` imports the path helper from `runner/`.
- **08 Q2** — the exec-time trial number comes from `mlflow.experiment.next_trial_number`
  (relocated to the seam to break the backward `runner→trial` import). `trial_id =
  <number>_<slug>` is an **exec-time** identity; the draft carries only `slug`.
- **08 trial↔runner edge** — acyclic via the type-vs-function split: `runner/` imports
  trial *types* (`TrialResult` is runner's own; trial types it needs are read-only); `trial/`
  imports runner's path helper + `run_trial` (a *function*). No cycle.
- **03 §9** — trial-delete is a thin wrapper over the `project/cleanup.py` cascade engine;
  resolves the run's universe via `mlflow.trial.get_parent_experiment` and checks it against
  the session before acting.
- **06 corrections** — `package_model` (notebook→source authoring, stdlib-only) and
  `load_model` (pyfunc load by run_id) both land in `trial/` because the model domain's
  outbound deps are `errors` only and cannot reach the seam.
- **02 §6.3** — the `mlflow.trial.*` surface (`active`/`start`/`end`, `log_*`,
  `get_details`/`get_metrics`/`list_artifacts`/`get_parent_experiment`); the trial type
  *homes* (`TrialSummary`/`TrialDetails`/`ParentExperimentRef` → `trial/types.py`;
  `TimingReport`/`TrialManifest` → `trial/metadata.py`). 02 locked `TrialSummary`'s fields
  and `ParentExperimentRef`; it left `TrialDetails`/`TimingReport`/`TrialManifest` field
  lists open — 10 settles them.
- **01** — the session convention: every IO-touching Tier-2 function takes
  `session: Session | None = None`, resolved `session if session is not None else
  automl.session()`.

---

## 2. Locked invariants this domain inherits (not re-litigated)

- **Seam-only (00 §9.1 / §13.4).** `trial/` never `import mlflow`. Every read/write goes
  through `automl.mlflow.<noun>` and comes back as a typed domain object.
- **Session convention (01).** `session: Session | None = None`; no `project` /
  `project_root` / `dry_run` parameters (`feedback_dry_run_is_a_container`).
- **Cleanup cascade lives in `project/` (03).** `trial/cleanup.py` is a thin wrapper.
- **Schema rule (02 §8).** Persisted/serialized typed schemas are frozen dataclasses with
  `schema_version: int = 1` + `from_dict` (strips unknown keys); additive-only; clean cut,
  no back-compat for old tag/field values.
- **Derive, don't store (07/09).** Derivable URIs (model/manifest/contract/predictions) and
  MLflow URLs are computed at the boundary, not stored on types.

---

## 3. Q1 — The read-side type model: `show_trial` returns a typed `TrialDetails`

**Current state.** Three divergent read shapes exist: `loop_context.show_trial` (5-key raw
dict), `inspect.show_trial` (that dict + an `eval` block read from `eval/manifest.json` +
per-label `report.json` + a derived `mlflow_url`), and `_run_to_trial_summary` (a 19-key
curated dict). Each re-parses MLflow runs independently — the drift the refactor fights.
02 named `mlflow.trial.get_details(run_id) -> TrialDetails` but left `TrialDetails`'s fields
open; 09 deferred `ComparisonResult.runs`'s element type to "follows `trial.show_trial`".

**DECISION: one typed `TrialDetails`; `show_trial` returns it.**

- `mlflow.trial.get_details(run_id) -> TrialDetails` (seam) maps a run's state — `params` /
  `metrics` / `tags` (raw) + artifact **paths** — into `TrialDetails`, with
  `evaluations=None` (cheap; no artifact-content downloads).
- `trial.show_trial(run_id, *, session=None) -> TrialDetails` (domain verb) = `get_details`
  **plus** loading the eval artifacts into a populated `TrialDetails.evaluations` (the one
  extra cost, paid only for a single-trial deep read).
- **`evaluations: list[EvalResult] | None`** disambiguates the two callers (review finding):
  `None` = "not loaded" (the cheap `get_details` form); `[]` = "loaded, no evaluations"
  (the `show_trial` form on a trial with no eval block). A consumer that needs evaluations
  must call `show_trial`; `None` is the explicit "you used the cheap read" signal.
- The `mlflow_url` is **derived at the CLI/view boundary** via the seam helper
  `mlflow.client.run_url` (09 carry-back to 02) — **not** a field on `TrialDetails`.
- `experiment/views/compare.py` composes `show_trial` per run; **`ComparisonResult.runs:
  list[TrialDetails]`** (resolves 09's deferred element type).

**Rejected:** keeping `show_trial` a raw dict (it would be the one untyped link in an
otherwise-typed read chain — 09 typed `LeaderboardData`/`ComparisonResult` precisely because
`compare`/`leaderboard` have programmatic consumers); a second `TrialView` type wrapping
`TrialDetails` (the near-duplicate drift 09 fought when it collapsed
`Experiment`/`ExperimentOverview`).

---

## 4. Q2 — `TrialSummary` and `TrialDetails` are independent; shared *derivation*, not shared *type*

**Current state.** The "row" curation (`_run_to_trial_summary`) and the "details" raw dump
(`show_trial`) are two independent code paths reading the same run — which is *why* they've
drifted. 02 locked `TrialSummary` (slim, cheap row). Q1 establishes `TrialDetails` (deep
single read).

**DECISION: two independent types; no type composition between them.**

- `TrialSummary` = the cheap **row** (02-locked + Q3 additions): built purely from tags/metrics
  already in hand, **zero per-trial artifact downloads**. Returned by `list_trials` /
  `top_n_by_metric` / `search_trials`; `leaderboard`/`recent_failures` consume it.
- `TrialDetails` = the deep **single read**: `run_id`, `status`, raw `params` / `metrics` /
  `tags`, `artifacts: list[ArtifactRef]`, `evaluations: list[EvalResult]`. It can afford the
  eval-artifact downloads `TrialSummary` deliberately avoids.
- **`TrialDetails` does NOT embed `TrialSummary`.** Once `TrialDetails` carries the raw
  maps, the two share only `run_id` + `status` (both legitimately identify a run) —
  everything `TrialSummary` curates is already inside `TrialDetails`'s raw `metrics`/`tags`.
  A `TrialDetails.summary` field would be a dependency for its own sake.

**The drift risk is solved by sharing the *builder*, not the *type*.** Both types are
constructed from an MLflow run **in the seam** (`trial/` never sees a raw run object). The
shared low-level derivation — "what status / primary-metric-name does this run's tags
encode" — lives in **private seam builder helpers** (e.g. `_status_from_run(run)`,
`_primary_metric(run)` under `mlflow/`), called by both the `TrialSummary` and `TrialDetails`
constructors. Single-sourced derivation, **zero type dependency** between the two domain
types. The domain types themselves carry only `from_dict` (JSON round-trip); the
run→type mapping is the seam's job (00 §9.1).

**Rejected:** `TrialDetails` composes `TrialSummary` (couples two types that, drawn
correctly, are nearly orthogonal — `feedback_clean_single_responsibility_domains`);
re-deriving curated fields independently in each constructor (the current drift —
`feedback_no_redundant_guards`).

---

## 5. Q3 — Field reconciliation: the legacy 19-key dict vs. the slim `TrialSummary`

**Current state.** `_run_to_trial_summary` returns 19 keys; 02's `TrialSummary` is 11 fields
(and added two forward-looking ones, `parent_run_id` + `dataset_hash`). Reconciling so
nothing silently vanishes:

| Legacy key(s) | Disposition |
|---|---|
| `run_id`, `trial_id`→`slug`, `strategy`, `status`, `primary`→`primary_metric_value`, `primary_metric_name` | Already in `TrialSummary` |
| `training_time_s`, `n_features` | **ADD to `TrialSummary`** (review reversal): single metric values → cost-free on the row (no artifact read), and the proposer context reads them on summary rows today. Also present raw in `TrialDetails.metrics`. |
| `pyfunc_model_uri`, `manifest_uri`, `data_contract_uri`, `proposal_artifact_uri`, `has_proposal_artifact` | **Derived at the boundary** (`runs:/{run_id}/…` + a tag) — derive, don't store |
| `snapshot_name` | **Retired** → `dataset_hash` (already in `TrialSummary`; 05 snapshot→dataset cut) |
| `proposal_slug`, `proposal_schema_version` | **Drop** — no consumer beyond the derivable proposal URI |
| `trial_number`, `hypothesis` | **ADD to `TrialSummary`** (below) |
| `training_origin` *(from `LeaderboardRow`)* | **ADD to `TrialSummary`** (below) |

**DECISION: add three demand-backed, tag-derived fields to `TrialSummary`** (additive
carry-back to 02 §6.2.2 — additive is explicitly allowed by the locked schema strategy):

- `trial_number: int | None` — in-experiment sequence (the `automl.trial.number` tag); human
  ordering + CLI `trial list`.
- `hypothesis: str` — "what this trial tried"; consumed by `agent/proposer_context` (11)
  when summarizing recent trials, and by the legacy `summary` (09).
- `training_origin: str` — `"automl"`/`"human"`; `leaderboard()` *filters* on it and
  `recent_failures` carries the filter (09 §8.1), and it displays on the row.

All three are tag-derived → **cost-free** (no artifact reads), so the "cheap row" property is
preserved. Resulting `TrialSummary`:

```python
@dataclass(frozen=True)
class TrialSummary:
    schema_version: int = 1
    run_id: str = ""
    slug: str = ""
    strategy: str = ""
    status: TrialStatus = TrialStatus.UNKNOWN
    primary_metric_name: str = ""
    primary_metric_value: float | None = None
    started_at: str | None = None          # ISO8601
    ended_at: str | None = None
    parent_run_id: str | None = None
    dataset_hash: str | None = None
    trial_number: int | None = None        # ADDED (Q3)
    hypothesis: str = ""                    # ADDED (Q3)
    training_origin: str = ""               # ADDED (Q3)
    training_time_s: float | None = None    # ADDED (Q3 — review reversal; cost-free metric)
    n_features: int | None = None           # ADDED (Q3 — review reversal; cost-free metric)
    @classmethod
    def from_dict(cls, p) -> "TrialSummary": ...
```

---

## 6. Q4 — `TrialDetails.evaluations` reuses eval's `EvalResult`

**Current state.** `inspect/views.py::_eval_views` reads `eval/manifest.json` + per-label
`eval/<label>/report.json` and builds a **raw dict** per evaluation — essentially a
serialized eval result. 07 consolidated the eval result family into a single typed
`EvalResult` (`eval/results.py`); the per-label `report.json` *is* a serialized `EvalResult`.

**DECISION: `TrialDetails.evaluations: list[EvalResult]`** (07's type), deserialized **via
the seam** (`mlflow/` imports `EvalResult` to return it typed — the standard 00 §9.1 seam
pattern). The legacy top-level `primary_label` collapses into the list (each `EvalResult`
already flags whether it is the primary evaluation).

- **Adds a `trial → eval` outbound dependency** (carry-back to 00 §8.7, which today lists
  trial's deps as project/runner/mlflow). A trial's evaluations *are* eval results — reusing
  the type is correct domain modeling, not a leak.
- **Acyclic:** `eval` does not import `trial` (07 §8.4 lists eval's deps as
  project/data/mlflow/utils), so `trial → eval` introduces no cycle.

**Rejected:** a trial-local `TrialEvaluation` duplicating `EvalResult`'s fields (the exact
redundancy/drift `feedback_no_redundant_guards` rejects); `evaluations: list[dict]` (the one
untyped hole in an otherwise-typed `TrialDetails`).

*(The exact `EvalResult` field mapping is owned by 07/`eval/results.py`; 10 references the
type, it does not redefine it.)*

---

## 7. Q5 — `TrialMetadata`: typing the draft, dropping the universe field

**Current state.** `create`/`fork`/`promote` write a local `metadata.json` the runner reads
back at execution. 00 §8.7 lists `TrialMetadata` as a schema in `trial/metadata.py`.

**DECISION: type it (frozen + `schema_version` + `from_dict`) with this reconciliation:**

| Field(s) | Decision |
|---|---|
| `trial_id`, `name`, `slug` | **Collapse to `slug`.** All three are the bare slug today; the real `trial_id = <number>_<slug>` is assigned at **exec time** (08 Q2) and does not exist at draft time. |
| `run_mode` + `dry_run` | **DROP both from the file.** The universe is encoded by the mode-segregated path (08 Q1) and the runner *verifies* path-vs-session (universe-isolation guard); recording the mode in the file too is the redundant, drift-prone state `feedback_no_redundant_guards` + `feedback_dry_run_is_a_container` cut. See §7.1 for the two superseded readers. |
| `strategy`, `hypothesis`, `training_origin`, `created_at`, `project_name`, `project_package`, `experiment_id` | **Keep** — genuine authoring + target metadata. |
| `seed` | **Typed `SeedSelection \| None`** (Q6). |

```python
@dataclass(frozen=True)
class TrialMetadata:
    schema_version: int = 1
    slug: str = ""
    strategy: str = ""
    hypothesis: str = ""
    training_origin: str = ""        # "automl" | "human"
    created_at: str = ""
    project_name: str = ""
    project_package: str = ""
    experiment_id: str = ""
    seed: SeedSelection | None = None
    @classmethod
    def from_dict(cls, p) -> "TrialMetadata": ...
```

### 7.1 The two superseded readers of the metadata mode field (blast-radius trace)

The `metadata.json` mode field has exactly two readers today; both are replaced in the new
design, so the field becomes dead:

1. **Runner** (`runner/_execute.py:283` → `runner/_stages.py::_metadata_declared_dry_run`):
   today resolves `dry_run` from a 3-way precedence chain (a passed default, `AUTOML_DRY_RUN`
   env, and the metadata `run_mode`/`dry_run` — raising on env↔metadata conflict). 08 Q3 +
   `feedback_dry_run_is_a_container` replace this with **mode = `session.dry_run` (single
   source)** + the path-based universe-isolation guard (08 Q1). The reconciliation chain is
   deleted, not ported.
2. **Cleanup** (`cleanup.py:190,384`): reads `metadata.get("run_mode")` for a local trial's
   mode. 03 §9 resolves mode via `mlflow.trial.get_parent_experiment` + session — not the
   local file.

A third reader of *other* dropped metadata keys: the runner reads `metadata.get("trial_id")`
/ `metadata.get("name")` (`_execute.py:123-124`) with a fallback to `trial_dir.name`. With
`trial_id`/`name` collapsed to `slug` (Q5) and the draft folder leaf being the slug (08 Q1),
the fallback to `trial_dir.name` yields the slug — so the runner reads `metadata["slug"]`
(or the dir leaf) with no behavior change. Listed here so the collapse is not a silent
surprise at implementation.

### 7.2 Cross-cutting note — the `run_mode` *routing string* (recorded, not a 10 decision)

Distinct from the metadata file field: the two-valued **routing string** `"dry_run"` /
`"full_run"` is threaded through GCS paths (`mlflow/artifacts/gcs_paths.py`), MLflow
experiment names, and data/eval routing in the legacy code. Per the session convention (01)
+ `feedback_dry_run_is_a_container`, **no domain function takes a `run_mode` (or `dry_run`)
parameter** in the new design — mode lives once on `session.dry_run` (bool), and the
universe token only needs to exist inside `mlflow/_routing.py` as a conditional **`dry_run/`
prefix** (no `"full_run"` literal). This is already implied by settled rules and is a
cross-cutting implementation cleanup (touches 02/05/07's legacy `run_mode=` references), not
a 10 decision; recorded in `open-questions.md` so implementation does not reintroduce the
string. 10 itself only confirms: `trial/` takes `session`, never `run_mode`.

---

## 8. Q6 — `SeedSelection` + where seed *resolution* lives

**Current state.** `_SeedSelection` (private, `creation.py`) records the chosen warm-start
run: `{selector, run_id, trial_id, metric_name, metric_value, strategy}`. `_resolve_seed`
interprets `seed ∈ {auto, best, latest, strategy:<name>}` and calls
`loop_context.queries.top_n_by_metric` / `._runs`; `_resolve_seed_model_source` downloads the
chosen run's `model.py` from MLflow artifacts. **Seed = warm-start the next trial from a good
prior run's `model.py`** (best by metric / latest success / best-of-strategy), instead of the
blank template.

**DECISION.**

- **`SeedSelection` → public frozen schema** in `trial/metadata.py` (`schema_version` +
  `from_dict`), nested as `TrialMetadata.seed`:

  ```python
  @dataclass(frozen=True)
  class ModelSource:                   # typed (review fix) — was an untyped {source, artifact_path} dict
      source: str = ""                 # e.g. "mlflow"
      artifact_path: str = ""          # e.g. "model/code/model.py"

  @dataclass(frozen=True)
  class SeedSelection:
      schema_version: int = 1
      selector: str = ""              # "auto" | "best" | "latest" | "strategy:<name>"
      run_id: str = ""
      trial_id: str = ""
      metric_name: str = ""
      metric_value: float | None = None
      strategy: str = ""
      model_source: ModelSource | None = None   # populated in ONE construction (frozen — no post-hoc mutation)
      @classmethod
      def from_dict(cls, p) -> "SeedSelection": ...
  ```

  **`model_source` is a typed `ModelSource`, not a `dict`** (review fix — the same
  anti-`list[dict]` logic that types `evaluations` in Q4 applies here). And because
  `SeedSelection` is frozen, it is built in a **single construction** with `model_source`
  already resolved — replacing the legacy two-phase pattern (build seed, then mutate
  `metadata["seed"]["model_source"]` in `creation.py:115`).

- **Resolution stays a private helper in `trial/create.py`**, re-pointed from
  `loop_context.queries` to the **seam**: prior-run lookup via
  `mlflow.experiment.top_n_by_metric` / `search_trials`; the `model.py` fetch via
  `mlflow.trial` artifact reads. It is intrinsic to `create(seed=...)` and has no other
  consumer — keeping it in `create.py` keeps trial-authoring logic in one place; it is a
  relocation of *callees* (loop_context → seam), not a redesign. The legacy logical→numeric
  experiment-id lookup (`_mlflow_experiment_id`, `creation.py:255`) is **absorbed by the
  seam** — the seam's `top_n_by_metric` / `search_trials` take the AutoML logical
  `experiment_id` and resolve the numeric id internally (09 Q5 made the numeric id
  seam-internal). The artifact-priority order for finding `model.py`
  (`source/model.py` → `model/code/model.py` → `trial_model_*` → other `model/code/*.py`,
  `creation.py:384`) is private impl that moves with the resolver.

**Rejected:** a separate `trial/seed.py` (one feature, one caller — structure ahead of
demand, `feedback_extension_points_follow_demand`).

---

## 9. Q7 — `TimingReport` + `TrialManifest` field lists

**Current state.** `runner/_stages.py::_TimingRecorder.snapshot()` →
`{unit, total_seconds, phases: dict[str,float]}`. The legacy `write_manifest`
(`mlflow/artifacts/manifest.py`) is a full **navigation spine**:
`{schema_version, run, data, model, evaluation, validation, deployment, artifacts, gcs,
[proposal]}`. **A reader trace (review finding) shows the spine is write-mostly:** the
trial-root `manifest.json` has exactly one consumer — `cleanup.py` (lines 208/358/369/376/
453) — which reads only `run.mlflow_run_id`, `run.experiment_name`, and `gcs.run_prefix`.
The `data`/`model`/`evaluation`/`validation`/`deployment` sections have **zero readers**.
And cleanup's three reads are exactly what **03 §9 re-points to the seam** (caller passes
`run_id`; `mlflow.trial.get_parent_experiment` resolves the experiment; the GCS prefix is
derivable from routing). 02 routes `write_timing` / `write_manifest` through the seam with
these types homed in `trial/metadata.py`, fields left open.

**DECISION.**

```python
@dataclass(frozen=True)
class TimingReport:                       # mirrors _TimingRecorder.snapshot()
    schema_version: int = 1
    unit: str = "seconds"
    total_seconds: float = 0.0
    phases: dict[str, float] = field(default_factory=dict)
    @classmethod
    def from_dict(cls, p) -> "TimingReport": ...

@dataclass(frozen=True)
class ManifestEntry:                      # one artifact the trial declared (nested; no schema_version)
    path: str = ""
    content_type: str = ""
    producer: str = ""
    description: str = ""

@dataclass(frozen=True)
class TrialManifest:                      # SLIM artifact table-of-contents + status (the spine is dropped)
    schema_version: int = 1
    run_id: str = ""                      # self-identifying (review fix) — manifest can be read standalone
    trial_status: TrialStatus = TrialStatus.UNKNOWN
    error: str = ""
    artifacts: list[ManifestEntry] = field(default_factory=list)
    @classmethod
    def from_dict(cls, p) -> "TrialManifest": ...
```

- **Slim TOC, not the legacy spine** (Q7 decision, evidence above). `run` / `data` /
  `model` / `evaluation` / `validation` / `deployment` / `gcs` are **dropped** — they have
  no reader, and their information lives in MLflow tags (`get_details`), the
  `TrialDataContract` (`data_contract.json`), `EvalResult`, and derivable routing. The
  manifest's genuine residual value is the artifact TOC (`producer` + `description`, which
  the seam's `list_artifacts` paths-and-sizes does not carry) + status.
- **`run_id` added** so the manifest self-identifies when read standalone (review fix).
- **`trial_status: TrialStatus`** (the 02 enum), not the legacy free strings (`"crashed"` →
  `FAILED`). One status vocabulary; clean cut.
- **`ManifestEntry` is distinct from 02's `ArtifactRef`.** `ArtifactRef` = what MLflow
  physically lists (path + size); `ManifestEntry` = what the trial *declared* it produced
  (path + producer + description, the self-authored TOC). It is **nested** in `TrialManifest`
  and not independently deserialized, so it carries **no** `schema_version`.
- **No separate per-slice `data` section** — the fit-slice data lives in the
  `TrialDataContract` (`data/contract.py`, 08 Q4); the manifest indexes `data_contract.json`
  as one `ManifestEntry`. No duplication.

*(Writers: `runner/manifest.py` assembles the manifest (08 Q6) and the runner the timing;
the seam (`mlflow.trial.write_manifest` / `write_timing`, 02) persists. 10 owns the types.
**Carry-back to 08:** the writer stops emitting the dropped spine sections. **Verify in 03:**
cleanup's local-manifest reads (`run.mlflow_run_id` / `experiment_name` / `gcs.run_prefix`)
are fully served by the `run_id`-based seam path; the local spine is not needed.)*

---

## 10. Q8 — Signature convention + the mechanical sweeps

**Session convention (01) on every Tier-2 function** — drop `project` / `project_root` /
`dry_run`; take `session: Session | None = None`. The real authoring args stay:

```python
# trial/create.py
def create(slug: str, strategy: str, *, hypothesis: str = "", seed: str | None = None,
           model_source: Path | None = None, training_origin: str = "automl",
           proposal: dict | None = None, session: Session | None = None) -> Path
# trial/fork.py
def fork(slug: str, *, seed: str = "best", strategy: str = "manual_fork",
         hypothesis: str = "", session: Session | None = None) -> Path
# trial/promote.py
def promote(slug: str, model_path: Path, *, hypothesis: str,
            strategy: str = "manual_promote", session: Session | None = None) -> TrialResult
# trial/cleanup.py  (03 §9)
def delete(run_id: str, *, apply: bool = False, hard_delete: bool = False,
           session: Session | None = None) -> CleanupReport
# trial/show.py
def show_trial(run_id: str, *, session: Session | None = None) -> TrialDetails
def load_model(run_id: str, *, session: Session | None = None) -> Any   # mlflow pyfunc
```

**Also:**
- **`run.py` dropped** — the 1-line `run()` delegate goes; callers use `runner.run_trial`.
- **`load_data_snapshot` leaves `trial/`** — it loads a *Dataset*, so it is
  `data.load_dataset_by_trial` (Appendix A), not a trial read.
- **`show.py` = `show_trial` + `load_model` only.** `load_model` lands here per the 06
  correction (pyfunc load needs the seam; `model/` deps = `errors` only).

---

## 11. Q9 — `trial/checks.py`: zero-file

**Current state.** Zero trial-targeted checks exist (legacy builtins are
config/contract/env/model/proposal — no trial); 04 locked **three orchestrators**
(project/model/proposal — no `trial` target); the contract validators (L1–L4) are
data-domain (`data/contract.py`, 05).

**DECISION: no `trial/checks.py`** — nothing to put in it and no orchestrator to dispatch to.
An empty no-caller stub is the speculative structure `feedback_extension_points_follow_demand`
rejects (mirrors 09's zero-file `diagnostics.py`). Add the file + a `validate trial` target
when a real trial-state check appears. **Carry-back to 00 §7/§8.7:** drop `trial/checks.py`
from the folder sketch and remove `checks.py` from trial's listed contents.

---

## 12. Q10 — Folder shape + Tier 2 exports

**DECISION: per-operation files** (mirrors the verb surface 1:1 — the "I know where to find
it with confidence" goal), with a **`types.py` (read contracts) / `metadata.py` (write
schemas)** split.

```
automl/trial/
├── __init__.py     ← Tier 2 exports (§ below)
├── create.py       ← create() + private seed-resolution helper (Q6)
├── fork.py         ← fork()         (create + training_origin="human")
├── promote.py      ← promote()      (create + runner.run_trial)
├── cleanup.py      ← delete()       (thin wrapper → project.cleanup, 03)
├── packaging.py    ← package_model  (notebook→source, stdlib-only, 06)
├── show.py         ← show_trial, load_model
├── types.py        ← TrialSummary, TrialDetails, TrialStatus, ParentExperimentRef   (READ-side seam contracts)
└── metadata.py     ← TrialMetadata, SeedSelection, TimingReport, ManifestEntry, TrialManifest   (WRITE-side schemas)
```
(No `checks.py` — Q9. No `run.py` — Q8.)

**Rejected:** folding `fork`/`promote` into `create.py` (hides two named verbs inside a third
file — weaker discoverability for a marginal file-count win).

**Tier 2 exports (`trial/__init__.py`):**

```python
from automl.trial.create import create
from automl.trial.fork import fork
from automl.trial.promote import promote
from automl.trial.cleanup import delete
from automl.trial.packaging import package_model
from automl.trial.show import show_trial, load_model
from automl.trial.types import TrialSummary, TrialDetails, TrialStatus, ParentExperimentRef
```

The write-schemas (`TrialMetadata` / `SeedSelection` / `TimingReport` / `TrialManifest` /
`ManifestEntry`) stay importable from `automl.trial.metadata` but are not hoisted into the
facade — they are seam/runner-written, not everyday consumer surface. The Tier-1 facade
(00 §12) exposes no trial verbs at the top (`automl.trial.<thing>` is the access path).

---

## 13. Typed schemas

| Type | Home | `schema_version` | Persisted as | Notes |
|---|---|---|---|---|
| `TrialSummary` | `trial/types.py` | 1 | seam return (tags/metrics) | 02-locked + 5 added fields (Q3); seam-returned, cheap row |
| `TrialDetails` | `trial/types.py` | 1 | seam return + eval enrich | Q1; raw maps + `artifacts` + `evaluations: list[EvalResult] \| None` |
| `TrialStatus` | `trial/types.py` | — | enum | 02; `UNKNOWN/RUNNING/FINISHED/FAILED/KILLED`; clean cut from legacy strings |
| `ParentExperimentRef` | `trial/types.py` | 1 | seam return | 02-locked (03 §9.2 cleanup) |
| `TrialMetadata` | `trial/metadata.py` | 1 | local `metadata.json` | Q5; `run_mode`/`dry_run` dropped; `seed: SeedSelection` |
| `SeedSelection` | `trial/metadata.py` | 1 | nested in `TrialMetadata` | Q6 |
| `ModelSource` | `trial/metadata.py` | — | nested in `SeedSelection` | Q6; typed `{source, artifact_path}` |
| `TimingReport` | `trial/metadata.py` | 1 | `timing/summary.json` | Q7 |
| `ManifestEntry` | `trial/metadata.py` | — | nested in `TrialManifest` | Q7; no `schema_version` (nested) |
| `TrialManifest` | `trial/metadata.py` | 1 | `manifest.json` | Q7; slim TOC + `run_id`; `trial_status: TrialStatus` |

**Artifact-schema rule (02 §8/§13.1):** the *type* lives here; the *writer* lives in
`mlflow/trial/*` / `mlflow/artifacts/*` (the writer imports the type, validates, persists).

---

## 14. Dependency directions (00 §8.7)

`trial/` outbound:
- `project` — `Session` (exposes `project_name`, `project_package`, `experiment_id`,
  `dry_run`, connection params — confirm in 01); `project.cleanup` (trial-delete wrapper, 03);
  `project/_import.py`.
- `runner` — the trial-folder **path helper** (08 Q1) + `run_trial` (`promote`); imports
  runner *functions*. **Cycle mechanism:** `runner/` needs trial *types* only as
  annotations, so it imports them under `if TYPE_CHECKING:` — no runtime import of `trial/`
  from `runner/`, so the bidirectional module edge does not become a runtime circular
  import. (The exec-time number already moved to the seam in 08 Q2, removing the other
  back-edge.)
- `eval` — `EvalResult` type for `TrialDetails.evaluations` (Q4). **NEW edge** — acyclic
  (`eval` does not import `trial`). Carry-back to 00 §8.7.
- `utils` — `SLUG_RE` (Q-fix): the slug regex is an AutoML-agnostic primitive shared by
  `trial/create.py` and `agent/proposal.py`. Homing it in `utils/` (not `agent/proposal.py`
  as the checklist drafted) avoids a `trial → agent` cycle (`agent` imports `trial.create`).
- `mlflow` — `mlflow.trial.*`, `mlflow.experiment.*` (`next_trial_number`, seed lookup),
  `mlflow.client.run_url` (URL derivation); the seam returns the trial types it owns.
- `errors` — trial errors as needed.

**Inbound:** CLI verbs (`trial show` / `delete` / `run` / `create` / `fork` / `promote`);
`experiment/views` (imports trial *types*; `compare`/`summary` call `show_trial`); `agent`
(`create` consumes a validated proposal; `proposer_context` reads trial summaries via the
seam); `runner` (imports trial *types* only, under `TYPE_CHECKING`). No cycle:
`experiment → trial`, `agent → trial`, and `runner ↔ trial` (resolved by type-vs-function +
`TYPE_CHECKING`) are all consistent with the seam being one-way (domains call `mlflow.*`;
`mlflow/` imports domain *types*).

*(Note: `automl trial lock {acquire,release}` is a CLI verb that maps to
`runner.trial_lock` (08 Q6 — `session_lock.py` lives in `runner/`), **not** a `trial/`
function. It is a trial-namespaced verb, not part of this domain's surface.)*

---

## 15. Mechanical migration map

(Status `[ ]` — design settled, not yet built.)

| Legacy symbol | New home | Notes |
|---|---|---|
| `trial/creation.py::create` | `trial/create.py::create` | 08 Q1 path via runner helper; session conv.; `trial_id/name/slug`→`slug` |
| `trial/creation.py::_SeedSelection` | `trial/metadata.py::SeedSelection` (public) | Q6 |
| `trial/creation.py::_resolve_seed*` / `_resolve_seed_model_source` | private helper in `trial/create.py` (re-pointed to seam) | Q6 |
| `trial/creation.py::_next_trial_number_from_mlflow` (+ `_run_trial_number`) | `mlflow.experiment.next_trial_number` (seam) | 08 Q2 |
| `trial/creation.py::_mlflow_experiment_id` (logical→numeric) | **absorbed by the seam** | seam takes logical `experiment_id`, resolves numeric internally (09 Q5) |
| `trial/creation.py::_normalize_training_origin` | private in `trial/create.py` | **drop the `"agent"→"automl"` alias** (clean cut; `"automl"`/`"human"` only) |
| `trial/creation.py::_move_orphan_trial_dir` | **DROP — dead code** | grep confirms zero callers |
| proposal write path `proposal/trial_proposal.json` | `proposal/proposal.json` | adopt 02 §9.2's path (rename; clean cut) |
| `trial/fork.py::fork` | `trial/fork.py::fork` | session conv. |
| `trial/promotion.py::promote` | `trial/promote.py::promote` | session conv.; calls `runner.run_trial` |
| `trial/cleanup.py::cleanup` | `trial/cleanup.py::delete(run_id, *, apply, hard_delete, session)` → `CleanupReport` | 03 §9; dropped `trial_id`/`dry_run`/`confirm_project` |
| `trial/packaging.py::package_model` | `trial/packaging.py::package_model` | 06 correction; stdlib-only |
| `trial/run.py::run` | **DROP** | callers use `runner.run_trial` |
| `inspect/views.py::show_trial` | `trial/show.py::show_trial` → `TrialDetails` | Q1; enriched with `evaluations` |
| `inspect/views.py::load_model` | `trial/show.py::load_model` | 06 correction |
| `inspect/views.py::load_data_snapshot` | `data.load_dataset_by_trial` | **leaves trial** (Appendix A) |
| `loop_context/queries.py::show_trial` (raw) | **DROP** → `mlflow.trial.get_details` + `trial.show_trial` | superseded |
| `mlflow/store.py::_run_to_trial_summary` | seam builder for `TrialSummary` (type owned in `trial/types.py`) | Q2/Q3; shared private builder helpers |
| `mlflow/store.py::get_trial_summaries` | `mlflow.experiment.list_trials` (seam) → `list[TrialSummary]` | 02 §6.2.2 |
| (timing) `runner/_stages.py::_TimingRecorder.snapshot` | `TrialManifest`/`TimingReport` types in `trial/metadata.py`; written by runner/seam | Q7 |
| `SLUG_RE` (dup in `creation.py` + `propose/__init__.py`) | **`utils/` (revised)** — not `agent/proposal.py`; both `trial/create.py` and `agent/proposal.py` import it | checklist 763; avoids `trial → agent` cycle |

---

## 16. Cross-doc reconciliations (precedence: 00 > sub-spec > checklist)

1. **`TrialDetails`/`TimingReport`/`TrialManifest` fields** — 02 named the types + homes but
   left fields open; 10 settles them (Q1/Q7). No conflict, completion.
2. **`ComparisonResult.runs` element type** — 09 deferred it to "follows `trial.show_trial`";
   10 sets `list[TrialDetails]` (Q1). This **supersedes** 09 §12's `list[dict]` placeholder
   *and* its prose that the element carried an `mlflow_url` (Q1 drops the stored URL — derived
   at the boundary). `ComparisonResult.from_dict` must deserialize nested elements
   (`[TrialDetails.from_dict(r) for r in payload["runs"]]`) — the generic flat `from_dict`
   recipe would leave them as raw dicts. Carry-back to 09 §12 **applied** (see §19).
3. **`trial/checks.py`** — 00 §7/§8.7 sketch lists it; 10 zero-files it (Q9). Carry-back.
4. **`trial → eval` edge** — not in 00 §8.7's trial deps; 10 adds it (Q4). Carry-back.
5. **`TrialSummary` field additions** — additive (allowed by the locked schema strategy);
   carry-back to 02 §6.2.2 (Q3).

---

## 17. Review log

### Round 1 — three agents (fresh-eyes design / codebase gap-detection / coverage validation)

Reviewed against 00/02/03/06/08/09 + current code; `pending/` excluded.

**Applied (real findings):**
- **`TrialManifest` was under-scoped.** The draft slimmed from the failure-rewrite code, but
  the real `write_manifest` is a full navigation spine. A reader trace showed the spine is
  write-mostly (only `cleanup.py` reads `run`/`gcs`, and 03 re-points that to the seam), so the
  **slim TOC** is correct — `data`/`model`/`evaluation`/`validation`/`deployment`/`gcs`/`run`
  dropped; `run_id` added for self-id (§Q7). Carry-back to 08 (writer stops emitting the spine).
- **`training_time_s` / `n_features`** re-added to `TrialSummary` — cost-free metrics the
  proposer reads on summary rows (§Q3 reversal).
- **`SeedSelection.model_source` typed** as `ModelSource` (was an untyped `dict`) + single-shot
  construction for the frozen dataclass (§Q6).
- **`get_details` vs `show_trial` evaluations ambiguity** — `evaluations: list[EvalResult] | None`
  (`None` = not loaded; `[]` = loaded-empty) (§Q1).
- **`SLUG_RE` → `utils/`** (not `agent/proposal.py`) to avoid a `trial → agent` cycle (§14/§15).
- **`trial ↔ runner` cycle mechanism** made explicit — `runner/` imports trial types under
  `TYPE_CHECKING` (§14).
- **`_move_orphan_trial_dir` dropped** — grep confirms dead code (§15).
- **`ComparisonResult.runs`** supersede note + nested `from_dict` requirement (§16); carry-back
  to 09 §12 applied.
- Smaller: `"agent"→"automl"` origin alias dropped (§15); `proposal/proposal.json` rename (§15);
  `_mlflow_experiment_id`→seam (§Q6/§15); `ManifestEntry` no `schema_version` (nested) §9/§13
  reconciled; `mlflow/_routing.py` filename (§7.2); `trial lock` clarified as `runner.trial_lock`
  (§14); `Session` field exposure noted (§14); runner `trial_id`/`name` reads noted (§7.1).

**Flagged as false-positive / no-change:**
- "`confirm_project` safety gate dropped" — **03 §9** decided no interactive confirmation. Intended.
- "`apply: bool = False` reverses always-apply" — **03's** plan/apply two-phase, preview-by-default. Intended.
- "`load_model` loses soft tracking-URI fallback" — intended seam discipline (`StorageError` when unbound).
- "`LeaderboardRow` has no home" — **09** owns it (dropped → `LeaderboardData`).
- "`EvalResult` field coverage unconfirmed" — **07** owns the fields; 10 correctly defers (§6).
- "`_candidate_seed_artifacts` priority unspecified" — private impl that moves with the resolver (§Q6).

---

## 18. Open decisions for human review

All resolved in the interview (2026-05-25). Recorded for traceability:

1. `show_trial` → typed `TrialDetails`; `get_details` (seam) + eval enrich; `ComparisonResult.runs: list[TrialDetails]`. **RESOLVED — Q1.**
2. `TrialSummary` / `TrialDetails` independent; shared seam builder helpers, no type composition. **RESOLVED — Q2.**
3. Add `trial_number` / `hypothesis` / `training_origin` to `TrialSummary` (additive). **RESOLVED — Q3.**
4. `TrialDetails.evaluations: list[EvalResult]`; `trial → eval` edge. **RESOLVED — Q4.**
5. `TrialMetadata` typed; `trial_id/name/slug`→`slug`; drop `run_mode`/`dry_run`. **RESOLVED — Q5.**
6. `SeedSelection` public schema; seed resolution stays in `create.py`, re-pointed to seam. **RESOLVED — Q6.**
7. `TimingReport` / `ManifestEntry` / `TrialManifest` fields; `trial_status: TrialStatus`. **RESOLVED — Q7.**
8. Session convention sweep; `run.py` dropped; `load_data_snapshot`→data; `show.py`=`show_trial`+`load_model`. **RESOLVED — Q8.**
9. Zero-file `trial/checks.py`. **RESOLVED — Q9.**
10. Per-operation files; `types.py`/`metadata.py` split; Tier-2 exports. **RESOLVED — Q10.**

---

## 19. Proposed carry-backs

*To `02`:*
- **Add five fields to `TrialSummary`** (§6.2.2): `trial_number: int | None`, `hypothesis: str`,
  `training_origin: str`, `training_time_s: float | None`, `n_features: int | None` (additive — Q3).
- **Define `TrialDetails`'s fields** (§6.3.3) — 02 named the return type but left it open:
  `run_id`, `status`, `params`, `metrics`, `tags`, `artifacts: list[ArtifactRef]`,
  `evaluations: list[EvalResult] | None` (Q1/Q4). `get_details` returns `evaluations=None`
  (cheap, not loaded); `trial.show_trial` fills it (`[]` = loaded-empty).

*To `00`:*
- **§8.7** — add `eval` to `trial/`'s outbound deps (`TrialDetails.evaluations: list[EvalResult]`, Q4).
- **§7 + §8.7** — drop `trial/checks.py` from the folder sketch + trial's listed contents (Q9, zero-file).
- **§8.7** — Tier-2 exports add `delete` (cleanup); `metadata`/`types` split is as drawn.

*To `09` (the 10-side carry-out it deferred):*
- **§12** — `ComparisonResult.runs: list[TrialDetails]` (was `list[dict]` placeholder); the element
  no longer carries `mlflow_url` (derived at boundary); `from_dict` deserializes nested.

*To the checklist (`SLUG_RE`):*
- `SLUG_RE` canonical home → **`utils/`** (was drafted as `agent/proposal.py`); avoids `trial → agent` cycle.

*To `08` (carry-back):*
- The `write_manifest` writer (`runner/manifest.py`) stops emitting the dropped spine sections
  (`run`/`data`/`model`/`evaluation`/`validation`/`deployment`/`gcs`); writes the slim
  `TrialManifest` TOC + status (Q7).

*To `open-questions.md`:*
- Record the **`run_mode` routing-string collapse** cross-cutting note (§7.2): the two-valued
  string → `session.dry_run` (bool) + conditional `dry_run/` prefix in `mlflow/_routing.py`;
  no domain threads `run_mode`/`dry_run`; no `"full_run"` literal stored.
- Mark the 09→10 carry-outs RESOLVED: `trial.show_trial` element type (`list[TrialDetails]`);
  `TrialSummary`/`TrialDetails`/`TimingReport`/`TrialManifest` field lists.

*To `migration-checklist.md`:*
- Flip the `automl/trial/*` rows + `inspect/views.py::show_trial`/`load_model` rows to reflect
  the homes settled here; `trial/run.py` → DROP; `load_data_snapshot` → `data.load_dataset_by_trial`;
  `_move_orphan_trial_dir` → DROP (dead).
