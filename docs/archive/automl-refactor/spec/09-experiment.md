# Sub-spec 09 — Experiment domain

**STATUS: APPROVED 2026-05-25.** (Human sign-off received. The §18 carry-backs to
`00`/`02` are batched with sub-spec 10's closeout — several touch the trial-type lines
10 redefines, so applying them once avoids double-edits.)

**Date:** 2026-05-25
**Sub-spec:** 09 of the AutoML refactor (see `README.md` + `00-structural-design.md`).

**Scope.** Settle the interface + internal shape of the **slimmed** `experiment/`
domain — the Experiment noun, its overview-run state, the experiment-scope cleanup
wrapper, and the **cross-trial read views** (leaderboard / compare / summary /
experiments-listing + the demand-backed aggregations). It does **not** cover Trial
operations (→ sub-spec **10**, `trial/`) or the Proposal contract + agent loop (→
sub-spec **11**, `agent/`).

**The big move.** During this sub-spec the legacy `experiment/` mega-domain — which
had absorbed Trial ops, the Proposal contract, every read view, the agent-loop
launcher, and the agent-timeline reconciliation — was **dissolved into three peer
domains**: `experiment/`, `trial/`, `agent/`. The structural change is applied to
`00` (§5/§6/§7/§8.6–§8.8/§11.1/§12/§13.1/§16/§17, Appendix A) and to `02` (Trial type
homes). See `00` §8.6 decomposition note + §17.12. This doc specs the `experiment/`
third; 10 and 11 spec the other two.

**Reshape, not invent.** The functionality already exists in
`automl_legacy/inspect/views.py`, `automl_legacy/loop_context/{queries,summary}.py`, and
`automl_legacy/mlflow/store.py` (the experiment-overview + summary pieces). 09 puts each
piece in its canonical home and threads it through the locked seam (02) and sibling
domains (01, 03, 10).

---

## 1. Context — what `experiment/` owns now, and what left

| Current location | What it does | New home |
|---|---|---|
| `mlflow/store.py` overview/lifecycle pieces | experiment-overview run state | `ExperimentOverview` type in `experiment/store.py`; IO via `mlflow.experiment.*` (02) |
| `mlflow/store.py::ensure_experiment_overview` | idempotent overview-run bootstrap | seam `mlflow.experiment.ensure_overview`; `experiment/lifecycle.py::create` is the explicit wrapper |
| `inspect/views.py::leaderboard` (+ `LeaderboardRow`) | leaderboard | `experiment/views/leaderboard.py` → `LeaderboardData` |
| `inspect/views.py::compare` | pairwise trial comparison | `experiment/views/compare.py` → `ComparisonResult` (composes `trial.show_trial`) |
| `inspect/views.py::experiments` | list logical experiments + counts | `mlflow.project.list_experiments` (seam) + view enrichment |
| `loop_context/queries.py::{recent_failures,strategies_attempted}` | aggregations | `experiment/views/queries.py` (compose seam) |
| `loop_context/queries.py::{top_n_by_metric,show_trial}` | raw searches | **seam** (`mlflow.experiment.top_n_by_metric`, `mlflow.trial.get_details`) |
| `loop_context/queries.py::{runs_using_strategy,runs_in_metric_band}` | no-caller analytics | **deferred** (recorded; no placeholder file — §Q4) |
| `loop_context/__init__.py::experiment_id` | numeric MLflow id | **dropped** (§Q5) |
| `loop_context/summary.py::{load_mlflow_context,build_summary,build_summary_from_context}` | experiment summary | `experiment/views/summary.py` (learning_counts dropped — §Q7) |

**Left this domain (specced elsewhere):**
- Trial ops (`create`/`fork`/`promote`/`cleanup`/`packaging`/`metadata`), `show_trial`,
  `load_model`, and the `TrialSummary`/`TrialDetails`/`ParentExperimentRef` types →
  **sub-spec 10 (`trial/`)**.
- The Proposal contract, `proposal_schema`, `gather_proposer_context`, `build_launch`,
  `agent_timeline` → **sub-spec 11 (`agent/`)**.

**Out of scope (stays in `automl_legacy/`):** the project-level **learning subsystem**
(`write_learning_cache` / golden-weak feature artifacts / `experiment_learnings` /
`project_learnings`). The new `summary` does not read or emit them (§Q7).

---

## 2. Locked invariants this domain inherits (not re-litigated)

- **Seam-only (00 §9.1 / §13.4).** `experiment/` never `import mlflow`. Every read/write
  goes through `automl.mlflow.<noun>` and comes back as a typed domain object; views are
  thin formatting/composition on top.
- **Session convention (01).** Every IO-touching Tier-2 function takes
  `session: Session | None = None`, resolved `session if session is not None else
  automl.session()`. Pure-compute helpers take no `session`.
- **Cleanup cascade lives in `project/` (03).** `experiment/cleanup.py` is a thin wrapper.
- **Schema rule (02 §8).** Persisted/serialized typed schemas are frozen dataclasses with
  `schema_version: int` + `from_dict` (strips unknown keys); additive-only.
- **Views import trial types + call `trial.show_trial` (00 §8.6/§8.7).** No cycle:
  `experiment → trial` is one-directional.

---

## 3. Q1 — The `Experiment` noun: one type, not two

**Current state.** 02 §6.2.1 locked `ExperimentOverview` — a frozen dataclass
(`schema_version` + `from_dict`) in `experiment/store.py` representing the persisted
experiment-overview-run state. Its locked field set is `{schema_version, experiment_id,
project_name, created_at, dry_run}`. 00 §5/§12 separately calls for an `Experiment` noun
class. They overlap heavily.

**DECISION: one type.** `ExperimentOverview` *is* the experiment's state record; the
facade noun `Experiment` is an alias of it (`Experiment = ExperimentOverview`,
re-exported under the noun name). `lifecycle.create()` returns `ExperimentOverview`.
There is exactly one record of an experiment's state; a second near-identical dataclass
is the drift the refactor fights. Unlike a Trial (which has a real summary-vs-details
split), an Experiment has no second representation that earns its own type.

**Rejected:** a lightweight `Experiment` "handle" separate from the persisted state —
no consumer needs it; one-liner to add later if so. *(Carry-back to 00 §12: the facade
`Experiment` maps to `ExperimentOverview` — applied.)*

---

## 4. Q2 — `lifecycle.create`

**Current state.** No lifecycle module today. Experiments are created *implicitly* by
`store.ensure_experiment_overview` (idempotent) during the agent bootstrap /
proposer-context gather. No explicit create/archive/delete in `store.py`.

**DECISION.**

```python
# experiment/lifecycle.py
def create(experiment_id: str | None = None, *, session: Session | None = None) -> ExperimentOverview:
    """Explicit experiment bootstrap — the explicit form of today's implicit ensure."""
```

- Wraps the seam's `mlflow.experiment.ensure()` + `ensure_overview()` (02 §6.2.1).
- **No predecessor parameter.** The legacy `predecessor_experiment_overview_run_id` tag
  was **write-only** (set in `store.py`, never read back anywhere) and is already marked
  *retired* in the active code (`data/pipeline.py`). 02 correctly dropped it; 09 does not
  resurrect it. `create` takes only `experiment_id` + `session`. *(The actual cold-start
  "prior experiment" feature is `find_prior_experiment`, which is independent of this
  dead tag and lives in `agent/proposer_context` — sub-spec 11; see §Q3.)*
- Called by the `automl experiment run` setup path.

**`create` vs the lazy ensure — both kept (not redundant).** `lifecycle.create` is the
explicit primary bootstrap; `agent/proposer_context`'s `ensure_overview` (sub-spec 11)
is a lazy safety net so a proposer turn always has an overview run to read even when
invoked outside the bootstrap. Both are idempotent ensures (02), so running both is
harmless; neither is "the redundant one."

**Seam-`ensure` side-effects (02's contract, not re-specified here).** `create` relies on
the seam's `ensure`/`ensure_overview` for the `created_by` tag on first creation and the
project-overview bootstrap. **It does NOT restore a soft-deleted experiment** — resolved in
02 §6.2.1 (2026-05-26): the legacy `_activate_experiment` restore is **dropped** (user
direction — archived stays archived; a same-name collision surfaces as `StorageError`,
resolved by `--hard-delete` or a different `experiment_id`, not by silent resurrection).
These are `02`'s `ensure` contract; 09 does not own them.

**`archive` — DEFERRED.** No implementation exists today; 00 §11.1 defers the CLI verb.
Per `feedback_extension_points_follow_demand`, 09 does not ship a non-functional
`archive`. *(Carry-back to 00 §8.6 Tier-2 exports: `archive` removed — applied.)*

---

## 5. Q3 — Query homes: seam vs. `views/` vs. deferred

**Current state + the conflict.** `inspect/views.py` and `loop_context/queries.py` each
call MLflow directly and return raw dicts; three files re-implement the same
`_read_json_artifact` / `_metric_value` helpers. The **migration-checklist** maps the
`loop_context/queries.py` functions into `experiment/views/queries.py`; but 00 §9.1/§6 +
02 §6.2.2 put **raw MLflow searches at the seam**. Genuine conflict.
**Precedence: 00 > sub-spec > checklist.**

**DECISION.**

- **Raw searches → seam** (02-locked; no new primitive): `list_trials(status=, limit=)`,
  `top_n_by_metric(...)`, `search_trials(filter_string=...)` → `list[TrialSummary]`;
  `mlflow.trial.get_details(run_id) -> TrialDetails`.
- **`views/queries.py` holds only compose-over-seam helpers** (zero `import mlflow`):

  | Helper | Realization |
  |---|---|
  | `recent_failures(n, *, session=None)` | `mlflow.experiment.list_trials(status=TrialStatus.FAILED, limit=n)` (newest-first) |
  | `strategies_attempted(*, session=None)` | aggregate `mlflow.experiment.list_trials()` by `strategy` |

- **No-caller analytics deferred:** `runs_using_strategy` / `runs_in_metric_band` have no
  caller today → recorded as deferred (no placeholder file — §Q4). Re-add when a real
  consumer appears: raw search at the seam + a thin `views/queries.py` composition.
- **`show_trial` (raw) dropped** → superseded by `mlflow.trial.get_details` + `trial.show_trial`.

**`recent_failures` and `compare` are IN scope** — the README "out of scope" prose lists
them under the diagnostics placeholder, but 00 §11.1 ships `automl experiment compare`
and `agent/proposer_context` (sub-spec 11) is a live consumer of `recent_failures`.
**Precedence: 00 §11.1 verb catalog + live consumer > README prose** (written before the
per-symbol demand pass). *(Carry-back to README out-of-scope wording — listed §11.)*

The duplicated `_read_json_artifact` / `_metric_value` helpers collapse because artifact
reads move behind the seam.

---

## 6. Q4 — Diagnostics placeholder: zero files

**The tangle.** 02 references a domain-side `experiment/diagnostics.py` placeholder (for
deferred analytics) and a seam-side `mlflow/experiment/diagnostics.py` (02 §4). The two
no-caller queries are MLflow reads (seam) whose composition would be domain logic.

**DECISION: zero placeholder files.** An empty no-caller stub is exactly the speculative
structure `feedback_extension_points_follow_demand` rejects; a missing file is cleaner
than one that invites premature filling. `runs_using_strategy` / `runs_in_metric_band`
are recorded as deferred in `open-questions.md` + `migration-checklist.md`. When a real
caller lands, the raw search goes to the seam and a thin composition to `views/queries.py`.
The slimmed `experiment/` tree in 00 §7 already omits `diagnostics.py`. *(Carry-back to
02: drop its `experiment/diagnostics.py` + `mlflow/experiment/diagnostics.py` placeholder
references — §11.)*

---

## 7. Q5 — Drop the public `experiment_id()` helper

**Current state.** `loop_context.experiment_id()` returns MLflow's **internal numeric**
experiment id. The seam resolves names→ids itself, so this is a seam implementation
detail; review found no external consumer.

**DECISION: drop.** The numeric id is seam-internal. `agent/proposer_context` (sub-spec
11) still emits an `mlflow_experiment_id` **data key** in its packet — but that is
composed from a seam read, not from this public function, so dropping the function loses
no data. If a tool ever needs the numeric id, a seam accessor (`mlflow.experiment.
numeric_id()`) is a one-liner — add on demand.

---

## 8. Q6 — View output typing

**Current state.** Every view returns a raw dict/list today (`LeaderboardRow` is the only
typed piece, and it carries a derived `mlflow_url`).

**DECISION.** Type the **structured** outputs; leave the **heterogeneous bag** a dict.
The cross-seam row elements (`TrialSummary` / `TrialDetails`) stay typed regardless —
they are durable seam contracts owned by `trial/` (sub-spec 10).

| Output | Type | Rationale |
|---|---|---|
| `leaderboard` | `LeaderboardData` (frozen; `from_dict`) | stable shape; CLI `--json` + inspect skill consume it |
| `compare` | `ComparisonResult` (frozen; `from_dict`) | stable shape; adds typed `metric_deltas` |
| `summary` | `dict` | heterogeneous full-history bag; no programmatic consumer; typing every nested section is over-abstraction |
| `experiments` (listing) | `list[dict]` | thin id + counts; CLI-facing |

`LeaderboardData.rows` are `TrialSummary` (the trial type); the legacy `LeaderboardRow`
+ its derived `mlflow_url` field are replaced — **the URL is derived by the CLI/view via
a seam URL helper, not stored on the row** (mirrors 07's `mlflow_url` drop from eval
results). 00 §13.1 already lists `LeaderboardData` / `ComparisonResult` — kept.

### 8.1 View signatures

```python
# experiment/views/leaderboard.py
def leaderboard(*, metric: str | None = None, n: int = 5,
                training_origin: str | None = None,
                session: Session | None = None) -> LeaderboardData
# metric defaults to the experiment's CURRENT primary = session.config.primary_metric (resolved
# when None; was a hardcoded "auc" — sub-spec 11 #2). Optional explicit metric overrides it
# (the `--metric` CLI flag). The metric is addressed by the cross-trial-stable <label>.<metric>
# key (02 top_n_by_metric note, sub-spec 11 #3), so a trial that never computed it is reported
# as MISSING (see LeaderboardData.n_unscored) rather than mis-ranked.
# wraps mlflow.experiment.top_n_by_metric. Routing (incl. the dry_run universe) comes from
# `session` via the seam — NOT a dry_run param (dry_run is a session-level container).
# Each row's MLflow URL is derived by the CLI/view via the seam helper `mlflow.client.run_url`
# (02 carry-back — the helper already exists as store.py::run_url), NOT stored on the row.

# experiment/views/compare.py
def compare(run_ids: list[str], *, session: Session | None = None) -> ComparisonResult
# composes trial.show_trial per run_id (each keeps its eval block); pairwise deltas over the first two.
# CLI `automl experiment compare <id1> <id2>` passes a 2-element run_ids list (run_ids = trials).
# NOTE: today's compare returns the deltas under key "metrics"; ComparisonResult renames it to
# `metric_deltas` (clean cut — the inspect skill prose is updated to match).

# experiment/views/summary.py
def build_summary(*, session: Session | None = None) -> dict
def build_summary_from_context(context: dict) -> dict          # pure-compute; no session
def load_mlflow_context(*, session: Session | None = None) -> dict
# Composes seam reads DIRECTLY: mlflow.experiment.list_trials() (no limit) for full history,
# top_n_by_metric for the leaderboard block, the views/queries helpers for failures/strategies.
# It must NOT call agent/proposer_context's gather — `experiment/` may not import `agent/`
# (the dependency runs agent → experiment). This replaces today's load_mlflow_context, which
# delegated to gather_proposal_context; the new one self-populates the keys
# build_summary_from_context reads (top_trials, recent_failures, …).

def experiments(*, session: Session | None = None) -> list[dict]
# mlflow.project.list_experiments() returns logical experiment_ids — the name-filter rules
# (prefix-match + exclude the "overview" experiment + exclude nested names) are a SEAM concern
# (carry-back to 02). Per-id enrichment: trial_count via list_trials; top metric via
# top_n_by_metric(n=1). Each row: {experiment_id, mlflow_experiment_id, name, trial_count,
# top_metric_name, top_metric_value}.

# experiment/views/queries.py
def recent_failures(n: int = 3, *, training_origin: str | None = None,
                    session: Session | None = None) -> list[TrialSummary]
# = mlflow.experiment.list_trials(status=TrialStatus.FAILED, limit=n); training_origin filter
# preserved (the live agent/proposer_context consumer passes "automl").
def strategies_attempted(*, session: Session | None = None) -> dict[str, int]
# aggregates mlflow.experiment.list_trials() with NO status filter (counts ALL trials, as today).
```

**`experiments()` listing — CLI destination.** `automl experiment list` calls the
enriched `experiments()` view above (not the bare seam `list_experiments`, which returns
only id strings). This is a **carry-back to 00 §11.1** (the verb's library destination
becomes `experiment.views.experiments`, not `mlflow.project.list_experiments`). The N+1
read pattern (one `list_trials` / `top_n_by_metric` per id) matches today's
per-experiment `search_runs`. The name-filter rules that decide which MLflow experiments
are "logical" (vs the overview experiment) move into the seam's `list_experiments`
(carry-back to 02).

---

## 9. Q7 — `summary` clean cut: drop `learning_counts`

**Current state.** `build_summary_from_context` reads `experiment_learnings` /
`project_learnings` to emit a `learning_counts` block.

**DECISION: drop the key.** With the learning subsystem out of scope (stays in
`automl_legacy/`), `learning_counts` is **not emitted** — not zeroed (clean cut,
`feedback_no_back_compat`). Everything else the summary emits is preserved:
`summary_kind`, `project`, `project_name`, `experiment_id`, `completed_at`, `overview`,
`trial_count`, `trial_count_by_status`, `best_trials`, `strategy_outcomes`,
`failed_strategies`, `data_context`, `data_context_keys`, `artifact_uris`,
`artifact_errors`.
`load_mlflow_context` keeps fetching **all** trials (no limit) to populate
`trial_summaries`, preserving full-history behavior. The `TrialStatus` enum (02 §6.2.2)
is canonical; legacy `"success"`/`"failed"` string casing is not read by new code.

---

## 10. Folder layout (`experiment/`)

```
automl/experiment/
├── __init__.py              ← Tier 2 exports (§ below)
├── lifecycle.py             ← create() ; Experiment = ExperimentOverview (archive deferred)   [Q1/Q2]
├── store.py                 ← ExperimentOverview type + thin overview accessors (IO via seam)  [Q1]
├── cleanup.py               ← delete() experiment-scope thin wrapper → project.cleanup (03)
├── checks.py                ← experiment-state checks (validate framework, 04)
└── views/
    ├── __init__.py
    ├── types.py             ← LeaderboardData, ComparisonResult, MetricDelta                   [Q6]
    ├── leaderboard.py       ← leaderboard() -> LeaderboardData
    ├── compare.py           ← compare() -> ComparisonResult (composes trial.show_trial)
    ├── summary.py           ← build_summary / build_summary_from_context / load_mlflow_context
    │                          + experiments-listing enrichment                                 [Q3/Q7]
    └── queries.py           ← recent_failures, strategies_attempted (compose seam; no MLflow)   [Q3]
```

No `diagnostics.py` (§Q4). No `proposal.py` / `agent_*` / `trial/` (moved to 11 / 10).

---

## 11. Tier 2 exports (`experiment/__init__.py`)

```python
from automl.experiment.lifecycle import create, Experiment   # Experiment = ExperimentOverview
from automl.experiment.store import ExperimentOverview
from automl.experiment.cleanup import delete                  # experiment-scope (03 cascade)
from automl.experiment.views import (
    leaderboard, compare, build_summary, experiments,
    LeaderboardData, ComparisonResult, MetricDelta,
)
```

Facade (00 §12) re-exports `Experiment` at `automl.Experiment`. `recent_failures` /
`strategies_attempted` are view-internal helpers (consumed by `agent/proposer_context`
via `automl.experiment.views.queries`), not facade nouns.

**Delete-verb asymmetry (deliberate).** `automl.experiment.delete` is the
experiment-scope wrapper; the trial-scope delete is `automl.trial.cleanup.delete`
(sub-spec 10). Mirrors the CLI `automl experiment delete` vs `automl trial delete`.

---

## 12. Typed schemas

| Type | Home | `schema_version` | Persisted as | Notes |
|---|---|---|---|---|
| `ExperimentOverview` | `experiment/store.py` | 1 (02) | experiment-overview run | 02-locked; aliased as `Experiment` (§Q1) |
| `LeaderboardData` | `experiment/views/types.py` | 1 | leaderboard JSON output | container; `rows: list[TrialSummary]` |
| `ComparisonResult` | `experiment/views/types.py` | 1 | compare JSON output | replaces ad-hoc dict |
| `MetricDelta` | `experiment/views/types.py` | — | nested in `ComparisonResult` | tiny row dataclass |

```python
@dataclass(frozen=True)
class LeaderboardData:
    schema_version: int = 1
    metric: str = ""               # the <label>.<metric> the rows are ranked by (resolved from config.primary_metric)
    experiment_id: str = ""        # AutoML experiment id (NOT the MLflow integer id)
    rows: list[TrialSummary] = field(default_factory=list)   # from trial/types.py (seam-returned); only trials that computed `metric`
    n_unscored: int = 0            # trials in the experiment that never computed `metric` (sub-spec 11 #2 — render "x/n not scored on <metric>"; x = len(rows), n = len(rows) + n_unscored)
    @classmethod
    def from_dict(cls, payload: dict) -> "LeaderboardData": ...

@dataclass(frozen=True)
class MetricDelta:
    metric: str = ""
    value_a: float | None = None
    value_b: float | None = None
    delta: float | None = None

@dataclass(frozen=True)
class ComparisonResult:
    schema_version: int = 1
    run_ids: list[str] = field(default_factory=list)
    runs: list[TrialDetails] = field(default_factory=list)          # settled by sub-spec 10 Q1 (was list[dict] placeholder)
    metric_deltas: list[MetricDelta] = field(default_factory=list)  # pairwise over the first two runs
    @classmethod
    def from_dict(cls, payload: dict) -> "ComparisonResult": ...
```

**Serialization-direction policy.** View-output types are serialized to JSON at the CLI
boundary via `dataclasses.asdict` (the `--json` formatter), so they need only `from_dict`
for round-tripping (no `to_dict`). `ExperimentOverview` carries `from_dict` per 02.
`ComparisonResult.runs` is `list[TrialDetails]` (settled by sub-spec 10 Q1 — the enriched
`show_trial` output, now typed). The element no longer carries an `mlflow_url` (sub-spec 10
drops the stored URL; it is derived at the boundary). `ComparisonResult.from_dict` must
deserialize nested elements (`TrialDetails.from_dict` per run). The typed value `compare`
adds is `metric_deltas`.

`TrialSummary` / `TrialDetails` (referenced above) are owned by `trial/types.py`
(sub-spec 10); the seam returns them. `FailureRecord` (00 §13.1 reserved name) is **not
defined** — `recent_failures` returns `list[TrialSummary]` with `status == FAILED`; no
distinct type has a consumer (`feedback_extension_points_follow_demand`).

---

## 13. Dependency directions (00 §8.6)

`experiment/` outbound:
- `project` — `Session`; `project.cleanup` (experiment-delete wrapper, 03); `project/_import.py`.
- `trial` — view types (`TrialSummary` / `TrialDetails`); `compare` / `summary` call `trial.show_trial`.
- `data` — active-Dataset pin read **via seam** (not by importing `data/` internals).
- `mlflow` — `mlflow.project.*`, `mlflow.experiment.*`, `mlflow.trial.*` (the seam).
- `validate` — `experiment/checks.py` registers with the framework (04).
- `errors` — experiment-state errors as needed.

**Inbound:** CLI verbs (`experiment leaderboard` / `compare` / `summary` / `delete`),
`agent/proposer_context` (sub-spec 11; composes the views + `views/queries.py` helpers).
No cycle: `experiment → trial` and `agent → experiment` are one-directional; the seam is
one-way (domains call `mlflow.*`; `mlflow/` imports domain *types*).

---

## 14. Mechanical migration map

(Status `[ ]` — design settled, not yet built.)

| Legacy symbol | New home | Notes |
|---|---|---|
| `mlflow/store.py` ExperimentOverview-state pieces | `experiment/store.py::ExperimentOverview` (type) + `mlflow.experiment.*` (IO) | 02 §6.2.1 |
| `mlflow/store.py::ensure_experiment_overview` | seam `mlflow.experiment.ensure_overview`; `experiment/lifecycle.py::create` wraps it | §Q2 |
| (new) `experiment/lifecycle.py::create` | — | explicit bootstrap |
| (experiment-scope delete wrapper) | `experiment/cleanup.py::delete` | thin → `project.cleanup` (03) |
| `inspect/views.py::leaderboard` (+ `LeaderboardRow`) | `experiment/views/leaderboard.py` → `LeaderboardData` | `mlflow_url` derived, not stored |
| `inspect/views.py::compare` | `experiment/views/compare.py` → `ComparisonResult` | composes `trial.show_trial` per run; deltas over first two |
| `inspect/views.py::experiments` | `mlflow.project.list_experiments` (seam) + view enrich | checklist 357 |
| `loop_context/queries.py::top_n_by_metric` | `mlflow.experiment.top_n_by_metric` (seam) | seam |
| `loop_context/queries.py::recent_failures` | `experiment/views/queries.py` (= `list_trials(status=FAILED)`) | view helper |
| `loop_context/queries.py::strategies_attempted` | `experiment/views/queries.py` (aggregate `list_trials`) | view helper |
| `loop_context/queries.py::runs_using_strategy / runs_in_metric_band` | **deferred** (no placeholder file) | no caller (§Q4) |
| `loop_context/queries.py::show_trial` | DROP → `mlflow.trial.get_details` (02) / `trial.show_trial` (10) | superseded |
| `loop_context/__init__.py::experiment_id` | DROP | numeric id is seam-internal (§Q5) |
| `loop_context/summary.py::{load_mlflow_context,build_summary,build_summary_from_context}` | `experiment/views/summary.py` | compose seam; `learning_counts` dropped (§Q7) |
| `mlflow/store.py::{write_learning_cache,…}` | STAY in `automl_legacy/` | out of scope |

(`inspect/views.py::show_trial` / `load_model` / `load_data_snapshot`,
`loop_context/proposer_packet.py`, `mlflow/store.py::get_context` → sub-specs 10/11/data;
not 09's rows.)

---

## 15. Cross-doc reconciliations (precedence: 00 > sub-spec > checklist)

1. **Query homes (checklist ↔ 00/02).** Raw searches at seam; view helpers for
   demand-backed aggregations; no-caller queries deferred. **00 wins** (§Q3).
2. **`recent_failures` / `compare` demand (README ↔ 00 §11.1).** Both are real — 00
   §11.1 verb catalog + live consumer beat README prose (§Q3).
3. **`Experiment` vs `ExperimentOverview` (00 §12 ↔ 02 §6.2.1).** One type, facade-aliased
   (§Q1).
4. **`archive` Tier-2 export (00 §8.6).** Deferred — no implementation/caller (§Q2).
5. **Diagnostics placeholder (00 §7 ↔ 02 §4).** Zero files; deferred queries recorded
   (§Q4).
6. **`LeaderboardData` / `ComparisonResult` typing.** Kept typed (human chose typed
   wrappers); already in 00 §13.1. View *containers* typed; `summary` stays a dict (§Q6).

---

## 16. Review log

### Round 1 — three agents (fresh-eyes design / codebase comparison / cross-spec consistency)

Reviewed against 00/02 + current code; `pending/` draft deliberately excluded.

**Applied (real findings):**
- **Predecessor linkage was over-claimed.** 09 had asserted `ensure_overview` carries
  `predecessor_experiment_overview_run_id` and that `ExperimentOverview` records it. Code
  trace: the tag is **write-only** (never read) and already *retired* in `data/pipeline.py`.
  → claim **removed**; `create` takes no predecessor param; `02` unchanged for it (§Q2).
- **`find_prior_experiment` ownership.** It is the real cold-start feature (independent of
  the dead tag), with a single consumer (the proposer packet). → assigned to
  `agent/proposer_context` (sub-spec 11), not `experiment/` (§Q2 note).
- **`mlflow_url` seam helper.** Referenced by 07/08/09 but absent from 02's public surface;
  the helper already exists as `store.py::run_url`/`artifact_url`. → carry-back to 02 to
  expose `mlflow.client.run_url`/`artifact_url` publicly (§8.1, §18).
- **`load_mlflow_context` dependency direction.** Today it delegates to
  `gather_proposal_context`, which now lives in `agent/`. `experiment/` must not import
  `agent/`. → `load_mlflow_context` composes seam reads directly (§8.1).
- **`recent_failures` lost its `training_origin` filter** (live consumer passes `"automl"`)
  → re-added to the signature (§8.1).
- **`experiments()` was unnamed + CLI destination ambiguous** (00 §11.1 said raw
  `list_experiments`; 09 described enrichment) → named `experiments() -> list[dict]` with
  key set; CLI calls the enriched view (carry-back to 00 §11.1); name-filter rules → seam
  (carry-back to 02) (§8.1).
- **`compare` output key rename** `"metrics"` → `metric_deltas` made explicit (§8.1).
- **`strategies_attempted`** aggregates `list_trials()` with **no** status filter (all
  trials, as today) (§8.1).
- **`data_context_keys`** was missing from the summary preserved-key inventory → added (§9).
- **`FailureRecord`** still listed in 00 §13.1 though 09 kills it → carry-back to remove (§18).
- **`_imports.py` → `_import.py`** filename typo (§13).

**Flagged as false-positive / no-change:**
- "`TrialStatus` 'success'/'failed' casing hides historical runs" — not a concern:
  `feedback_no_back_compat` means new code does not read old runs; the enum *values* are
  02's contract (§9 already states the clean cut).
- "`leaderboard` lost its `dry_run` param" — by design: `dry_run` is a session-level
  container (`feedback_dry_run_is_a_container`); routing comes from `session` via the seam
  (clarification added, not the param — §8.1).
- "`ComparisonResult.runs: list[dict]` is mutable inside a frozen dataclass" — pervasive
  doc-wide pattern; element type intentionally deferred to sub-spec 10. Acceptable.

**Decomposition sweep (highest-value cross-spec output):** the consistency agent produced
an exhaustive line-by-line list of stale `experiment/trial/` / `experiment/proposal.py` /
`experiment.agent_*` / `experiment/views/proposer_context` references across 02–07 +
living docs. Applied as the "prior-spec sweep" (§18; user-approved).

---

## 17. Open decisions for human review

All resolved in the interview (2026-05-25). Recorded for traceability:

1. `Experiment` collapses into `ExperimentOverview` (one type, aliased). **RESOLVED — §Q1.**
2. `lifecycle.create` + both ensures kept (explicit bootstrap + lazy safety net).
   **RESOLVED — §Q2.**
3. Query-home split (raw→seam, helpers→views, no-caller deferred; `recent_failures` +
   `compare` in-scope). **RESOLVED — §Q3.**
4. Zero diagnostics placeholder files. **RESOLVED — §Q4.**
5. Drop public `experiment_id()` helper. **RESOLVED — §Q5.**
6. Typed `LeaderboardData` / `ComparisonResult`; `summary` stays a dict. **RESOLVED — §Q6.**
7. Drop `learning_counts` from summary (clean cut). **RESOLVED — §Q7.**

---

## 18. Proposed carry-backs

**Already applied during the foundation pass (this session):**
- `00`: full decomposition (§5/§6/§7/§8.6–§8.8/§11.1/§12/§13.1/§16/§17, Appendix A);
  §8.6 Tier-2 exports reflect §11 here (`archive` removed; `Experiment` = `ExperimentOverview`).
- `02`: Trial type homes (`TrialSummary`/`TrialDetails`/`ParentExperimentRef` →
  `trial/types.py`; `TimingReport`/`TrialManifest` → `trial/metadata.py`);
  proposer-context home → `agent/`.

**APPLIED 2026-05-26 (batched into sub-spec 10's closeout — verified by grep, see below).**
Every item in this list is now in place: `run_url`/`artifact_url` on the seam surface (02
§6.4 client.py); `diagnostics.py` placeholders zero-filed (02); `list_experiments` returns
logical ids (02 §6.2 line ~255); `ensure`/`ensure_overview` contract documented (02 §6.2.1 —
the verify item, resolved as **no restore**: archived stays archived, `created_by` on create,
collision → `StorageError`; user direction 2026-05-26);
`FailureRecord` removed from 00 §13.1; `automl experiment list` → `experiment.views.experiments`
(00 §11.1 line ~478); the decomposition sweep across 02–07 + living docs applied (the lone
straggler, README's 04-summary `experiment/checks.py::proposal_schema`, fixed → `agent/checks.py`).

*To `02` (DONE):*
- **Add `run_url` / `artifact_url` to the public seam surface** (e.g. `mlflow/client.py`)
  — the helpers already exist as `store.py::run_url`/`artifact_url`; 07/08/09 all derive
  MLflow links via them. (NOT a predecessor change — that tag is dead, §Q2.)
- Drop the `experiment/diagnostics.py` + `mlflow/experiment/diagnostics.py` placeholder
  references (§Q4 — zero files); `recent_failures`/`strategies_attempted` are in-scope in
  `experiment/views/queries.py`, not deferred.
- `list_experiments` should return **logical** experiment_ids (apply the name-filter rules
  — prefix-match, exclude `overview`, exclude nested — at the seam).
- *(verify — DONE)* the seam's `ensure`/`ensure_overview` contract (02 §6.2.1): sets
  `created_by` on first creation; **no soft-delete restore** (`_activate_experiment` dropped —
  user direction 2026-05-26; a same-name collision surfaces as `StorageError`).

*To `00`:*
- Remove `FailureRecord` from §13.1 (09 kills it — no consumer; §12).
- §11.1: `automl experiment list` library destination → `experiment.views.experiments`
  (the enriched view), not the bare `mlflow.project.list_experiments` (§8.1).

*The decomposition sweep (Round-1 consistency agent; user-approved) — stale references
the split makes wrong, across the prior specs + living docs:*
- `02`: `experiment/proposal.py` → `agent/proposal.py`; `experiment.agent_timeline.*` →
  `agent.timeline.*`; the hook-stub example `automl.experiment.agent_timeline` →
  `automl.agent.timeline`.
- `03`: `experiment/trial/cleanup.py` → `trial/cleanup.py`; `experiment.trial.delete` →
  `trial.cleanup.delete`.
- `04`: `experiment/checks.py::proposal_schema` → `agent/checks.py`; `experiment/proposal.py`
  → `agent/proposal.py`; the "sub-spec 09 (Experiment): confirm proposal" note → sub-spec 11.
- `05`: proposer-context aggregator home `experiment/views/proposer_context.py` →
  `agent/proposer_context.py`, sub-spec **11** territory.
- `06`: `experiment/proposal.py` → `agent/proposal.py`; `experiment/checks.py::proposal_schema`
  → `agent/checks.py`; `experiment/trial/` (package_model / load_model) → `trial/`, sub-spec **10**.
- `07`: proposer-packet reader home → `agent/proposer_context.py` (sub-spec 11).
- `migration-checklist.md`: `cli/inspect.py` row split (`show-trial`/`load_model` → `trial.show`);
  `cli/loop_context.py` → `agent.proposer_context`; `cli/run_loop.py` → `agent.build_launch` /
  `agent/launch.py`; `LeaderboardRow` → DROP (replaced by `LeaderboardData`); `loop_context.experiment_id`
  → DROP; `loop_context/queries.py` raw searches → seam (split the 4-way row per §Q3);
  `gather_proposal_context` / `find_prior_experiment` → `agent/proposer_context.py` (sub-spec 11);
  `automl/trial/*` ops → `trial/` (top-level); `propose/*` (SLUG_RE, fields, validate) → `agent/`.
- `open-questions.md`: `ParentExperimentRef`/`TrialDetails` home → `trial/types.py`
  (was `experiment/views/types.py`); proposer-context → `agent/`, sub-spec 11.
- `README.md`: §108 `experiment/trial/cleanup.py` → `trial/cleanup.py`; §132/§228 diagnostics
  placeholder → zero-file + `recent_failures`/`compare` **in** scope; §183/§227 sub-spec table
  row — 09 now covers Experiment+views only (Trial→10, Proposal+agent→11); flip 09 row + add
  "What's done" entry; note the 09→09/10/11 split.
