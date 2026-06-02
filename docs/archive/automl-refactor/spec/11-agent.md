# Sub-spec 11 — `agent/` (the agentic loop)

**STATUS: APPROVED 2026-05-27.** (Design interview + three-agent review [fresh-eyes
+ codebase-gap + coverage/cross-spec] + fixes applied.)

The `agent/` domain — the last of the three peers the old `experiment/`
mega-domain was split into (09 = `experiment/`, 10 = `trial/`, 11 = `agent/`).
It owns the **agentic loop**: the agent **launcher** (was `cli/run_loop.py`),
the agent **timeline** reconciliation (was `hooks/agent_timeline.py`), the
**Proposal** contract (proposer↔coder handoff) + its `proposal_schema` check,
and the **proposer-context** assembly (the input packet the proposer reads).

It does **not** cover trial execution (→ `runner/`, sub-spec 08), trial
lifecycle/reads (→ `trial/`, sub-spec 10), or the cross-trial views (→
`experiment/`, sub-spec 09). The proposer↔coder *sequencing* stays
**LLM-driven** — `agent/` defines, launches, and observes agents; it is not a
state machine (00 §8.8, CLAUDE.md "loop is LLM-driven").

---

## 1. Scope — relocate, not redesign

Per 00 §8.8 + §17.11, v1 is a **straight relocation** of today's proposer↔coder
loop into the library, with three boundary changes applied throughout:

1. **Route MLflow writes through the 02 seam** (the §13.4 invariant — domain
   code never imports `mlflow` directly).
2. **Apply the 01 session convention** (`session: Session | None = None`,
   resolved via `automl.session()`).
3. **Collapse `run_mode`/`route_namespace`** to `session.dry_run` (10 §7.2).

**No** driver abstraction, **no** agent/role registry, **no** multi-agent
orchestration — those are the §17.12 forward-looking axes, built on demand. The
timeline reconciliation algorithm is **ported verbatim** (00 §15.1).

A handful of **cheap, low-risk improvements ride along** with code we're already
touching (§8) — explicitly *not* a refactor; each is called out.

---

## 2. The `agent/` shape (00 §7 holds — no new files)

```
agent/
├── __init__.py            ← Tier-2 exports
├── proposal.py            ← Proposal (frozen dataclass) + DISALLOWED constant + field roster
├── checks.py              ← proposal_schema (registered; consumed by validate.proposal)
├── proposer_context.py    ← gather_proposer_context (dict packet) + find_prior_experiment (internal)
├── launch.py              ← build_launch + LaunchSpec + ClaudeRole
└── timeline.py            ← handle_event + publish (Tier-2) + reconciliation engine (internal)
```

**Tier-2 exports (`agent/__init__.py`):** `Proposal`, `build_launch`,
`handle_event`, **`publish`**, `gather_proposer_context`.
- `proposal_schema` is **framework-facing** — imported by `validate/targets.py`
  (04), not a user-facing export.
- `find_prior_experiment` stays **internal** to `proposer_context.py` — no
  external caller (`feedback_extension_points_follow_demand`).

**Outbound deps (00 §8.8):** `project` (Session / model routing / allowed-deps),
`experiment.views` + `trial` (`proposer_context` composes their reads), `data`
(active Dataset / profile via seam), `mlflow.project`/`experiment`/`trial` +
`mlflow._routing` (path helpers), `utils.io.gcs`, `utils` (`SLUG_RE`), `validate`
(`checks.py`). **No direct `runner` import** — `runner.run_trial` is reached only
transitively via `trial.promote` (a `trial/` concern); nothing in `agent/`
imports `runner` (the launcher spawns the loop as a subprocess, it does not call
the runner in-process). 00 §8.8's "runner" outbound entry is this transitive
reach, not an import edge.

**Inbound:** CLI verbs (`experiment run` → `build_launch`; `experiment
proposer-context` → `gather_proposer_context`), `hooks/` stub
(→ `timeline.handle_event` / `timeline.publish`).

**§17.11 extensibility stays deferred** — `build_launch` remains the single
place the claude-specific invocation is assembled (the future-driver seam).

---

## 3. Proposal contract — `agent/proposal.py` (Q1, Q2)

### Q1 — `Proposal` becomes a typed object

**Current behavior.** No `Proposal` class exists. `propose/schema.py` holds three
field-name lists (`REQUIRED_FIELDS`/`OPTIONAL_FIELDS`/`DISALLOWED_FIELDS`) + the
legacy `Issue`/`ValidationReport` dataclasses (04 moves those into
`validate/base.py`). `propose/__init__.py::validate()` operates on a raw
`dict[str, Any]`; the proposal lives its whole life as JSON.

**Decision (A).** Introduce a **frozen `Proposal` dataclass** in
`agent/proposal.py` with `schema_version: int`, `from_dict` (strips unknown
keys), and `to_dict` — matching the schema strategy locked across
02/04/05/06/07/10 and fulfilling 00 §5 ("Proposal is a class") + §8.8 (Tier-2
export). Flow: proposer LLM writes JSON → `proposal_schema(dict)` validates the
untrusted dict (§4) → `Proposal.from_dict(dict)` produces the typed object
`trial.create` consumes (10). **Validation stays dict-shaped** (you validate
untrusted JSON); the *consumed* form is typed.

### Q2 — Field roster + encoding + `schema_version`

**Decision (A).** The **dataclass is the single roster source.** The check (§4)
derives its name sets by introspecting `dataclasses.fields(Proposal)`: required =
no default, optional = defaults to `None`. `DISALLOWED = ("parent_id",)` stays as
one explicit constant beside the dataclass (retired fields can't be dataclass
fields, but rejecting them yields a useful error). Per-field *format* rules
(slug regex, non-empty-list, `seed_hint` enum) live in the check — never
expressible as lists anyway.

```python
@dataclass(frozen=True)
class Proposal:
    schema_version: int                        # = 2
    slug: str
    strategy: str
    hypothesis: str
    implementation_plan: list[str]
    constraints: list[str]
    required_dependencies: list[str]
    rationale: str | None = None
    evidence: list[str] | None = None          # optional fields are loosely typed —
    data_checks: list[str] | None = None       # the proposer-prompt contract, not
    risk_notes: str | None = None              # check-enforced; precise typing is
    seed_hint: str | None = None               # a plugin-layer concern
    required_preprocessing: list[dict] | None = None   # NEW (06) — see below
```

- **`schema_version` stays 2.** Adding `required_preprocessing` is additive (the
  locked additive-only rule → no bump); no-back-compat means old persisted
  proposals aren't read regardless; the proposer/coder prompt contract already
  says "v2."
- **`required_preprocessing`** (06 carry-in): `list[dict] | None`, mirroring
  `model.describe_required_transformers(session) -> list[dict]` (name/type/import
  path/columns). Proposer populates it from the helper; coder reads it; **the
  check *allows* but does NOT re-enforce** — the single enforcement point is the
  model gate (06 §Q3.3). Informational/handoff + provenance only.
- **`SLUG_RE`** is imported from `utils/` (10 carry-in — moved off
  `agent/proposal.py` to avoid a `trial → agent` cycle). Both the dataclass side
  and the check import it from there.

---

## 4. `proposal_schema` check — `agent/checks.py` (Q3)

**Current behavior.** `propose.validate(*, proposal: dict, allowed_dependencies:
list[str])` takes the allow-list explicitly. The three callers supply it three
ways: `automl propose validate` (CLI flags), `cli/trial.py:76` (passes the
proposal's **own** `required_dependencies` → the `dep_not_allowed` check is a
tautology enforcing nothing), and the `proposal_checks.py` adapter (04 deletes
it). The canonical allow-list source is `project.dependencies.allowed_dependencies`.

**Decision (A) — session-resolved allow-list.**

```python
# agent/checks.py
def proposal_schema(proposal: dict, *, session: Session | None = None) -> list[Issue]
```
- Resolves the allow-list internally via
  `project.dependencies.allowed_dependencies(session)`; returns canonical
  `validate.Issue` (04). Drops the explicit `allowed_dependencies` param **and**
  the `--allowed-deps-file` / `--allowed-dependencies-json` CLI flags.
- **Fixes the `trial.create` tautology** — the allow-list now checks proposals
  against the project's *real* installed deps everywhere, including at
  trial-create. Collapses three supply paths into one; matches the session
  convention + the 09 "session-based" carry-in.
- Takes the raw `dict` (validate untrusted JSON), then the caller does
  `Proposal.from_dict` on pass (§3 flow). Absorbs the legacy `validate()` logic
  + the deleted `proposal_checks.py` adapter (04). **Allows-not-enforces**
  `required_preprocessing` (06).
- **CLI:** `automl validate proposal` (the 04 dedupe of `automl proposal
  validate`) keeps `--json` + `--output`; loses the deps flags.

---

## 5. `proposer_context` — `agent/proposer_context.py` (Q4, Q5, metric reconciliation)

### Q4 — Returns a `dict`

**Decision (A).** `gather_proposer_context` returns a free-form `dict` — same
call 09 made for `experiment.views.summary`, same reason: it's a heterogeneous,
agent-facing JSON aggregate whose only consumer is the prompt renderer + the
LLM. No field-level programmatic consumer, no persisted-schema-evolution concern
(recomputed every turn, never stored as a typed artifact). The composed *pieces*
stay typed (`LeaderboardData`, `TrialSummary`, `EvalResult`, the data types); the
envelope is a dict.

`gather_proposer_context` is a **composer** — it calls `experiment.views.*`,
`trial` reads, the data seam, and its own `find_prior_experiment`, doing **no raw
MLflow searches** (those moved to the seam per Appendix A).

### Q5 — Packet roster (drops + reshapes)

The merged legacy packet (`gather_proposal_context` + `get_context`) has ~20 keys,
several redundant or out-of-scope. The rebuilt dict:

| Key | Disposition | Source |
|---|---|---|
| `project` (slim metadata dict: name/package/config_path) | keep | `project` (session) |
| `project_name`, `experiment_id`, `mlflow_experiment_id` | keep | session + seam |
| `metric` | keep | session/config |
| `higher_is_better` | keep (**additive** — see note) | config/the eval `Metric` direction |
| `project_instructions` | keep | `instructions_path.read_text()` |
| `overview` (slim run-ids + names) | keep | `mlflow.project`/`experiment` |
| `leaderboard` | keep | `experiment.views.leaderboard` (09) |
| `human_trials` | keep | `experiment.views.leaderboard(training_origin="human")` |
| `recent_failures` | keep | `experiment.views.queries.recent_failures` (09) |
| `strategies_attempted` | keep | `experiment.views.queries.strategies_attempted` (09) |
| `trial_count` | keep | derived from summaries |
| `prior_experiment` (cold-start only) | keep | `find_prior_experiment` (below) |
| `data_context` (active **dataset** + profile + `dataset_usage`) | keep, **reshaped** | data seam + `mlflow.project` |
| `top_trials` (== `leaderboard`) | **DROP** | redundant key |
| `experiment_learnings`, `project_learnings` | **DROP** | learning subsystem out of scope — never written |
| `artifact_uris`, `artifact_errors` | **DROP** | vestigial (fed the learnings reads) |
| `primary_eval` per-row enrichment | **DROP** | see metric reconciliation |

- **Learnings out.** The project-level learning subsystem is out of scope (stays
  in `automl_legacy/`, never written), so the packet can't surface what doesn't
  exist. Removes `get_context`'s `_list_artifact_paths` / `_try_read_json_artifact`
  learning reads + the `artifact_errors` collection.
- **`primary_eval` enrichment dropped** — legacy did 2 artifact reads *per
  leaderboard row* (`eval/manifest.json` + `eval/<label>/report.json`) reaching
  into eval internals (`_primary_eval_for_run` + `validate_eval_label`). The
  `TrialSummary` already carries `primary_metric_name` + `primary` (10), which is
  what the proposer ranks on; eval-substrate detail is available on drill-down via
  `trial.show_trial → TrialDetails.evaluations` (10). Re-add as a `leaderboard`
  option on real demand.
- **`data_context` reshaped** to 05 vocabulary: `active_data_snapshot` →
  `active_dataset`; drop `prepare_event_id` (05 Q4 audit); `snapshot_identity_hash`
  → the composite `identity_hash`; **profile sub-object shape per 05 Q5** (data
  card / observations / charts URLs), sourced from the **project-overview** run
  (05 Q5 moved profiles there). The legacy `data_context.snapshot_usage` map
  (all-dataset usage counts, `store.py:1020`) is kept, reshaped to **`dataset_usage`**
  (cold-start signal — how many trials used each materialized dataset); the
  per-active-dataset `trial_usage_count` is derived agent-side from summaries.
- **`project` metadata + `higher_is_better` notes.** The legacy packet carries a
  `project` *metadata dict* (`_project_metadata` — name/package/config_path;
  `instructions_path` is already surfaced as `project_instructions`); kept slim.
  `higher_is_better` is **additive** to the packet — legacy only set it in the
  `summary.py` path (hardcoded `True`), never in `gather_proposal_context`; the
  rebuilt packet sources it from config / the eval `Metric` direction (no longer
  hardcoded), so the missing-metric ranking (below) sorts correctly for
  lower-is-better metrics.
- **`prior_experiment`'s role narrows** (learnings dropped): from "pointer to prior
  *learnings*" to "pointer to a prior experiment the proposer can *explore*
  (leaderboard/trials) for cold-start." No code change — narrower value.

### Metric ranking — reconciliation across heterogeneous / switchable primaries

A single experiment holds many trials and the user may **switch the experiment's
primary metric over time** (AUC early, revenue later). The ranking model:

- An experiment has **one *current* primary metric = `config.primary_metric`**
  (mutable). The sort metric is an **explicit parameter** on everything that ranks
  (`experiment.views.leaderboard`, `human_trials`, `gather_proposer_context`),
  **defaulting to `config.primary_metric`** (resolved from session). The CLI verbs
  (`experiment leaderboard` / `proposer-context`) take an optional `--metric`
  override falling back to that default.
- The leaderboard / proposer-context rank **all** trials by the *current* metric.
  Each trial's own `eval.primary_metric` (what it optimized when it ran) is
  **provenance/display only** — decoupled from how the experiment ranks today.
- A trial that never computed the current metric is **reported as missing** — the
  leaderboard returns ranked trials that have it, plus a callout *"x/n trials not
  scored on `<metric>`"*. **No re-eval hook / no back-fill machinery** (re-eval is
  §17.5, forward-looking).

This forces carry-backs (§9) to 09/02/07/08 — the *foundation* (07 `EvalResult` =
`metrics` + `primary` pointer; 09 `leaderboard(*, metric, …)`) already supports it;
the gap is the **cross-trial sort key** (the bare `<metric>` column at
`eval/evaluate.py:596` is logged only for each trial's own primary, so ranking must
address the metric by a name present across the whole locked set —
`<label>.<metric>`, line 588).

### `find_prior_experiment` — cold-start, agent-owned (09 carry-in)

Lives in `proposer_context.py` (an agent concern, 09). Fires only when the current
leaderboard is empty. **Cheap win #1:** sort candidates by experiment
**`creation_time`** (from `search_experiments`), not the legacy lexicographic
`experiment_id`-string descending (which mislabels "highest-sorting DS-chosen name"
as "most recent").

---

## 6. Launcher — `agent/launch.py` (Q6)

Relocate of `cli/run_loop.py` (00 §8.8 "relocate, not redesign").

**Signature (session convention):**
```python
def build_launch(*, session: Session | None = None,
                 automl_args: list[str], max_budget_usd: str, output_format: str,
                 claude_bin: str = "claude",
                 permission_mode: str = "bypassPermissions") -> LaunchSpec
```
- Drops `project_root` + `project` params. Model routing reads
  `session.config.models` (the `ProjectConfig` derived property, 01 — `{manager,
  proposer, coder}` each `{model, effort}`). `cwd` / `--add-dir` / `--plugin-dir` /
  `AUTOML_PROJECT_ROOT` derive from session's project root; ambient resolved via
  `automl.session()`.
- **Build/execute split preserved** (00 §8.8 exports `build_launch`, not a
  `run_loop`): `build_launch` is a pure builder returning `LaunchSpec`; the thin
  `cli/` wrapper for `automl experiment run [<id>]` does the `subprocess.run`. The
  optional `[<id>]` ports as a session override (`update_session(experiment_id=…)`)
  before launch.
- **Ported verbatim into internals:** `LaunchSpec` + `ClaudeRole` (frozen
  dataclasses), `_role_settings`/`_model_settings`, the `agents/*.md` frontmatter
  parsing, `--agents` JSON construction, `_normalize_automl_args`, the
  `/brigit-automl:automl <args>` slash command + env injection.
- **Env injection:** `AUTOML_SESSION_ID`/`CLAUDE_SESSION_ID`, `AUTOML_PROJECT_ROOT`,
  and **`AUTOML_INHERIT_DRY_RUN`** (new — see §7/§9) so the spawned subprocess tree
  shares one dry_run value. Session-id minting unchanged (`env AUTOML_SESSION_ID
  or CLAUDE_SESSION_ID or uuid4()`).
- **Cheap win #2:** under the session convention the CLI wrapper's double
  project/arg resolution (legacy `resolve_project_context` + re-derive + a second
  `_normalize_automl_args`) collapses — resolve the session once, `build_launch`
  reads it. CLI-wrapper-internal, low risk.

**Plugin-layer boundary (flag, don't redesign):** the `automl run` → `automl
experiment run` rename is a CLI-verb change; the spawned slash command still targets
the **skill** (`/brigit-automl:automl …`), unchanged. The exact inner skill-arg
contract (the `run` subcommand the skill prose parses) is plugin-layer.

---

## 7. Timeline — `agent/timeline.py` + thin `hooks/` stub (Q7, Q8, Q9)

Relocate of `hooks/agent_timeline.py` (1955L). The reconciliation algorithm is
**ported verbatim** (00 §15.1); the decisions are at the boundaries.

### Q7 — Library entry surface + stub split

**Trigger map (grounded in `hooks.json`):** `hook-event` is the only
hook-triggered subcommand (SubagentStart/Stop); on **coder-stop** it auto-publishes
that trial inline. `publish` is invoked by the **automl skill at end-of-loop**
(`render_context.py::safe_commands.timeline_publish`). `summarize` has **no
production caller** (its reconciliation is reused internally by `publish`).

**Decision (A) — two library entries, reconciliation internal:**
```python
def handle_event(payload: dict, *, session: Session | None = None) -> dict
def publish(*, session: Session | None = None, session_id: str) -> dict
```
- **`publish_mlflow` dropped** (review). The legacy `--publish-mlflow` flag is
  *always* passed by the one production caller (`render_context.py:387`), so the
  False path (stage + GCS, skip MLflow) is dead config. `publish` always does the
  full publish; the skill stops passing `--publish-mlflow`.
- **One source of truth** = the append-only event log
  `agent_timeline.jsonl`; **one reconciliation engine** (`_summarize_hook_events` +
  ~40 helpers, ported verbatim as internal functions). `handle_event` appends the
  event and, on coder-stop, flushes *that trial's* artifacts to *its* run; `publish`
  reconciles the *whole* log into the *session* summary on the *overview* run. Two
  triggers/scopes over the same data — the existing per-event-vs-end-of-loop split,
  named as functions instead of argparse branches.
- **Drop the standalone `summarize` subcommand** (no production caller; tests target
  the internal reconciliation function — impl-time touch-up).
- The stub `hooks/agent_timeline.py` keeps `hook-event` + `publish` only: parse argv
  (`--project-root`, subcommand), read stdin payload, bootstrap session, delegate.
  Thin. `hooks.json` unchanged.
- **Carry-back to 00 §8.8:** agent Tier-2 exports add `publish`.

### Q8 — Seam-routed persistence + drop the manifest-merge

**Routing (mandated — 00 §8.8 + §13.4):** 02 already exposes the needed post-hoc
writers (designed for agent reports — 02 §3.7/§6.2.4):
- JSON reports/manifests/session-summary: `client.log_artifact(file)` →
  `mlflow.trial.log_json(run_id, "agent/…", payload)` (per-trial) and
  `mlflow.experiment.log_json("agent/sessions/…", payload)` (session summary →
  overview). `client.log_metric(run_id, "agent.*", …)` → `mlflow.trial.log_metric`.
- `store.ensure_experiment_overview` → `mlflow.experiment` overview-ensure (02/09).
- Gzipped transcripts + raw bytes (not `log_json`-able) → GCS via **`utils.io.gcs`**,
  paths from **`mlflow/_routing.py`** — **dropping the `importlib`-loaded per-project
  `gcs_paths.py` contract and the raw `google.cloud.storage` calls** (cheap cleanup,
  kills the dynamic-module hack). No 02 carry-back needed for the write API.
- **Agent-events GCS prefix (review gap).** Legacy `_trial_agent_events_object_prefix`
  reads a *runner-assigned* `gcs.agent_events_prefix` off the trial manifest to decide
  where to upload agent events — a runner→timeline path handshake via the manifest.
  Resolution: **`mlflow/_routing.py` becomes the single source for the agent-events
  GCS prefix**, computed deterministically from `(session, run_id)`; both the runner
  (when it writes) and the timeline (when it uploads) call the *same* helper, so they
  agree by construction — the manifest-read handshake is **dropped** (it loses
  nothing; the prefix is derivable, not stored). New carry-back to 02/`_routing.py`
  (§9 #6).

**Drop the manifest-merge (A).** Legacy `_update_root_manifest_with_agent_artifacts`
downloads the trial's `manifest.json`, merges agent entries, re-uploads. 10 slimmed
`TrialManifest` to the runner's artifact TOC and established its only consumer is
`cleanup.py`, which deletes by GCS-prefix + MLflow-run **tree** (03) — so it never
needs agent entries enumerated. The hook stops mutating the trial's `TrialManifest`;
it writes its own standalone `agent/manifest.json` (the agent-artifact TOC). Removes
the download+merge+reupload round-trip; loses nothing.

### Q9 — Route + dry_run from session (kill the sys.argv hack)

**Current behavior.** Route resolved three ways (explicit `--route`, session-lock
lookup, config); `--dry-run` parsed off the hook's *own* `sys.argv` (never matches on
`hook-event` → hook events always treated non-dry-run); `route_namespace=""`
everywhere; `dry_run/project/exp` route segments string-parsed.

**Decision (A) — session + the `AUTOML_INHERIT_DRY_RUN` env (transport-only).**
- There is **one** `dry_run` (the session container). It lives in a contextvar (01)
  which **cannot survive `subprocess.run`**, so the parent encodes it into the env
  and the child decodes it back into *its own* `session.dry_run`.
  **`AUTOML_INHERIT_DRY_RUN="1"/"0"`** is that carrier — **transport-only, never an
  independent source of truth** (renamed from the misleading `AUTOML_DRY_RUN`, which
  read as a system-wide *mode*). The launcher (§6) injects it; the runner reads it
  (08); the hook reads it.
- The **stub** bootstraps the session from `--project-root` (→ `use_project`, giving
  `project_name` + `experiment_id` from config) and reads `AUTOML_INHERIT_DRY_RUN`
  into `session.dry_run`. Library functions take `session`; **route + GCS/MLflow
  paths derive from session via `mlflow/_routing.py`** (the `dry_run/` prefix
  conditional on `session.dry_run`).
- **Killed:** the `sys.argv` parse, the `_route_*` / `_project_name_from_route` /
  `_experiment_id_from_route` string-parsing, the session-lock route lookup
  (`_active_route_from_session_lock` — session is authoritative), the agent's own
  `route_namespace=""` handling + route-string parsing, and the `--dry-run` flag on
  `publish` (env carries it). **(Scope note, final pass 2026-05-27:** "killed
  route_namespace" here means the *timeline's* dead local usage — it always passed `""`
  and parsed route strings. The **seam's** bound field *survives*, renamed `namespace`,
  and is now a wired first-class isolation dimension fed by the top-level `--namespace`
  flag; the agent simply gets its paths from `mlflow/_routing.py` — which applies the
  `namespace` + `dry_run` segments — instead of parsing them itself. See open-questions
  → `route_namespace`/`namespace` resolution + 02 §13 item 5.**)**
- **Plugin ripple:** the skill's `timeline_publish` invocation
  (`render_context.py`) sheds three args — `--dry-run` (now in the inherited env),
  `--route` (route derives from session, no param), and `--publish-mlflow` (dropped,
  publish always writes MLflow). See §9 plugin carry-forwards + caller/test list.

### Granularity — one file (relocate-verbatim)

`timeline.py` stays **one file** (00 §7/§8.8). The relocate already touches it
substantially (seam-routing, killing sys.argv/route-parsing); a cohesive split would
be reshaping a verbatim port, contradicting the relocate-only mandate. A split is a
deferred follow-up if it proves unwieldy.

---

## 8. Cheap wins applied (riding along — not a refactor)

1. **`find_prior_experiment` orders by `creation_time`**, not lexicographic
   `experiment_id` string (§5). Fires only on cold-start; pure correctness.
2. **CLI launcher double-resolution collapses** under the session convention (§6).
3. **Drop the dynamic `gcs_paths.py` `importlib` contract** + raw
   `google.cloud.storage` — GCS via `utils.io.gcs` + `mlflow/_routing.py` (§7 Q8).
4. **Drop the manifest-merge** — agent writes its own `agent/manifest.json` (§7 Q8).

Deliberately **not** done (considered, out of scope): deleting the bare-`<primary>`
metric log (has a live consumer — `store.py:432` reads it for `TrialSummary.primary`);
simplifying `_normalize_automl_args` (plugin-coupled); loop-stop hard-enforcement
(deliberate LLM-driven loop, `docs/to-do/loop-state-machine.md`); splitting
`timeline.py` (relocate-verbatim mandate).

**Review-applied simplification:** `publish_mlflow` param dropped (always-True dead
config — §7 Q7).

**`SLUG_RE` stays in `utils/`** (per 10). It's duplicated today in `propose/__init__.py:15`
*and* `trial/creation.py:14` — both `agent/` (proposal-slug validation) and `trial/`
(trial-folder naming) need it. The cycle only bites if it lives in `agent/` (forces
`trial → agent`); `utils/` is a neutral shared leaf both import, de-duping the two
copies. The pattern is a generic snake_case identifier validator (same shape as
`PROJECT_NAME_RE` / `AUGMENTATION_NAME_RE`) — no AutoML semantics in the regex itself
— so `utils/` is the right home for the shared primitive.

---

## 9. Carry-backs (record in open-questions; apply at closeout)

| # | Target | Change |
|---|---|---|
| 1 | **00 §8.8** | Add `publish` to the agent Tier-2 exports (handle_event alone undersold the timeline surface). |
| 2 | **09** (`experiment.views.leaderboard` + `LeaderboardData`) | Default `metric` resolves from `config.primary_metric` (not the hardcoded `"auc"`); `LeaderboardData` gains an unscored-count (*"x/n trials not scored on `<metric>`"*). |
| 3 | **02** (`mlflow.experiment.top_n_by_metric`) | Sort/lookup must address the metric by a **cross-trial-stable name** (the primary-label-namespaced `<label>.<metric>`, logged for the whole locked set — `eval/evaluate.py:588`), so "missing" means *genuinely uncomputed*, not "not this trial's primary." Sorting on the bare `<metric>` column over-reports missing. |
| 4 | **07/08** | Confirm the locked metric set stays logged under stable namespaced names (already true). Note: the bare-`<primary>` log (`evaluate.py:596`) is per-trial convenience, not the cross-trial sort key. |
| 5 | **08 + cross-cutting** | Rename `AUTOML_DRY_RUN` → **`AUTOML_INHERIT_DRY_RUN`** (transport-only; the launcher sets it, the runner + hook read it). Remove the now-obsolete metadata-conflict check (`runner/_execute.py:289`) — already implied by 10 §7.2's `run_mode` collapse (TrialMetadata no longer carries dry_run to conflict with). |
| 6 | **02 / `mlflow/_routing.py`** | Add a deterministic **agent-events GCS prefix** helper, computed from `(session, run_id)`, called by *both* the runner and the timeline (replaces the runner→timeline manifest handshake — §7 Q8). |
| 7 | **00 §11.1** | The `validate <target>` row (line ~495) lists **six** targets `{project, config, contracts, model, proposal, experiment}`; 04 Q3 froze **three** `{project, model, proposal}`. Stale 04 carry-back — fix at closeout (with the §8.8 `publish` addition). |

**Plugin-layer carry-forwards (implementation):**
- Skill `render_context.py::timeline_publish` sheds `--dry-run` (env), `--route`
  (session-derived), `--publish-mlflow` (dropped).
- Skill `render_context.py::persist_proposal` → `automl validate proposal` (was
  `automl propose validate`), and drops the `--allowed-dependencies-json` flag
  (session-resolved now — §4).
- Skill `render_context.py` `loop_context` safe-command → `automl experiment
  proposer-context` (was `automl loop-context for-proposer`).
- Proposer/coder agent wiring for `required_preprocessing` (already a 06 plugin
  carry-forward) — proposer populates from `describe_required_transformers`, coder
  reads it.
- The inner slash-command arg contract (`run` subcommand the skill parses).

**Caller + test updates (implementation — enumerated so none are lost):**
- **CLI module dissolution:** `cli/loop_context.py` is removed — its subverbs
  redistribute: `for-proposer` → `experiment proposer-context` (this spec),
  `leaderboard`/`summary` → `experiment …` (09), `show-trial` → `trial show` (10);
  `recent-failures`/`strategies` → `experiment …` (09). `cli/run_loop.py` →
  `agent/launch.py` + thin `experiment run` wrapper. `cli/propose.py` (`automl
  propose validate`) retired → `automl validate proposal` (04).
- **Contract tests:** `tests/contracts/test_phase_b_retired_scripts.py` requires
  `cli/run_loop.py` to *exist* — update for the new launcher home/verb;
  `tests/integration/test_skill_render_context.py` asserts the legacy `loop-context`
  + `timeline_publish` arg shapes — update to the renamed verbs/dropped flags;
  `tests/contracts/test_skill_plugin_contract.py` (`agent_timeline.py` + `hook-event`
  in hooks) still holds (stub stays).
- **Unit/integration tests** importing `from automl.propose import validate`,
  `gather_proposal_context`, `run_loop.build_launch`, and the `agent_timeline.py`
  CLI subcommands repoint at the new module paths / internal reconciliation fn.

---

## 10. Migration mapping (legacy → `agent/`)

| Legacy | New home |
|---|---|
| `cli/run_loop.py` (`build_launch`, `LaunchSpec`, `ClaudeRole`, `_*`) | `agent/launch.py`; CLI verb `experiment run` → thin `cli/` wrapper |
| `hooks/agent_timeline.py` (1955L) | `agent/timeline.py` (`handle_event`, `publish`, reconciliation) + thin `hooks/agent_timeline.py` stub; writes seam-routed |
| `propose/schema.py` (field lists) + `propose/__init__.py::validate()` | `agent/proposal.py` (`Proposal` dataclass + `DISALLOWED`) + `agent/checks.py::proposal_schema` (per 04); legacy `Issue`/`ValidationReport` → `validate/base.py` (04) |
| `loop_context/proposer_packet.py` (`gather_proposal_context`, `find_prior_experiment`) | `agent/proposer_context.py` (the `_primary_eval_*` enrichment dropped) |
| `loop_context/queries.py::{recent_failures, strategies_attempted, top_n_by_metric}` | `experiment/views/queries.py` + seam (09 — not agent) |
| `loop_context/queries.py::show_trial` | `trial.show_trial` (10) |
| `loop_context/queries.py::{runs_using_strategy, runs_in_metric_band}` | deferred, no placeholder file (09) |
| `loop_context/summary.py` (`build_summary`, `build_summary_from_context`) | `experiment/views/summary.py` (09 — not agent) |
| `loop_context/summary.py::load_mlflow_context` | dissolved — its job (wrap context + add `higher_is_better`/`trial_summaries`) folds into `agent/proposer_context.py` (the composer) + `experiment/views/summary.py` (09) |
| `loop_context/__init__.py::experiment_id()` | **dropped** — 09 already dropped the public `experiment_id()` helper (numeric id is seam-internal); the one test caller updates |
| `loop_context/queries.py::strategies_attempted` (no `training_origin` filter — counts all trials) | `experiment/views/queries.py` (09) — **preserve the no-origin-filter behavior** (09 carry-back note) |
| `mlflow/store.py::get_context` (~230L aggregator) | dissolved into `agent/proposer_context.py` (composer over 09 views / 10 trial reads / data seam / 02 seam); learnings reads dropped (out of scope) |

---

## 11. Open-questions updates (applied at closeout)

- 🔵 **06 carry-in — `Proposal.required_preprocessing`** — RESOLVED (§3): `list[dict]
  | None`; proposer populates, coder reads; `proposal_schema` allows-not-enforces.
- 🔵 **09 carry-in — `find_prior_experiment` + `proposal_schema` session-signature** —
  RESOLVED (§4/§5): `find_prior_experiment` is agent-owned (cold-start, creation_time
  ordering); `proposal_schema(proposal, *, session=None)` session-resolves allow-list.
- 🔵 **10 carry-in — `SLUG_RE` from `utils/`** — RESOLVED (§3): imported from `utils/`.
- 🟡 **New carry-backs** (#1–#7, §9) — to apply against 00 (§8.8 + §11.1), 09, 02
  (seam + `_routing.py`), 07/08 at closeout.
- ⚫ **Proposer-context composite** (the 500-line `get_context`) — RESOLVED (§5/§10):
  rebuilt domain-side as a composer; learning subsystem stays out of scope.
