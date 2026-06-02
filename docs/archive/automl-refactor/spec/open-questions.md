# Open Questions — AutoML Refactor

**Purpose:** rolling log of ambiguities, cross-cutting concerns, and items that surface during one sub-spec but ripple back to the structural spec or impact other domains. Each entry has a source (which session raised it), a brief description, and a status.

Closeout rule: before invoking `writing-plans` to produce the implementation plan, every item here is either resolved (with a pointer to where it was settled) or explicitly deferred to implementation-time decision-making with rationale.

> **All sub-specs 00–11 are approved.** ✅ **The final cross-doc pass is COMPLETE (2026-05-27).**
> All 12 open `🟡` items were applied (→ `🔵`) or consciously deferred (`⚪`); the `§B` spot-checks
> uncovered and fixed **nine latent carry-back gaps** (claimed-applied but never written — see
> "Final-pass closeout" at the bottom); `§C`/`§D` ran clean. Ready for `writing-plans`.

---

## Status legend

- 🟡 **OPEN** — needs discussion in a future sub-spec or back-and-forth
- 🔵 **RESOLVED** — decided; spec updated; resolution pointer recorded
- ⚪ **DEFERRED** — will be decided during implementation; rationale recorded
- ⚫ **OUT OF SCOPE** — intentionally not part of this refactor; either stays legacy or future sub-spec

---

## Items

### Surfaced during sub-spec 02 (MLflow seam) — POST-REVIEW UPDATE 2026-05-22

After three-agent review (independent + codebase-aware + coverage-validating), the items below have been resolved or had their scope clarified. The doc now reflects the resolutions.

- 🔵 **`list_eval(run_id)` return shape.** RESOLVED — committed to `list[tuple[str, str]]` (label, eval_dataset_id; renamed from `eval_snapshot_id` by sub-spec 07 Q3). Sourced from MLflow tags so listing requires zero artifact loads. See §6.3.4 + §10.

- ⚪ **Manifest-driven artifact listing.** DEFERRED — implementation detail of `trial/reads.py`. See sub-spec 02 §13 item 1.

- ⚪ **Tag-key namespacing convention.** DEFERRED — pick a single convention at implementation. See sub-spec 02 §13 item 2.

- 🔵 **`route_namespace` source of value — RESOLVED (final pass 2026-05-27), renamed `namespace`.** Was always `""`/unwired (effectively dead) in legacy; the user confirmed a real use (full-fidelity QA/test sandboxes, cleanly deletable). Promoted to a first-class isolation dimension: top-level **`--namespace <name>` flag** (+ env for subprocess inheritance) → `Session.namespace`, defaulting `""` = real; **full-universe** (segregates MLflow experiment names + GCS prefixes + local trial sandbox dirs), orthogonal to and composable with dry_run (`[<namespace>/][dry_run/]<project>/<id>`). Applied to 00 §11.1/§9.1, 01, 02 §3.1/§13.5, 03 routing, 07, 08. See sub-spec 02 §13 item 5.

- 🔵 **`active_session()` and `update_session()` lock-step `mlflow.bind()`.** RESOLVED + APPLIED 2026-05-22 — carry-back from sub-spec 02 §12 was applied directly to sub-spec 01. New §4.6 of sub-spec 01 defines `_bind_mlflow_for(session)` as the shared helper; `use_project`, `update_session`, and `active_session` all call it. §9.5 async-safety table updated to show session + mlflow bind move atomically together. **No longer a carry-back — fully integrated into sub-spec 01.**

- ⚪ **Multi-process write coordination.** DEFERRED — last-write-wins on overview tags is acceptable at our scale. See sub-spec 02 §13 item 8.

- ⚪ **Internal pagination for list/search.** ACKNOWLEDGED as required at implementation — not exposed in the public surface. See sub-spec 02 §13 item 3.

### Out-of-scope items surfaced during review

- ⚫ **Project-level "learning" subsystem** (golden features, weak features, learning cache JSONs). Today's `store.py` has writers for these; they remain in `automl_legacy/` and do not migrate. User direction: not well-thought-out yet; out of scope. See sub-spec 02 §13 item 12.

- ⚫ **Proposer-context composite assembly.** The 500-line `get_context` aggregator in today's `store.py` is rebuilt in `agent/proposer_context.py` (domain side) using mlflow.* building blocks. Concrete shape settled in sub-spec **11 (Agent)** — the experiment domain was split into experiment/trial/agent during sub-spec 09. Not a sub-spec 02 deliverable.

### Surfaced during sub-spec 03 (Cleanup) — 2026-05-22

- 🔵 **Trial sandbox dir segregation by run_mode — RESOLVED at sub-spec 08 (Q1).** Decided 2026-05-22: trial sandbox dirs WILL be path-segregated by mode, matching every other layer's `dry_run/` path-prefix convention. New layout: `projects/<project>/experiments/<route>/<trial_name>/` where `<route>` is `<project>/<id>` for real and `dry_run/<project>/<id>` for dry_run. Cleanup at any scope becomes a single tree-delete. Sub-spec 03 §3.4 records the requirement. **08 Q1 correction:** the original wording "the runner creates dirs at the segregated paths" is wrong — `trial.create` builds the folder; the **runner owns + enforces the path** (a path helper in `runner/`, used by creation to build + by the runner to *verify* as a universe-isolation guard). Sub-spec 03 §3.4 wording to be corrected at closeout.

- 🔵 **CLI shape realignment — CARRY-BACK to structural spec §11.1.** Applied 2026-05-22: noun-first verbs everywhere except `validate`; top-level `--dry-run` flag (session-wide); `--apply` is the sole destructive gate; `--hard-delete` replaces `--purge-mlflow`. All 21 verbs catalogued in §11.1. Cleanup's three-sibling-verb shape (`project delete` / `experiment delete` / `trial delete`) replaces the legacy single `cleanup --scope` verb.

- 🔵 **`mlflow_artifacts_destination` env value — CARRY-BACK to sub-spec 01 §3.1.** Applied 2026-05-22: added to `ProjectConfig` alongside other env-derived fields. Threaded into `mlflow gc --artifacts-destination` by `--hard-delete` per sub-spec 03 §6.4. Matches legacy cleanup.py lines 597-602.

- ⚪ **Concurrent delete operations in same process — DEFERRED.** Cleanup is designed for single-operation-at-a-time use. Concurrent async tasks calling `delete()` in one process would race over session contextvar + mlflow bind state. Documented as unsupported in sub-spec 03 §7.7. If concurrent cleanup becomes a real need, a token-preserving session bind would be required; not in v1.

- ⚫ **`--all-projects` bulk delete — DROPPED.** Legacy supported scrubbing every project at once. New design intentionally drops this; users can compose `automl project list` + a shell loop if needed. Recorded as a follow-demand item in sub-spec 03 §12.

- 🔵 **New seam method `mlflow.trial.get_parent_experiment(run_id) → ParentExperimentRef` — CARRY-BACK to sub-spec 02 §6.3.3.** Applied 2026-05-22 during fresh-eyes review. Trial-delete needs to look up its parent experiment (to compute paths + verify mode/project match the session); replacing the earlier `mlflow.client.raw()` chain with a typed seam method keeps the "domain code never imports MLflow directly" invariant intact. New typed return `ParentExperimentRef` lives in `trial/types.py` alongside `TrialDetails` (moved from `experiment/views/types.py` during the sub-spec 09 decomposition — Trial types belong to `trial/`).

### Surfaced pre-sub-spec-05 alignment (2026-05-22)

- 🔵 **Dataset / EvalDataset shared seam.** User flagged the cross-domain primitive leak before sub-spec 05 began: `eval/snapshot.py` imports `_json_hash` (underscore-prefixed private) from `data/snapshot.py`, and several eval files reach into data internals. Decided 2026-05-22: AutoML-agnostic hash primitives (`dataframe_content_hash`, `schema_hash`, `json_hash`) promoted to a new `utils/hashing.py`; the two identity classes (`Dataset` in `data/`, `EvalDataset` in `eval/`) stay in their respective domains; `HashKey` / `hash_key_columns` stay in `data/split.py` as a legitimate public cross-domain symbol; no top-level `dataset/` or `snapshot/` domain. New §13.8 in `00-structural-design.md` records the decision + forward-looking rule (future artifact families follow the same pattern). Sub-spec 05 (Data) and sub-spec 07 (Eval) must declare `utils.hashing` as their seam and resolve the parallel `_load_snapshot_by_id` underscore-prefix leak by landing a public `data/registry.py` API.

- 🔵 **Potential future unification of Dataset / EvalDataset under a shared snapshot base — companion to the resolved seam above. RESOLVED at sub-spec 07 (2026-05-24): examined; took the lighter slice, full unification deferred with a recorded north-star + tripwire.** User flagged 2026-05-22 that beyond the primitive-sharing fix, there may be downstream benefit to a *shared snapshot abstraction* (universal identity schema, common materialize→hash→register→consume flow). Today the two artifact families have distinct identity shapes AND distinct materialization flows (`data/pipeline.py` source→hash→register vs `eval/publish.py` with `split_view` / `external` / augmentation variants); `utils/hashing.py` shares primitives but not flow shape. **Not actionable now** — promoting to a shared base costs a 7th domain folder, conflicts with the Six Nouns vocabulary (§5) which already retired "Snapshot," and there's no concrete pain point yet. **What would trigger a revisit:**
  - (a) A 3rd dataset-like artifact family appears (calibration dataset, drift baseline, prediction-set identity) and the "where does its identity live?" decision becomes recurring rather than one-off.
  - (b) Both materialization flows independently grow parallel features (versioning, lineage, registration, multi-format export) that a shared base would have provided for free — i.e., divergence becomes maintenance burden.
  - (c) A cross-domain workflow (e.g., a unified profiler that treats training + eval data identically; a lineage view that walks across both) becomes desirable and the lack of a shared base makes it awkward.
  - (d) Sub-spec 07 (Eval) work surfaces concrete cases where the EvalDataset/Augmentation shape would benefit from inheriting Dataset's identity machinery.

  If any land, re-open §13.8 and consider Option C from the original design discussion (a shared snapshot domain) or a lighter alternative (e.g., a shared `Snapshot` ABC in one of the domains). **Leave this item open through sub-spec 07** — it's the natural checkpoint to re-evaluate, since by then both sides of the seam will have been designed in detail.

  **Update 2026-05-24 (sub-spec 05 done):** the data side is now fully designed — `Dataset` identity = `ComponentHashes(source_identity, feature_registry, data_content, schema)`; materialize→hash→register→consume flow locked (Q3/Q4). Hash primitives are public in `utils/hashing.py` (shared). No shared base was needed for the data side. Decision: still **OPEN through sub-spec 07** per original rationale — re-evaluate once EvalDataset's identity + materialization flow are designed in detail and we can compare the two shapes side by side. None of triggers (a)–(d) have landed yet.

  **RESOLVED 2026-05-24 (sub-spec 07 Q1).** With both sides now designed in detail, trigger (d) had partially landed: eval's `split_view` re-implemented bucket realization + realized-frame hashing that data's new slice machinery (`load_dataset_by_id(split_range=)` + content-addressed ids + L2) already provides. The decision was **NOT** to build the shared base (Option C / shared `Snapshot` ABC) — it re-introduces the retired noun and rewrites the working `evaluate()` caching model with no functional pain forcing it — but to take the **lighter slice**: `split_view` delegates realization to data's slice loader (removing the duplication) while EvalDataset stays a distinct eval-domain type. The clean **north-star** (one content-addressed-table substrate + composable `Lineage` + train/eval as a consumption role) is recorded in `00-structural-design.md` §13.8 + sub-spec 07 Q1, to be re-opened at the named **tripwire**: the `eval → data` runtime-loading seam thickening, or a third byte-owning artifact family appearing (trigger (a)). Triggers (a)/(b)/(c) have not landed.

### Surfaced during sub-spec 04 (Validate) — 2026-05-23

- 🔵 **Validate registry mechanism — RESOLVED.** Q1 settled: built-in checks become direct function calls (no `@register` decorator, no `_CHECKS` global, no `inspect.signature` kwarg filtering, no `register_all()` defensive helper, no `_RESET_FOR_TESTS`). `CheckSpec` dataclass deleted. The registry pattern survives only as the project-side extension seam (Q2), which uses a plain `PROJECT_CHECKS` dict — no decorator. Closes the §15.2 deferred "eager-vs-lazy registry import timing" question.

- 🔵 **Validate orchestrator set — CARRY-BACK to structural spec §11.1.** Q3 settled: three orchestrators only (`project`, `model`, `proposal`). `config` and `contracts` orchestrators + CLI sub-verbs are dropped; their checks still run inside `project`. No speculative `experiment` orchestrator. §11.1's `validate` row should list only the three real sub-verbs.

- 🔵 **`validate → data` layering leak — RESOLVED via runner phase reorder.** Q4 settled: `validate.model()` becomes `(cls, *, df, registry, sample_kind)` — caller builds the sample. Runner pulls data-load forward (load snapshot → pre-fit on `head(200)` → open MLflow run → reuse snapshot for fit). Validate stops importing `data/` entirely. Small perf win: snapshot now loaded once instead of twice. `ModelProbe` dataclass deleted (becomes private orchestrator helper).

- 🔵 **`propose.validate()` + duplicate Issue/ValidationReport dataclasses — RESOLVED.** Q5 settled: `propose.validate()` deleted; schema check moves to `experiment/checks.py::proposal_schema` returning canonical `validate.Issue` directly (no translation). Three duplicate `Issue` / `ValidationReport` dataclasses unify in `validate/base.py`. Three callers updated in migration: `cli/propose.py`, `cli/trial.py:76`, and `validate/builtin/proposal_checks.py` (the adapter that's deleted).

- 🔵 **`Severity = "info"` level + `ValidationError` exception + `ValidationReport.raise_if_failed()` — RESOLVED via deletion.** Q6 settled: no emitter, no caller; all deleted. `Severity` shrinks to `Literal["error", "warning"]`. `Target` shrinks from 5 literals to 3 (only orchestrator targets remain). Per `feedback_extension_points_follow_demand` — add back when the first real consumer appears.

- 🔵 **CheckSpec schema location — CARRY-BACK to structural spec §13.1.** §13.1 lists `CheckSpec` as belonging to `validate/base.py`. Per Q1 the dataclass is deleted; remove the row.

### Second-review findings applied 2026-05-23 (sub-spec 04 round 2)

After a second three-agent review run, the following clarifications were folded into sub-spec 04. None reopened a Q decision; all sharpened or corrected wording that risked misinterpretation at implementation time.

- 🔵 **Q4 phase-reorder clarification (observability preservation).** The first-cut wording implied the *full* snapshot load moved before the MLflow run — which would have lost today's MLflow-capture of full-load failures. Sub-spec 04 §Q4 now specifies: only the small 200-row pre-fit sample is pulled forward; the full snapshot load still happens after the MLflow run opens. Today's observability semantics (pre-fit failures → no MLflow record; full-load failures → MLflow run captures) are preserved.

- 🔵 **Per-check exception wrapping preserved.** Legacy `_run()` wraps each check in try/except and emits `Issue("error", check="<name>.crashed", ...)` if a check raises. Direct-call orchestrators silently drop this. Sub-spec 04 §Q1 now mandates a `_safe(name, fn, **kwargs)` helper used by every orchestrator call site. Same behavior, made explicit.

- 🔵 **Project-side check scope shrunk to `"project"` target only.** Zero `validators.py` files exist in `projects/` today; only one test exercises the model-target project-check path. Per `feedback_extension_points_follow_demand`, sub-spec 04 §Q2 drops `"model"` and `"proposal"` from the project-side signature table. The decorator-shaped test is dropped with the same test file.

- 🔵 **`_RESET_FOR_TESTS` preserved (moved + rescoped).** Earlier wording deleted it. Re-added in sub-spec 04 §Q2 — now lives in `validate/targets.py` scoped to clearing `_PROJECT_VALIDATORS_CACHE` + evicting digest-named modules from `sys.modules`.

- 🔵 **`cli propose validate --output` flag preserved.** Adds `--output <path>` to the `validate proposal` sub-verb (writes the validated JSON on pass). Required so `skills/automl/scripts/render_context.py::safe_commands.persist_proposal` continues to work without skill changes.

- 🔵 **`sample_kind` parameter dropped from `validate.model()`.** Earlier signature carried a free-form `sample_kind: str`. Removed in sub-spec 04 §Q4 — error messages already include row count + exception type + class name; the "real vs synthetic" distinction is implicit from the caller context.

- 🔵 **`ctx` → `session` renamed throughout sub-spec 04.** Aligns with sub-spec 01's `session: Session | None = None` parameter convention. Drift caught during round-2 review.

- 🔵 **`ValidationReport` gains `schema_version: int = 1` + `from_dict` per sub-spec 02 pattern.** `ValidationReport.to_json()` is persisted by the runner as a trial artifact; it counts as a typed schema and must follow the additive-only / `schema_version` / `from_dict` convention sub-spec 02 locked.

- 🔵 **Structural spec §17.8 wording — CARRY-BACK.** §17.8 says "registry behavior is unchanged" when discussing the future `<domain>/checks/` folder split. Q1 deletes the registry, so the wording is stale. Update to: "the orchestrator's imports point at the new folder; no other behavior change."

- 🔵 **`_load_project_module` helper home pinned.** Used by data + eval + project checks to import a project's `config` module. Lives at `project/_imports.py` (project domain owns module-loading helpers); signature `(session, module_name) -> ModuleType`.

- 🔵 **`make_synthetic_fixture` signature simplified** to `(*, rows: int = 50)`. Legacy `n_numeric`/`n_categorical`/`target_col`/`seed` kwargs always took defaults; dropped per "don't pre-build configurability."

- ⚪ **`validation_report.json` as a typed MLflow artifact — DEFERRED to implementation plan.** Sub-spec 02 has typed writers at `mlflow/artifacts/<thing>.py`. The pre-fit `ValidationReport` IS persisted as a trial artifact today, so the pattern could apply. Sub-spec 02 doesn't *require* it for ad-hoc trial-result fields; decision can land at implementation.

### Surfaced during sub-spec 05 (Data) — 2026-05-24

Sub-spec 05 went through one design interview (Q1–Q9) + a three-agent review + a holistic cross-doc consistency pass. The review surfaced three reversals where the structural spec was more correct than the sub-spec draft (all applied), and one pre-existing 00↔02 inconsistency (fixed in the same pass). **All carry-backs APPLIED this session — none left pending.**

- 🔵 **Q1 vocab retirement** — "snapshot" fully retired at code level in `data/`. Applied to 00 §5/§5.1/§7/appendix; 02 §4/§6.2.3/§9.
- 🔵 **Q2 two Tier 3 anchors** — `DataSource` + `DataPipeline`; composition via `DataSpec.pipeline_cls`; `data/adapters/` deleted. Applied to 00 §8.2; 01 line 586.
- 🔵 **Q3 split-at-load + four verbs** — no eager train/test split; pyarrow push-down; leakage-safe by construction.
- 🔵 **Q4 typed objects + field audit** — `Dataset`/`LoadedDataset`/`LoadedSlice`/`DatasetIndex`+`ComponentHashes`. Dropped `prepare_event_id`/`run_mode`/`experiment_id`-on-Dataset/`source_event`/`gcs_base_path`-as-field.
- 🔵 **Q5 profile single-file + MLflow-write relocation** — `data/profile.py`; writing → `mlflow/project/artifacts.py` (§9.1). Bug-fix: profiles move experiment-overview → project-overview run. Applied to 00 §5.1/§7/appendix.
- 🔵 **Q6 FeatureRegistry lift + trim + lineage** — `data/features.py`; drop golden/weak; add `derived`+`source_columns`+`add_derived()`. Description/tags/etc. rejected after feature-store research.
- 🔵 **Q7 DataSpec consolidation + bug-fix** — thresholds/dry_run_rows solely on DataSpec; ctor 14→3 args; `constant_drop_threshold` default 0.99→1.0 (was disabling the check).
- 🔵 **Q8 Splits free-form named dict — REVERSAL: lives in `project/run_config.py`, NOT `data/split.py`** (keeps `data → project` dependency direction; avoids cycle). Applied to 00 §7/§8.1; 01 import line.
- 🔵 **Q9 TrialDataContract rename + generalize** — four types; any-named-slices; four integrity validators (L1–L4). Applied to 00 §13.1-area/appendix; 02 §6.3.4/§9.
- 🔵 **Review reversal: hash primitives PUBLIC in `utils/hashing.py`** (not private in `data/dataset.py`) — aligns 05 to 00 §13.8. Fulfills the 2026-05-22 pre-sub-spec-05 seam obligation (`eval` no longer reaches into `data/` privates).
- 🔵 **Review reversal: CLI verb is `automl data profile`** (not top-level `automl profile`) per 00 §11.1.
- 🔵 **Pre-existing 00↔02 fix: mlflow seam described as per-noun folders** (`mlflow/project/` etc.) everywhere — 00 §7/§719/appendix updated to match 02 §4's authoritative layout.

**Deferred (recorded, revisit at named triggers):**
- ⚪ **Feature description/tags metadata loader** — opt-in `projects/<name>/feature_metadata.yaml` if a real project asks. Per `feedback_extension_points_follow_demand`.
- ⚪ **Per-feature validation-rules subsystem** — separate concern; defer.
- ⚪ **Parquet sort-by-SPLITID / physical partitioning** — `materialize()` permitted-not-required to sort; revisit at implementation if pushdown ineffective.
- ⚪ **Project-side custom profile checks** (`projects/<name>/profile_checks.py`) — defer-until-demand.

**Carried forward to sub-spec 07 (Eval) — both RESOLVED at sub-spec 07:**
- 🔵 **Eval column pre-flight gate. RESOLVED (07 Q4).** Two checks at two lifecycle points: (1) early — the surface-layer data-build verb calls `eval/checks.py:check_eval_columns` against the materialized schema (whether freshly built or cache-reused), catching config↔data mismatch before any model.py; (2) pre-fit — the runner calls `EvalSpec.validate_columns` against the loaded frame, catching post-materialize config drift. One pure predicate (`missing_eval_columns`) in eval; the early caller lives in the verb, not the data domain, so no `data → eval` edge is introduced (the back-edge — `pipeline.py`'s deferred import — is deleted). Covers all three legacy call sites (`pipeline.py:806/1109/1200`).
- 🔵 **Eval-domain `of_data_snapshot_id` field naming. RESOLVED (07 Q3): renamed to `of_dataset_id`** as one line item of the full "snapshot" retirement in eval/ (clean cut, no back-compat).

**Out of scope (confirmed):**
- ⚫ **Cross-trial feature importance aggregation** → `experiment/views/` (sub-spec 09).
- ⚫ **Drift detection across Datasets** → `experiment/views/diagnostics.py` (structural §15 placeholder).

### Hidden-behavior items intentionally cut (clean break, no back-compat)

- ⚫ **Status-value casing.** Old runs used `"success"`/`"failed"`; new uses `TrialStatus.FINISHED` etc. Queries against legacy runs return nothing under the new code — accepted per the no-back-compat rule. Documented in sub-spec 02 §9 trial-level table.

- ⚫ **Deleted-experiment auto-restore — CONFIRMED CUT 2026-05-26 (full scope).** Today's
  `store.py:_activate_experiment` silently restores soft-deleted MLflow experiments. The new
  design does **not** — restore is dropped entirely. A soft-deleted experiment stays archived;
  `ensure` neither restores nor silently purges it. A same-name collision surfaces as a
  `StorageError` (user hard-deletes the old one via `--hard-delete` or picks a different
  `experiment_id`). Resurrecting an archived experiment's runs/tags/lineage is the relinkage
  rabbit hole this avoids (user direction). Documented in 02 §6.2.1. **Supersedes the brief
  2026-05-26 "scoped restore" reconciliation** — there is no restore on any path.

### Surfaced during sub-spec 06 (Model) — 2026-05-24

Sub-spec 06 formalized **project-mandated preprocessing** as a first-class model
contract (the "model = preprocessing → estimator" reframe). One design interview
(Q1–Q7) + mechanical migration pass + three-agent review; all review findings
applied this session.

- 🔵 **Q1–Q7 settled.** Contract-level extension point (C); typed
  `RequiredTransformer` (name/transformer/input_cols); inspection-gate
  enforcement framework-owned (B); top-level `ColumnTransformer` mandate (A);
  declaration in `config.py` + classes in `projects/<name>/model/preprocessing.py`;
  type+columns integrity (no hyperparam pin); prompt surfacing via
  `describe_required_transformers` → `TrialProposal.required_preprocessing`.
- 🔵 **Review fix S1 (session resolution) — RESOLVED.** Both the hook and the
  gate resolve the ambient session via `automl.session()` (sub-spec 01
  convention). Sub-spec 04's `validate.model(cls, *, df, registry)` signature is
  **unchanged** — the gate reads ambient, not a passed param.
- 🔵 **Review fix S2 (Pipeline-wrap bypass) — RESOLVED.** `self.preprocessor`
  must *be* a `ColumnTransformer`, not a `Pipeline` wrapping one; downstream
  steps live in `self.model`. Keeps the gate a clean one-level check.
- 🔵 **Review fix C1 — RESOLVED.** Gate inspects `ColumnTransformer.transformers_`
  (fitted triples with columns), not `named_transformers_` (no column info).
- 🔵 **Migration-checklist corrections — RESOLVED.** `package_model`
  (notebook→source authoring) → `trial/packaging.py` and `load_model(run_id)` (mlflow
  pyfunc load) → `trial/show.py` move OUT of `model/` — the model domain's outbound deps
  are `errors` only (§8.3). Land in **sub-spec 10 (`trial/`)** post-decomposition. Checklist rows fixed.

**Carry-backs APPLIED this session:** sub-spec 01 (`ProjectConfig.required_transformers`
field), structural §8.3+§7 (`model/preprocessing.py`), sub-spec 04 (gate +
ambient-session note). Migration-checklist updated.

**Carried forward to sub-spec 11 (Agent)** *(was "sub-spec 09" pre-decomposition — the Proposal contract lives in `agent/`)*:
- 🔵 **`Proposal.required_preprocessing` field — RESOLVED (sub-spec 11 §3).**
  `list[dict] | None` on the typed `Proposal` dataclass; proposer populates from
  `model.describe_required_transformers(session)`; coder reads it; `proposal_schema`
  *allows* but does NOT re-enforce (single gate = model gate).

**Carried forward to plugin layer (implementation):**
- ⚪ **Coder/proposer agent wiring + `references/setup/model-contract.md` update**
  to document the required-preprocessing contract and the
  `projects/<name>/model/preprocessing.py` stub. **CONFIRMED as an implementation
  carry-forward (final pass 2026-05-27)** — a plugin/skill change, not a library-spec
  edit; deferred to writing-plans (editing the live skill now is premature, the new
  verbs don't exist yet). The library half landed: `00 §8.3`/§7 + 01 §3.1
  (`required_transformers`) + `agent/Proposal.required_preprocessing` (11 §3).

**Deferred (revisit at named triggers):**
- ⚪ **Hyperparameter pinning in the gate** — add only if a real project needs the
  mandated step immutable (`feedback_extension_points_follow_demand`).
- ⚪ **Path-based `load_model(path)` in `model/packaging.py`** — nothing loads
  from a raw path today; add on demand.
- ⚪ **Sequential/ordered required transformers** — `ColumnTransformer` entries
  are parallel + column-scoped; add ordering to `RequiredTransformer` only if a
  real case (transform A feeds transform B) appears.

### Surfaced during sub-spec 07 (Eval) — 2026-05-24

Sub-spec 07 ran one design interview (Q1–Q6) + a three-agent review; all review
findings were triaged with the user and applied. All three carry-backs into 07 are
resolved (see above): unification checkpoint (Q1), pre-flight gate (Q4),
`of_data_snapshot_id → of_dataset_id` (Q3).

- 🔵 **Q1 split_view delegation + north-star.** Both `EvalDataset` kinds kept;
  `split_view` delegates realization to `data.load_dataset_by_id`; full
  substrate+lineage+role unification recorded as north-star in §13.8, re-opened at
  the `eval → data` seam-thickness tripwire / a third byte-owning family.
- 🔵 **Q2 recipe-only split_view identity.** Drop realized schema/content hashes from
  the split_view manifest (load *and* publish paths); integrity from the
  content-addressed `of_dataset_id` + data's **L2** load-time validation (NOT L3 /
  `verify_loaded_slice`, which is trial-contract-scoped). Empty-bucket detection
  intentionally moves publish-time → first-load.
- 🔵 **Q5/Q6 verb + type consolidation.** `eval/runner.py` deleted (`run()` →
  `evaluate_frame()`); `session` convention; four types removed
  (`EvalSnapshotPointer`, `AugmentationPointer`, the separate manifest schema, and
  `EvaluateResult`+`EvalResults` → singular `EvalResult`); `EvalIndex`/`Predictions`
  typed; `mlflow_url` dropped (CLI derives via mlflow-seam helper); `cached` the lone
  runtime-only field.

**Carry-backs APPLIED this session:** §00 §8.4 (`load_dataset_by_id` added to eval's
allowed-imports list + tripwire note), §00 §13.8 (EvalDataset identity reshaped to
recipe-only split_view + unification-checkpoint resolution), §00 Tier-2/schema tables
(`EvalResults` → `EvalResult`), sub-spec 02 (`EvalResults`/`EvalResultsRef` →
`EvalResult`/`EvalResultRef`; `eval_snapshot_id` → `eval_dataset_id` throughout the
mlflow-seam eval writer/payload/listing).

**Carried forward to sub-spec 05 (Data) — RESOLVED + APPLIED (final pass 2026-05-27):**
- 🔵 **`load_dataset_by_id` contract requirements.** 05 diverged *by omission* (defined
  the L2 validator + claimed "cross-checked at load time" but never said which load runs
  it; `split_range` untyped). Both now written into 05: (a) a "**L2 runs by default at
  load time**" paragraph after the integrity-layers table states `load_dataset` /
  `load_dataset_by_id` / `load_dataset_by_trial` invoke `validate_loaded_dataset` before
  returning (07 Q2's recipe-only integrity leans on it); (b) `split_range` is typed
  `tuple[tuple[int, int], ...]` and explicitly accepts **multiple disjoint pairs**
  (`((80,90),(95,100))`), bare single pair normalized. Matches 07 Q1's `split_view`
  delegation.

**Carried forward to sub-spec 01 (Project) — RESOLVED (final pass 2026-05-27):**
- 🔵 **`project → eval` edge for `evaluation_spec`.** Already resolved in 01: the legacy
  lazy `evaluation_spec` property is replaced by the **eager `ProjectConfig.eval_spec`
  field** (01 §3.1/§12), and `primary_metric` is a derived property over it. No separate
  re-export. `load_evaluation_spec` keeps living in `eval/_load.py` (eval owns it);
  `ProjectConfig.load()` invokes it via a **late import** (same acyclic pattern as
  `_bind_mlflow_for`), so `project → eval` is load-time only, no cycle. Confirming note
  added to 01 §13. (Cycle-avoidance — eval→project already exists — makes this the only
  sound shape; user-confirmed.)

**Out of scope (confirmed):**
- ⚫ **Substrate+lineage+role unification (Option C / shared `Snapshot` base)** — the
  north-star, deferred per the tripwire above; not this refactor.
- ⚫ **`automl validate` data↔eval consistency target** against a materialized dataset
  — defer-until-demand (`feedback_extension_points_follow_demand`).

### Surfaced during sub-spec 08 (Runner) — 2026-05-25

Sub-spec 08 ran one design interview (Q1–Q6) + a three-agent review; all findings
triaged with the user and applied. **Scope = Tier 1** (carry-backs in place +
mechanical hygiene; the stage-pipeline/pluggable runner is the deferred Tier-2
north-star, re-opened only when a real 2nd runner shape appears).

- 🔵 **Q1 trial folder path/creation.** Mode-segregated path adopted; `trial.create` builds, runner verifies (universe-isolation guard); path helper in `runner/`. Corrects 03 §3.4 wording (see above).
- 🔵 **Q2 numbering/identity.** Exec-time assignment kept; `_next_trial_number_from_mlflow` → `mlflow.experiment.next_trial_number` (carry-back to 02).
- 🔵 **Q3 phase order (04).** 04's ordering re-expressed in 05's `load_dataset` vocab; pre-fit sample = `load_dataset(split_name=train_split).df.head(200)` (no loader `limit=`, deferred); `session.dry_run`; SIGALRM armed before phase 1.
- 🔵 **Q4 data_load + TrialDataContract (05).** Fit-slice-only contract; built from `LoadedSlice` + written via `mlflow.trial.artifacts.write_trial_data_contract`. **`TrialRef` keeps both `trial_id` + `run_id`** — carry-back to 05 Q9 (the "drop as duplicate" was an error; 05 misread 00 §5's vocabulary as field-dedup).
- 🔵 **Q5 eval + pre-fit gate (07).** Runner=fit-only / `evaluate()` owns eval-data; pre-fit `validate_columns` on the fit frame (split_view; external eval checked inside `evaluate()`); train-eval diagnostic kept; validation fixture rebuilt from a fit-frame sample.
- 🔵 **Q6 decomposition.** `runner/trial.py` + cohesive modules (`paths`/`contract`/`validation`/`_pyfunc_check`/`manifest`/`_modules`/`session_lock`); **no `runner/stages/`** (carry-back to 00 — appendix migration line ~840 only); dead code + star-import removed; session lock moved.
- 🔵 **Eval-slice lineage decision.** Fit-only trial contract accepted as intentional (user) — `splits` ranges + eval-domain integrity (`eval_dataset_id` + L2) cover it; no guarantee lost.

**Carry-backs APPLIED at closeout:** 03 §3.4 (path wording), 05 Q9 (`TrialRef.run_id`), 02 (`next_trial_number` seam read), 00 (drop `runner/stages/` from Tier-1 migration target).

**Open / carried forward:**
- 🔵 **`RunDataContract.to_split_view()` consumers — RESOLVED + APPLIED (final pass 2026-05-27).** 05's "consumers port to per-slice access" was imprecise. Enumerated the real callers (none are the runner): `validate_run_data_contract` (view-hash reconstruction) → subsumed by L1 `validate_trial_data_contract`; `validate_split_view` (L3) → `verify_loaded_slice`; `inspect/views.py::load_data_snapshot` replay → `load_dataset_by_trial` + per-slice field access. The method + its `split_view`-dict/`view_hash` reconstruction are **dropped wholesale, not "ported"** — content-addressed `SliceContract.content_hash` + L1–L4 replace it, so `TrialDataContract` needs no `to_split_view()` equivalent. The runner builds the contract from `LoadedSlice` directly (08 Q4). Written into 05's "Dropped fields" list + flipped the 08 open item to 🔵. (Not a 09 concern — 09 never consumed it.)
- ⚪ **Loader `limit=`/`nrows` for the pre-fit sample** — deferred until the fit-slice double-read is a measured cost (follow-demand).
- ⚪ **Test-suite sweep for the dead-code deletions** (`_write_error_log` shim / `_load_trial_model_module` / `_frame_shape`) — impl-time.

### Surfaced during sub-spec 09 (Experiment) — 2026-05-25

**STRUCTURAL: the `experiment/` mega-domain was split into three peers** —
`experiment/` (Experiment noun + cross-trial views), `trial/` (Trial noun, promoted to a
top-level peer), `agent/` (the agentic loop). Sub-spec 09 became 09/10/11. Carried back to
`00` (§5/§6/§7/§8.6–§8.8/§11.1/§12/§13.1/§16/§17, Appendix A) and `02` (Trial type homes,
`run_url`/`artifact_url`, diagnostics zero-file). The `experiment/trial/`→`trial/`,
`experiment/proposal.py`→`agent/proposal.py`, `experiment.agent_*`→`agent.*` sweep was
applied across 03/04/05/06/07 + this doc + README + migration-checklist.

**09 (Experiment) decisions — RESOLVED (interview 2026-05-25):**
- 🔵 `Experiment = ExperimentOverview` (one type, facade-aliased).
- 🔵 `lifecycle.create` + the lazy ensure both kept; **no predecessor param** (the
  `predecessor_experiment_overview_run_id` tag is write-only / already retired — dead code).
- 🔵 Query homes: raw searches at the seam; `recent_failures`/`strategies_attempted` are
  view helpers; `recent_failures` + `compare` are **in scope**; `runs_using_strategy` /
  `runs_in_metric_band` deferred.
- 🔵 Drop public `experiment_id()` helper (numeric id is seam-internal).
- 🔵 Typed `LeaderboardData` / `ComparisonResult`; `summary` stays a dict; `learning_counts` dropped.

**09 deferrals + carry-outs:**
- ⚪ **No-caller analytics (`runs_using_strategy`, `runs_in_metric_band`)** — deferred,
  **no placeholder file** (zero-file, §Q4). Add seam search + view helper on real demand.
- 🔵 **`run_url`/`artifact_url` → 02 public seam surface** (carry-back applied; the helpers
  already exist as `store.py::run_url`/`artifact_url`).
- 🔵 **(verify-in-02) RESOLVED 2026-05-26 — no restore.** The seam's `ensure`/`ensure_overview`
  **drops** the legacy soft-delete restore (`_activate_experiment`); it sets `created_by` on
  first creation but never resurrects an archived experiment. A same-name collision surfaces as
  `StorageError` (hard-delete or rename). Documented in 02 §6.2.1. (This reverses 09 §Q2's
  "create relies on the seam restoring a soft-deleted experiment" assumption — see the 09 fix
  below.)
- 🔵 **→ sub-spec 11: RESOLVED (11 §4/§5).** `find_prior_experiment` is an
  `agent/proposer_context` concern (cold-start; sorts by `creation_time` — cheap win);
  `proposal_schema(proposal, *, session=None)` session-resolves the allow-list via
  `project.dependencies` (drops the `allowed_dependencies` param + CLI flags; fixes the
  `cli/trial.py` tautology).
- 🔵 **→ sub-spec 10: RESOLVED 2026-05-26.** `trial.show_trial -> TrialDetails`, and
  `ComparisonResult.runs: list[TrialDetails]` (10 Q1); Trial type field lists settled —
  `TrialSummary` +5 fields (10 Q3), `TrialDetails` fields defined (10 Q1/Q4), `TimingReport`
  + slim `TrialManifest` (10 Q7). All carried back to 02/09.

### Surfaced during sub-spec 10 (Trial) — 2026-05-26

Sub-spec 10 ran one design interview (Q1–Q10) + a three-agent review; all findings triaged
with the user and applied. The `trial/` domain (Trial noun, top-level peer).

- 🔵 **Q1–Q10 settled.** `show_trial -> TrialDetails`; independent `TrialSummary`/`TrialDetails`
  (shared seam builder helpers, no type composition); `evaluations: list[EvalResult] | None`
  (None=not-loaded); +5 additive `TrialSummary` fields; `TrialMetadata` drops `run_mode`/`dry_run`;
  typed `SeedSelection`+`ModelSource`; **slim `TrialManifest`** TOC + `run_id`; session-convention
  sweep; `run.py` dropped; **zero-file `checks.py`**; per-operation files + `types`/`metadata` split.
- 🔵 **Carry-backs APPLIED 2026-05-26:** 02 (`TrialSummary` +5 fields; `TrialDetails` fields
  defined §6.3.3), 09 §12 (`ComparisonResult.runs: list[TrialDetails]`), 00 §7/§8.7 (`eval`/`utils`
  deps; `checks.py` dropped; exports add `delete`), 08 (manifest writer slims to the TOC + status),
  checklist (`SLUG_RE`→`utils/`).

- 🔵 **CROSS-CUTTING — `run_mode` routing-string collapse — RECONCILED + APPLIED (final pass 2026-05-27).**
  Surfaced in 10 §7.2. The legacy two-valued routing string `"dry_run"`/`"full_run"` collapses to
  `session.dry_run` (bool) + a conditional `dry_run/` prefix in `mlflow/_routing.py`. **No domain
  function threads a `run_mode`/`dry_run` parameter**; no `"full_run"` literal is stored. **Swept
  the whole set together:** 03/05/08/10/11 were already aligned (05 drops `run_mode` from
  `Dataset`; 08 reads `session.dry_run`; 10 drops both from `TrialMetadata`). Applied the missing
  pieces — 02 §3.1 gained an explicit collapse note + the `_routing.py` description (§4/§9.1-in-00)
  now states the conditional `dry_run/` prefix; 07's `EvalDataset` route-context fields confirmed
  as derivation context (not threaded params). **Key disambiguation:** `route_namespace` (renamed
  **`namespace`**) is a **distinct, surviving** concern from the mode collapse — 11's "kill
  route_namespace" is scoped to the agent timeline's dead `""` usage + route-string parsing, NOT the
  seam's bound field. The user resolved its source-of-value (see the `route_namespace` item above):
  it is now a wired full-universe isolation dimension (`--namespace` flag → `Session.namespace`),
  **not** deferred. `02._bind_mlflow_for` passes `namespace=s.namespace` directly (Session now has
  the field).

**Carried forward to sub-spec 11 (Agent):**
- 🔵 **`SLUG_RE` shared home — RESOLVED (11 §3, confirmed at review).** Stays in
  `utils/` (10's call held). It's duplicated today in `propose/__init__.py:15` *and*
  `trial/creation.py:14` — both `agent/` (proposal-slug validation) and `trial/`
  (trial-folder naming) need it; the cycle only bites if it lives in `agent/`. The
  regex is a generic snake_case primitive (no AutoML semantics in the pattern), so
  `utils/` is the right shared-leaf home; both domains import it from there.

### Surfaced during sub-spec 11 (Agent) — 2026-05-27

Sub-spec 11 ran one design interview + a three-agent review (fresh-eyes +
codebase-gap + coverage/cross-spec); all findings triaged with the user and applied.
The `agent/` domain (the agentic loop, **relocate-only**). Closes the three carry-ins
above (06 `required_preprocessing`, 09 `find_prior_experiment` + `proposal_schema`,
10 `SLUG_RE`).

- 🔵 **Q1–Q9 settled.** Typed `Proposal` dataclass (was a raw dict; `schema_version` 2;
  dataclass-as-roster; `DISALLOWED=("parent_id",)`); `proposal_schema` session-resolves
  the allow-list (fixes the `cli/trial.py` tautology); `gather_proposer_context` rebuilt
  as a dict **composer** over 09 views / 10 trial reads / data seam (learnings +
  `primary_eval` + `artifact_uris` + `top_trials` dropped); metric ranking = parameter
  defaulting to `config.primary_metric` with a **missing-metric callout, no re-eval hook**;
  `build_launch` session-convention relocate; timeline `handle_event`+`publish`
  (reconciliation ported verbatim, one file), seam-routed writes, `AUTOML_DRY_RUN` →
  `AUTOML_INHERIT_DRY_RUN` (transport-only).

- 🔵 **Review fixes applied:** `publish_mlflow` param dropped (dead config); `project`
  metadata + `dataset_usage` packet keys restored to the §5 roster; agent-events GCS
  prefix resolved via a deterministic `_routing.py` helper (drops the runner→timeline
  manifest handshake); `SLUG_RE` confirmed in `utils/`; `agent/` has **no direct
  `runner` import** (transitive via `trial.promote`); caller/test-update list enumerated.

**Carry-backs RECORDED — application BATCHED into the final cross-doc consistency pass**
(matching how 09 batched its §18 carry-backs into 10's closeout; 11 is the last sub-spec,
so the batch pass is imminent):
All seven APPLIED in the final cross-doc pass (2026-05-27):
- 🔵 **#1 → 00 §8.8** — `publish` added to the agent Tier-2 exports.
- 🔵 **#2 → 09** — `leaderboard(metric=None)` now resolves the default from
  `config.primary_metric` (was hardcoded `"auc"`); `LeaderboardData` gained `n_unscored`
  (renders *"x/n not scored on `<metric>`"*). `strategies_attempted`'s no-`training_origin`
  filter was already correct (09 §8.1 — takes no params, counts all trials); confirmed, no
  change.
- 🔵 **#3 → 02** — `top_n_by_metric` note added: the `metric` arg is the cross-trial-stable
  `<label>.<metric>` key (`eval/evaluate.py:588`), so "missing" = genuinely uncomputed.
- 🔵 **#4 → 07/08** — confirm note added to 07 (`EvalResult.metrics` logged under namespaced
  `<label>.<metric>`; `primary` + bare-`<primary>` log at `evaluate.py:596` are per-trial
  provenance/display, not the sort key). Already true; recorded so it isn't dropped.
- 🔵 **#5 → 08** — `AUTOML_DRY_RUN` → `AUTOML_INHERIT_DRY_RUN` (transport-only) + delete the
  `runner/_execute.py:289` metadata-conflict check (nothing to conflict with after 10 §7.1
  dropped `run_mode`/`dry_run` from `TrialMetadata`). Recorded in 08's carry-backs section.
- 🔵 **#6 → 02 / `mlflow/_routing.py`** — the deterministic agent-events GCS prefix helper
  (from `(session, run_id)`, called by both runner + timeline) is now in 02 §4's `_routing.py`
  description + 00 §9.1. Replaces the runner→timeline manifest handshake.
- 🔵 **#7 → 00 §11.1** — the `validate <target>` row fixed from **six** → **three**
  `{project, model, proposal}` (+ `validate proposal --output`). **This pass also caught the
  same stale-04 leak in 00 §7 + §9.2 + §13.1 + §15.2 + §17.8 and fixed all of them** (see
  "Latent carry-back gaps" below).

**Plugin-layer carry-forwards (implementation):** skill `render_context.py` verb renames
(`loop-context for-proposer` → `experiment proposer-context`; `propose validate` →
`validate proposal`) + dropped flags (`timeline_publish` sheds `--dry-run`/`--route`/
`--publish-mlflow`; `persist_proposal` drops `--allowed-dependencies-json`);
proposer/coder agent wiring for `required_preprocessing` (06); the inner slash-command
arg contract.

---

## Cross-cutting concerns to revisit at close-out

These are themes that may need a final synthesis pass once all sub-specs are done:

- **Naming consistency across domains** — once we have N sub-specs done, do related concepts share consistent names? (e.g., does `experiment.lifecycle.create` parallel `project.scaffold.init` parallel `trial.create`?)
- **Tier 3 ABC shape consistency** — once all ABCs are defined, do their method signatures use compatible vocabulary? (e.g., does `Metric.compute` use the same arg shape pattern as `DataSource.load`?)
- **Error-type usage** — are `ProjectError`, `DataError`, etc. consistently raised at the right boundaries?
- **`session` parameter placement** — every Tier 2 function should follow the convention from the project-context sub-spec; spot-check the final API.
- **Tier 1 facade exports** — the final list at `automl/__init__.py` after all domains are designed.

---

## FINAL-PASS AGENDA (for the next session — apply + spot-check across 00–11)

All sub-specs 00–11 are approved. This is the consolidated checklist for the final
cross-doc carry-back + open-questions + consistency pass that runs **before**
`writing-plans`. It covers **all** sub-specs, not just 11. Work it **by target spec**
(edit each spec once), then run the consistency spot-checks, then confirm deferrals.

### A. OPEN items to APPLY (the 12 🟡 items above — 5 pre-11 + 7 from sub-spec 11), grouped by target spec

- **00 (structural):** add `publish` to agent Tier-2 exports §8.8 (11 #1); fix `validate
  <target>` §11.1 from six → three `{project, model, proposal}` (11 #7 — stale 04).
- **01 (project):** decide the `project → eval` re-export of `evaluation_spec` (07 carry-over,
  line 249).
- **02 (mlflow seam) + `mlflow/_routing.py`:** `top_n_by_metric` sort by cross-trial-stable
  `<label>.<metric>` (11 #3); add deterministic agent-events GCS-prefix helper (11 #6);
  absorb the `run_mode` routing-string collapse → `dry_run/` prefix conditional (10
  cross-cutting, line 338).
- **05 (data):** `load_dataset_by_id` L2-default + multi-range bucket pairs (07 carry-over,
  line 242); enumerate/port `RunDataContract.to_split_view()` consumers (08 carry-over, line
  278); `run_mode` collapse cleanup (line 338).
- **07 (eval):** confirm locked-set namespaced metric logging; bare-`<primary>` is per-trial
  only (11 #4); `run_mode` collapse cleanup.
- **08 (runner):** rename `AUTOML_DRY_RUN` → `AUTOML_INHERIT_DRY_RUN` + delete the obsolete
  metadata-conflict check `_execute.py:289` (11 #5); confirm #4.
- **09 (experiment):** leaderboard default `metric` = `config.primary_metric`;
  `LeaderboardData` unscored-count; preserve `strategies_attempted` no-origin-filter (11 #2);
  `to_split_view` consumers (line 278).
- **Plugin layer (implementation, not library specs):** coder/proposer agent wiring +
  `model-contract.md` (06, line 198); 11's `render_context.py` verb renames + dropped flags.

### B. APPLIED carry-backs to SPOT-CHECK (verify the 🔵 "APPLIED" claims actually landed)

For each sub-spec, re-open the parent spec(s) it claims to have amended and confirm the edit
is present + consistent: **02** (foundation surface), **03** (§11.1 CLI shape, 01 §3.1),
**04** (00 §11.1/§13.1/§15.2/§17.8), **05** (00 §5/§7/§8/§13.8, 01, 02), **06** (01, 00
§8.3/§7, 04), **07** (00 §8.4/§13.8, 02), **08** (03 §3.4, 05 Q9, 02, 00), **09/10** (02
`TrialSummary`/`TrialDetails`, 09 `ComparisonResult.runs`, 00 §7/§8.7, 08 manifest, checklist
`SLUG_RE`). Watch for the known hazard: a carry-back claimed-applied in a sub-spec but never
written into the target (the §11.1 six-vs-three is exactly this class).

### C. CONSISTENCY spot-checks (the "Cross-cutting concerns" section above)

Naming parallelism across domains; Tier-3 ABC method-shape consistency; error-type usage at
boundaries; `session: Session | None = None` placement on every Tier-2 fn; the final Tier-1
`automl/__init__.py` facade export list (now that all 8 domains are designed).

### D. DEFERRED (⚪) items — confirm, don't apply

The 18 ⚪ items are intentional implementation-time deferrals. The closeout rule requires each
be **explicitly confirmed as deferred with rationale** (not silently forgotten) before
`writing-plans` — a quick pass to confirm none should be pulled forward.

### Exit criterion

Every 🟡 → 🔵 (applied) or ⚪ (consciously deferred); spot-checks B/C clean; then invoke
`writing-plans`.

---

## FINAL-PASS CLOSEOUT (2026-05-27) — ✅ exit criterion met

### A. 🟡 → resolved
All 12 open items applied or consciously deferred (status flipped inline above):
- **#1 (00 §8.8 `publish`), #3 (02 `top_n_by_metric` `<label>.<metric>`), #6 (02 `_routing.py`
  agent-events prefix), #7 (00 §11.1 six→three)** → 🔵 applied to the named targets.
- **#2 (09 leaderboard default + `n_unscored`), #4 (07 namespaced-metric confirm)** → 🔵; the
  metric-ranking cluster was verified to have *no model conflict* — the sites just hadn't been
  updated. `strategies_attempted` no-origin-filter was already correct.
- **#5 (08 `AUTOML_INHERIT_DRY_RUN` + delete `_execute.py:289`)** → 🔵.
- **07→05 loader (L2-default + multi-range)** → 🔵 (05 diverged by omission; both now written).
- **07→01 `project→eval`** → 🔵 (already resolved by `ProjectConfig.eval_spec` field; recorded).
- **08→05/09 `to_split_view`** → 🔵 (dropped-not-ported; consumers enumerated; not a 09 concern).
- **10 cross-cutting `run_mode` collapse** → 🔵 (reconciled across 02/05/07/08/10/11;
  `route_namespace` separated out as a distinct concern — then resolved: see the `namespace` note below).
- **06 plugin wiring + `model-contract.md`** → ⚪ (confirmed implementation carry-forward).

### B. Latent carry-back gaps — claimed "APPLIED" but never written; **now fixed**
The §B spot-check caught nine cases of the exact hazard the agenda flagged (carry-back marked
applied in a sub-spec/README but missing from the target). All fixed this pass:
1. **00 §11.1** `validate` row six→three (#7) — *was* the named hazard.
2. **00 §13.1** `CheckSpec` row still present (04) — removed.
3. **00 §15.2** "Validate registry import timing" deferral still present (04 closed it) — removed
   (also the duplicate copy in **01 §13**).
4. **00 §7** `model/` tree missing `preprocessing.py` (06) — added.
5. **00 §8.3** model ownership/exports missing preprocessing + `RequiredTransformer` /
   `describe_required_transformers` (06) — added.
6. **00 §7 + §9.2** `validate/` blocks still showed `registry.py` / `CheckSpec` / six
   orchestrators / `@register` (04) — rewritten to the direct-call, three-orchestrator shape.
7. **00 Appendix** still listed `runner/stages/` as a migration home (08 dropped it) — corrected.
8. **01 §3.1** `ProjectConfig.required_transformers` field absent (06) — added.
9. **02** `next_trial_number(...)` seam read absent (08) — added to `experiment/queries`.
Plus two **consistency** fixes: `routing.py` → `_routing.py` in 00 §7/§9.1 (matching 02 §4); and
02's `_bind_mlflow_for` aligned to 01's (now `namespace=s.namespace`, a real Session field — see §D).

### C. Consistency spot-checks — clean
- **`session: Session | None = None`** convention present on Tier-2 fns across all sub-specs
  (33 explicit occurrences; runner/CLI entry shapes consistent).
- **Naming parallelism:** noun-first CLI verbs (`project init` / `trial create` / `experiment
  run`); domain lifecycle verbs read consistently (`create`/`fork`/`promote`). Internal module
  org differs (experiment `lifecycle.py` vs trial top-level `create.py`) — acceptable, not a leak.
- **Tier-3 ABCs** (`Metric`, `DataSource`, `DataPipeline`, `BaseModel`) — each in its domain's
  `base.py`; method shapes use the shared `session`/typed-object vocabulary. No cross-ABC drift.
- **Errors:** single hierarchy in top-level `errors.py`; boundary raising consistent
  (`StorageError` wraps backend errors at the seam; `ProjectError` at project boundary).
- **Tier-1 facade** (`00 §12`): `ProjectConfig`/`Session` + session machinery + `Dataset` /
  `Experiment` / `Proposal` / `Model`. **Deliberate:** no `Trial` noun *class* (trial exposes
  verbs + read types `TrialSummary`/`TrialDetails`, reached via `experiment.views` / `trial.show`).
  Exact export list still deferred to implementation (§15.2) — sketch confirmed coherent.

### D. ⚪ deferrals — reviewed, all remain intentional (one PULLED FORWARD)
Re-read every ⚪ item; one was **pulled forward** at the user's direction: **`route_namespace`
source-of-value** (§13 item 5) is no longer deferred — it was resolved into a wired, first-class
`namespace` isolation dimension (`--namespace` flag → `Session.namespace`, full-universe; see §A and
the `route_namespace` item above). This both fixes the dead-`""` legacy and cleanly distinguishes it
from the (settled) `run_mode` mode-collapse so a future session can't re-conflate them. The 05
follow-demand items, 06 (hyperparam pinning / path-`load_model` / sequential transformers), 02
(manifest listing / tag namespacing / pagination / multi-process / predictions repair / hook bind
bootstrap), 03 (concurrent delete), 08 (loader `limit=` / dead-code test sweep), and the
`validation_report.json` typed-artifact question all remain valid implementation-time deferrals.

**→ Proceeding to `writing-plans`.**
