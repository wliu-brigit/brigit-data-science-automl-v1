# Final Review Open Items

Final whole-refactor audit completed 2026-05-28. A follow-up spec-coverage/readiness review on
2026-05-28 found the items below. Accepted Phase 0-7 gates were green at the final audit commit,
and the implementation work is committed, but this branch is **not merge-ready** until these items
are explicitly triaged.

## O1. Source Trace Artifact Logging Is Specified But Not Implemented — Fixed

- **References:** `spec/00-structural-design.md` Dataset row says source lineage is logged
  alongside materialized data; `spec/05-data.md` materialization guarantees require
  `DataSource.artifact_files(pipeline)` output to be logged under
  `data/datasets/<dataset_id>/source_trace/`.
- **Resolution:** follow-up fix routes `materialize()` through the source trace hook and logs
  returned files to the project overview run. Tests now cover both the project seam writer and
  materialization consuming `artifact_files(...)`.
- **Blocks cutover/merge:** no, pending the normal final verification gates.

## O2. Snowflake Source Is Still A Stub

- **References:** `spec/00-structural-design.md` notes Snowflake remains stubbed in the dev
  workspace; `migration-checklist.md` marks `SnowflakeSource` as `[x]` only as a Phase 2
  resolvable stub and drops live Snowflake helpers as deferred. Current
  `automl/data/sources/snowflake.py::SnowflakeSource.load()` raises `NotImplementedError`.
- **Why this is ambiguous:** The specs acknowledge the stub for this refactor, but a production
  merge may still require Snowflake-backed projects.
- **Options:** keep stubbed and document unsupported source type; or implement live Snowflake
  loading and source trace files before merge.
- **Recommendation:** blocker only if any first-cutover project needs Snowflake. Otherwise keep
  as explicit unsupported/deferred scope.
- **Blocks cutover/merge:** project-dependent.

## O3. Project Learning Subsystem Was Intentionally Dropped

- **References:** `spec/02-mlflow-seam.md`, `spec/05-data.md`, `spec/09-experiment.md`, and
  `spec/11-agent.md` all drop golden/weak features, learning caches, `learning_counts`, and
  project/experiment learnings. `migration-checklist.md` rows for learning artifacts are `[-]`.
- **Why this is ambiguous:** This is spec-approved, but it is still a product capability removed
  from the cutover tree.
- **Options:** accept the drop for v1; or design a new first-class learning subsystem in a later
  phase before merge.
- **Recommendation:** do not block merge unless the product requires historical learning behavior.
- **Blocks cutover/merge:** no, unless learning artifacts are required by users.

## O4. Deferred Experiment Analytics And Archive/Create Verbs

- **References:** `spec/00-structural-design.md` and `spec/09-experiment.md` defer
  `runs_using_strategy`, `runs_in_metric_band`, and standalone `experiment create/archive` demand.
  Current CLI exposes `experiment run`, `delete`, `leaderboard`, `compare`, `summary`, and
  `proposer-context`.
- **Why this is ambiguous:** The design intentionally avoids no-caller surfaces, but a merge
  readiness review may decide operators need these verbs.
- **Options:** accept the current noun-first surface; or add specific operator verbs with tests.
- **Recommendation:** do not implement unless an operator workflow needs them.
- **Blocks cutover/merge:** no under the accepted specs.

## O5. Generic Project-Scoped Loose JSON Helper Remains Deferred — Fixed

- **References:** `spec/02-mlflow-seam.md` describes loose `project.log_json(...)` for
  project-scoped artifacts.
- **Resolution:** follow-up fix implements `automl.mlflow.project.log_json(...)` on the project
  overview run and exports it from the project seam.
- **Blocks cutover/merge:** no, pending the normal final verification gates.

## O6. Multi-Process Overview Write Coordination Is Deferred

- **References:** `spec/open-questions.md` records multi-process write coordination as deferred
  with last-write-wins overview tags acceptable at current scale.
- **Why this is ambiguous:** Serial gates pass, but parallel agent/trial runs could race on shared
  overview state.
- **Options:** accept current scale assumption; or add write coordination/merge semantics before
  increasing concurrency.
- **Recommendation:** not a merge blocker for serial operation; revisit before parallel loop
  execution.
- **Blocks cutover/merge:** no for current accepted gates.

## O7. Merge Readiness Has Not Been Granted

- **References:** root README, refactor README, plan README, and implementation strategy now state
  that Phase 0-7 implementation is committed but merge readiness is intentionally open.
- **Why this is ambiguous:** The prior final audit proved accepted gates, not business readiness.
- **Options:** run a merge-readiness review that explicitly disposition O1-O6; or continue fixing
  blocker items before requesting review.
- **Recommendation:** do not merge until the follow-up fixes have fresh local/external gate
  evidence and O2 is checked against intended first-cutover projects.
- **Blocks cutover/merge:** yes; this is the current handoff state.
