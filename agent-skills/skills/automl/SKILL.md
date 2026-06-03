---
name: automl
description: Run the agent-backed AutoML loop with MLflow-backed state and Proposal handoff.
disable-model-invocation: true
---

# AutoML

Runs the controlled AutoML loop for an active repo-native project under `projects/<project_name>`. The loop keeps durable state in MLflow, uses a compact main-conversation ledger, and delegates fresh-context work to `automl-proposer` and `automl-coder`.

## Invocation Context

1. **Preflight: validate project setup.** Run this before rendering the AutoML
   run context:

   ```bash
   uv run automl --project <project_name> validate project
   ```

   If `passed: false`, print the JSON report verbatim and stop. Do not render
   the run context or create a trial. The user must fix the reported issues
   (`config.py` contracts, project-level validators) before re-running.

!`uv run "${CLAUDE_SKILL_DIR}/scripts/render_context.py" --project-root "${AUTOML_PROJECT_ROOT:-.}" --arguments "$ARGUMENTS"`

Treat the rendered context as the source of truth for invocation mode, execution semantics, dry-run routing, confirmation, budgets, project contract, and safe commands. It does not carry runtime state: the active dataset is whatever `safe_commands.materialize_dataset` prints when it runs, and the current MLflow summary comes from `safe_commands.loop_context`.
Claude's main-session model is selected before this skill starts. Use
`uv run automl experiment run` from inside `projects/<project_name>` (or
`uv run automl --project <project_name> experiment run` from an ambiguous repo root), or
launch Claude with the rendered project's configured
`models.manager.model` as `--model`, `models.manager.effort` as `--effort`, and
generated `--agents` JSON for proposer/coder model and effort when model
routing matters. The launcher also sets `AUTOML_SESSION_ID`; use it for shell
commands that need to match Claude hook `session_id` values. Direct skill
invocations may fall back to Claude's own `CLAUDE_SESSION_ID`.

## Modes

- `help`: render status and common next actions, then stop.
- `investigate`: perform read-only investigation and ask before side effects.
- `run`: print the pre-flight summary, confirm when required, then execute the loop.

## Protocol

1. Use the rendered context for all argument parsing and routing decisions.
   Project inference is allowed from inside `projects/<project_name>`, or when
   the repo has exactly one configured project. If multiple projects are
   visible from the repo root, use `automl --project <project_name> experiment run`.
2. If mode is `help`, render status and common next actions, then stop.
3. If mode is `investigate`, perform read-only investigation and ask before side effects.
4. If mode is `run`, print the pre-flight summary — at minimum the resolved
   `project_name`, dry-run vs full-run mode, `max_iterations`, and active
   dataset. If `needs_confirmation` is true, ask the user to confirm
   before continuing. The confirmation should let the user catch a wrong
   project (e.g. ambiguous `cwd`-based inference or a typo in `--project`)
   before any trials run. `--auto-confirm` skips this prompt.
5. Acquire the AutoML session lock before dataset materialization or trial dispatch:

   ```bash
   uv run automl --project <project_name> trial lock acquire --session-id "${AUTOML_SESSION_ID:-${CLAUDE_SESSION_ID}}"
   ```

   Use the active project from the rendered context. If acquisition exits `2`,
   stop and report the active route/session shown by the helper. Capture the
   `lock_id=...` value printed by a successful acquire. Release the lock at
   the end, including early stop and failure paths:

   ```bash
   uv run automl --project <project_name> trial lock release --session-id "${AUTOML_SESSION_ID:-${CLAUDE_SESSION_ID}}" --lock-id "<lock_id_from_acquire>"
   ```

6. Treat the rendered context as route rendering only. Project validation has
   already run through `uv run automl --project <project_name> validate project`.
   If the context reports `invocation.mode = "error"`,
   surface that error and stop.
7. Prepare or validate the active dataset with `safe_commands.materialize_dataset` from the rendered context. Full-run and dry-run routes must stay separate.
   Dry-run is not proposal-only. Dry-run means use the dry-run data volume,
   dry-run MLflow experiment route, and dry-run GCS prefix. It MUST still create trial directories, MUST still dispatch `automl-coder`, MUST still execute the copied trial `run.py`, and MUST still produce MLflow trial runs under the dry-run route. Do not skip coder dispatch because `invocation.dry_run` is true.
   Treat `execution_semantics.dry_run_is_proposal_only=false` and
   `execution_semantics.logs_mlflow_trial_runs=true` as authoritative.
8. For each iteration, use the `automl-proposer` agent to produce either a stop
   JSON or a complete `Proposal`. Include
   `environment.allowed_dependencies` from the rendered context in the proposer
   task. If `user_instructions` is non-empty, include those instructions
   verbatim in the proposer task and treat them as hard constraints for this
   AutoML invocation. For example, `--instruction "Use linear regression only"`
   means the proposer must emit a linear-regression-compatible proposal or stop
   with an explanation if that cannot satisfy the project contract.
   Ensembles are opt-in for latency-sensitive runs; tell the proposer not
   to propose `strategy: "ensemble"` unless the user explicitly asks, project
   instructions explicitly allow latency-heavy models, or the project
   constraints state latency is not a concern. Agent wall time and tool counts
   are recorded by Claude Code hooks; do not manually write timing marks.
   Tell the proposer to keep the plan model-facing: no cloudpickle
   serialization steps, no `mlflow.log_*` calls, and no MLflow artifact writes
   in `model.py`. The runner owns pyfunc model logging and every MLflow
   metric/tag/artifact. Include `project_contract` in the proposer task and
   tell the proposer that required project transformers and the normalized
   target column are mandatory contract inputs, not optional suggestions.

9. Validate required `Proposal` fields before dispatching implementation: `schema_version`, `slug`, `strategy`, `hypothesis`, `implementation_plan`, `constraints`, and `required_dependencies`. `implementation_plan`, `constraints`, and `required_dependencies` must be non-empty lists of strings; `constraints` must never be an object. Every `required_dependencies` entry must be present in `environment.allowed_dependencies`. Optional fields such as `rationale`, `evidence`, `data_checks`, and `risk_notes` may be passed through unchanged when present. `seed_hint`, when present, must be `auto`, `best`, `latest`, or `strategy:<name>`; never include a run ID. The seed model is selected by `trial.create` via metric query at creation time; the proposer does not pick a parent.
10. Persist the exact validated proposal before trial creation by using `safe_commands.persist_proposal` from the rendered context, not the Write tool. This command writes to the route-scoped `paths.proposal_handoff`; do not use a shared proposal filename under `.cache/automl/tmp/`.

   ```bash
   <safe_commands.persist_proposal> <<'JSON'
   <exact Proposal JSON>
   JSON
   ```

   Then create the trial with `safe_commands.create_trial` so `proposal/trial_proposal.json` becomes the durable rationale artifact for the trial. The rendered command includes `--project <project_name>` when the active project is known. The printed path is the trial directory to use, including in dry-run mode. Pass the exact persisted proposal path, created trial directory, and route-correct direct run command to `automl-coder`: dry-run uses `AUTOML_INHERIT_DRY_RUN=1 uv run <trial_dir>/run.py`, full run uses `uv run <trial_dir>/run.py`. Do not pass a full-run command for a dry-run route. Do not ask `automl-coder` to generate, edit, or update `run.py`; `safe_commands.create_trial` already created the deterministic runner. Do not describe the data source as SQL/Snowflake unless the active project's `projects/<project_name>/config.py` actually uses `SnowflakeSource`; the project-local data pipeline and active dataset context are the source of truth.
11. Use the `automl-coder` agent to implement and run exactly one trial from
    the validated proposal. Include `project_contract`, `data_context`, and
    `environment.allowed_dependencies` in the coder task. Agent wall time and
    tool counts are recorded by
    Claude Code hooks; do not manually write timing marks. When the
    `automl-coder` `SubagentStop` hook fires, the hook publishes that trial's
    canonical `agent/coder/report.json` and `agent/coder/tool_events.json`
    artifacts to the MLflow trial run. Do not ask the coder to call
    `mlflow.log_*` from `model.py`; the deterministic runner owns MLflow
    logging. If the trial needs model-specific diagnostics, ask for a
    JSON-serializable `Model.training_report()` payload only when the validated
    Proposal explicitly requires model-specific diagnostics, and let the
    runner log it as `model/report.json`. Do not mention `training_report()` in the coder task unless the validated Proposal explicitly requires model-specific diagnostics. Run the direct trial command exactly once.
    Do not rerun the trial for verification and do not run post-success shell
   checks. If the direct trial command exits nonzero or returns any status
   other than `"success"`, stop immediately after lock release and timeline
   publish. Do not inspect, repair, edit, run helper commands, or rerun after a
   nonzero trial exit. The main session must never edit the trial's `model.py`
   or run the trial command itself after coder failure; a failed coder run is a
   terminal iteration result for user review.
   The coder should report the direct command status and stdout/stderr
   markers, and the main session can publish/inspect MLflow artifacts after
   a successful coder run.

12. Keep only compact ledger state in the main conversation: proposal slug, strategy, seed, status, primary metric, run ID, and one-line takeaway.
13. If the coder reports `missing_dependency`, print the exact package request and stop so the user can install and resume. If the coder reports `failed`, print the reported `AUTOML_ERROR=` marker when present, publish the timeline, release the lock, and stop without repair.
14. Publish the route-scoped agent timeline with `safe_commands.timeline_publish`.
    Publishing reconciles the route-scoped session: the overview run gets
    `agent/sessions/<session_id>/report.json`, trial runs are
    backfilled with `agent/proposer/report.json`,
    `agent/proposer/tool_events.json`, `agent/coder/report.json`, and
    `agent/coder/tool_events.json` when needed, and raw hook
    events/transcripts are stored in GCS when project storage is configured.
    Treat this as end-of-session reconciliation, not the only source of trial
    observability. Do not log timing as MLflow metrics; timing belongs in
    artifacts.
15. Confirm the session lock has been released when a lock was acquired,
    including early stop and failure paths.
16. End by rendering the stop reason and a short review-oriented summary from `safe_commands.loop_context` in the rendered context. This command already includes `--dry-run` when dry-run routing is active.

## Proposal Contract

When prompting `automl-proposer`, request exactly this shape. The proposer must return JSON only.
The `slug` must be short lowercase snake_case, start with a letter, and omit
the numeric trial prefix; the orchestrator assigns trial numbers.

```json
{
  "schema_version": 2,
  "slug": "lgbm_baseline",
  "strategy": "baseline",
  "hypothesis": "Establish a reference model on the current feature pool.",
  "implementation_plan": [
    "Train a baseline model using registry-selected feature columns.",
    "Let the runner log metrics, validation fixtures, and the MLflow pyfunc model."
  ],
  "constraints": [
    "Do not read test data directly.",
    "Do not change target or primary metric.",
    "per_trial_seconds: 600"
  ],
  "required_dependencies": [
    "pandas",
    "numpy",
    "scikit-learn"
  ],
  "rationale": "No successful trials exist yet.",
  "evidence": [],
  "data_checks": [],
  "risk_notes": []
}
```

## Agent Boundaries

`automl-proposer` is read-only. It sees project direction, bounded MLflow summaries, active dataset context, and optional profile artifacts. It must return exactly one JSON object.

`automl-coder` receives one validated `Proposal`, project constraints, `allowed_dependencies`, data context, and the trial directory with any seed `model.py` already copied by `trial.create`. It writes only the current trial and runs only that trial.

Active project files are:

- `projects/<project_name>/config.py`
- `projects/<project_name>/PROJECT_INSTRUCTIONS.md`
- `projects/<project_name>/data/queries/`
- `projects/<project_name>/data/pipeline.py` (optional custom DataPipeline subclass)
- `projects/<project_name>/eval/metrics.py` (optional custom Metric subclasses)

## Resume

Re-run `/brigit-automl:automl experiment run --project <project_name>`. MLflow is the
durable resume source. Scratch lock and loop files are coordination state only;
do not treat them as durable run state.

When scanning `experiments/<trial_id>/` for interrupted work to resume, read
`metadata.json` first and skip any trial whose `training_origin` is `"human"`.
Human-authored trials are owned by the data scientist; the agent loop must not
invoke the coder on them or attempt to complete or modify them.
