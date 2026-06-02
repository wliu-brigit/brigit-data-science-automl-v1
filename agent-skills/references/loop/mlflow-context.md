# MLflow Context Schema

Trial state is stored as MLflow runs under the routed experiment:
`<project_name>/<experiment_id>` for full runs and
`dry_run/<project_name>/<experiment_id>` for dry runs.

## Trial Run Fields

Each trial run should expose these compact tags, metrics, and artifact pointers:

- `project.name` - active project under `projects/<project_name>/`.
- `experiment.id` - configured modeling cycle.
- `experiment.name` - MLflow experiment route.
- `trial.id` - `N_short_name`.
- `trial.name` - MLflow run name, normally the same as `trial.id`.
- `trial.status` - `success`, `failed`, `timeout`, `crashed`, or `missing_dependency`.
- `trial.origin` - AutoML-generated or human-authored origin.
- `dataset_hash` plus dataset artifact URIs recorded in the trial data contract.
- `eval.primary_metric` plus the primary MLflow metric value.
- `validation.status` - `success`, `failed`, `warning`, or `error`.
- `run_id` - MLflow run id.

Detailed status, error, dataset, and validation information lives in artifacts
under matching folders such as `data/`, `eval/`, `validation/`, and `logs/`.

## Context JSON Fields

`uv run automl --project <project_name> --project-root <project_root> experiment proposer-context`
returns the compact context consumed by AutoML skills and agents:

- `overview` - project and experiment overview experiment names and run ids.
- `leaderboard` / `top_trials` - successful trials sorted by primary metric.
- `recent_failures` - recent non-success trial summaries.
- `strategies_attempted` - count of trial strategy tags observed in MLflow.
- `experiment_learnings` - accepted learnings scoped to this experiment.
- `project_learnings` - accepted learnings reusable for this project.
- `data_context` - active dataset manifest/profile state, dataset usage, and
  observations from the experiment overview.
- `artifact_uris` - MLflow artifact URIs grouped by overview or trial run.
- `artifact_errors` - non-fatal artifact read failures surfaced for inspection.
- `project_instructions` - current project instructions inlined for proposer use.

## Reading conventions

- Use `uv run automl --project <project_name> --project-root <project_root> experiment proposer-context` from skills, or the equivalent command supplied by `safe_commands.loop_context`.
- Append `--dry-run` when reading the dry-run route.
- Treat local context files as MLflow-derived cache only.
- Source-trial parentage is read from the `trial.parent_id` tag. MLflow's
  parent run tag links each trial run to the experiment overview run.
- `data_context.active_dataset` contains the active dataset manifest and
  identity fields.
- `data_context.dataset_usage` maps dataset identity hashes to the number of
  trial runs that recorded them.
