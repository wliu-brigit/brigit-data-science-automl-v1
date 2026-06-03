---
name: inspect
description: Summarize AutoML progress from MLflow for an active Brigit AutoML project.
disable-model-invocation: true
---

# Inspect

Summarize an AutoML project's progress using MLflow as the source of truth.
Use `--project <project_name>` to select the active project and `--dry-run` to
inspect the dry-run route.

The inspection commands are read-only. The summary command below may ensure the
routed `project_overview` and `experiment_overview` MLflow runs exist. Do not
refresh data, materialize datasets, or launch trials from inspect.

## Steps

Render the top trials with:

```bash
uv run automl --project <project_name> --project-root <project_root> experiment leaderboard --metric <metric> --n <n>
```

If `--dry-run` is provided, append `--dry-run`.

For selected trial details, use:

```bash
uv run automl --project <project_name> trial show <run_id>
```

For side-by-side run inspection, use:

```bash
uv run automl --project <project_name> experiment compare <run_id_a> <run_id_b>
```

Render the leaderboard, selected run IDs, metrics, tags, active dataset id,
per-trial dataset hash, and artifact URIs when available. Do not compare
trials as equivalent when they used different datasets.

For selected top trials, show `pyfunc_model_uri` when present. Fetch model artifacts only when a concrete MLflow artifact fetch helper is available in the project; otherwise do not claim model code was fetched or cached.

Build the experiment summary payload with:

```bash
uv run automl --project <project_name> --project-root <project_root> experiment summary
```

If `--dry-run` is provided, append `--dry-run`.

Render the summary payload sections that matter most, especially:

- `winning_techniques`: techniques with evidence from top trials.
- `recommendations_for_next_experiment`: concrete guidance for future proposal turns.

For questions these CLI verbs don't cover, write ad-hoc **read-only** Python
against the library (`automl.experiment`, `automl.trial`, `automl.mlflow`) with
`uv run` — query, aggregate, and render freely, but never create, modify, or
delete runs, datasets, or artifacts from inspect.

Until a durable inspect writer script exists, do not claim `experiment_summary.json` has been saved to MLflow. The durable writer must write through `automl.mlflow.store` to the routed `experiment_overview`, not to local project state first.

End with a suggested next action: another
`/brigit-automl:automl experiment run --project <project_name>`, deeper manual
inspection, or stop.
