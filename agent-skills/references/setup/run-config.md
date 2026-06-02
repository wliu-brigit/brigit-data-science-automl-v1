# projects/<project_name>/config.py — RUN_CONFIG

`RUN_CONFIG` is one of four module-level constants in `config.py`. It contains
only declarative run metadata. Values coupled to data logic live in `DATA`,
values coupled to metrics live in `EVAL`, and service locations live in `.env`.

## Required shape

```python
from automl.project import RunConfig, Splits, ModelsConfig, ModelRoute

RUN_CONFIG = RunConfig(
    experiment_id="2026-Q2",
    splits=Splits(train=[(0, 80)], test=[(80, 100)]),
    models=ModelsConfig(
        manager =ModelRoute("sonnet", "medium"),
        proposer=ModelRoute("sonnet", "medium"),
        coder   =ModelRoute("sonnet", "medium"),
    ),
    per_trial_seconds=600,
)
```

## Fields

| Field | What goes here |
|---|---|
| `experiment_id` | One modeling cycle within the project, such as `2026-Q2` or `2026-05-07`. Use only letters, numbers, `_`, `-`, and `.`; slashes, spaces, and URI punctuation are rejected. Lex-sortable strings are recommended. |
| `splits` | Train/test split ranges as `Splits(train=[(start, end)], test=[(start, end)])` where ranges are over the SPLITID column (0-99). |
| `models` | Per-role model routing for the manager, proposer, and coder agents via `ModelsConfig` with `ModelRoute(model, effort)` for each. |
| `per_trial_seconds` | Timeout budget for a single trial execution. |

## What is not in RUN_CONFIG

- Target column and task type: `TASK` in `projects/<project_name>/config.py`.
- Data source and column roles: `DATA` in `projects/<project_name>/config.py`.
- Primary metric and task: `EVAL` in `projects/<project_name>/config.py`.
- GCS bucket/prefix: `.env` as `GCS_BUCKET` and `GCS_PREFIX`.
- MLflow URI and credentials: `.env` as `MLFLOW_TRACKING_URI`,
  `MLFLOW_TRACKING_USERNAME`, and `MLFLOW_TRACKING_PASSWORD`.
- Dependencies: root `pyproject.toml`.
- Run iteration count: `/brigit-automl:automl experiment run --project <project_name> --max-iter`; the skill default is 10.

## Validation

Run:

```bash
uv run automl --project <project_name> validate project
```

Check IDs beginning with `config.*` point back to `RUN_CONFIG` fields.
Check IDs beginning with `contracts.data_*` point to `DATA` in `config.py`.
Check IDs beginning with `contracts.eval_*` point to `EVAL` in `config.py`.
