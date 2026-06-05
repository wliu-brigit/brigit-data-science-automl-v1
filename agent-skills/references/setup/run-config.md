# projects/<project_name>/config.py — RUN_CONFIG

`RUN_CONFIG` is one of four module-level constants in `config.py`. It contains
only declarative run metadata. Values coupled to data logic live in `DATA`,
values coupled to metrics live in `EVAL`, and service locations live in `.env`.

## Required shape

```python
from automl.project import RunConfig, Splits, Where, ModelsConfig, ModelRoute

RUN_CONFIG = RunConfig(
    experiment_id="2026-Q2",
    splits=Splits(train=Where("SPLIT_PCT") < 80, test=Where("SPLIT_PCT") >= 80),
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
| `splits` | Named row-criteria over the materialized dataset, as `Where(...)` predicates. Ops: `== != < <= > >= .isin([...]) .notin([...]) .is_null() .not_null()`, composed with `& \| ~`. See "Splits" below. |
| `models` | Per-role model routing for the manager, proposer, and coder agents via `ModelsConfig` with `ModelRoute(model, effort)` for each. |
| `per_trial_seconds` | Timeout budget for a single trial execution. |

## Splits

A split is a named, durable row-criterion over the immutable materialized
dataset. `SPLIT_PCT` is an ordinary column — a deterministic 0–99 hash bucket
of each row's `split_group_key` — so the default 80/20 split is just
`Where("SPLIT_PCT") < 80` / `Where("SPLIT_PCT") >= 80`. Any column of the
persisted frame works the same way; time-based splits need no extra machinery:

```python
splits=Splits(
    train=Where("application_date") < "2026-03-01",
    test=(Where("application_date") >= "2026-03-01") & (Where("SPLIT_PCT") < 50),
)
```

- **Record, don't police.** Overlapping splits are legitimate methodology
  (full-data views, progressive train sets); the harness records exactly what
  each named split meant for any trial and enforces nothing about
  disjointness.
- **Column availability is the only requirement.** A criterion referencing a
  missing column fails loudly at load with the column name and the available
  columns.
- Rolling/backtesting windows are a family of named splits
  (`train_q1`, `test_q2`, ...).
- Criteria are data, not code: no lambdas. Trial contracts and eval
  identities serialize the predicate as a small JSON AST.

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
