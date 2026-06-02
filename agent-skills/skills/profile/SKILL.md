---
name: profile
description: Build deterministic profile artifacts against the active AutoML dataset.
disable-model-invocation: true
---

# Profile

Produces MLflow-backed profile artifacts for the active dataset:

- `data_card.json`
- `data_observations.json`
- deterministic EDA charts such as label distribution, missingness, correlations, target segments, and leakage checks

Local files are temporary staging under `.cache/automl/tmp/` or MLflow-derived cache under `.cache/automl/mlflow/`.

## When to Run

Run once after an active dataset exists. Re-run when data, schema, or target changes.

## Steps

Run:

```bash
uv run automl --project <project_name> data profile
```

For dry-run context:

```bash
uv run automl --project <project_name> --dry-run data profile
```

This skill does not create datasets. If no active dataset exists, run `automl` or ask for an explicit data refresh first.

Optionally answer ad-hoc profile questions from temporary parquet files. Prompt before writing any extra result that is not part of the deterministic set.

End with a concise summary of row count, target distribution, top correlations, and key observations from MLflow context.
