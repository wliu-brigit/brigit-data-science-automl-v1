# Leakage Rules — Mutable Surface

## Coder write boundary

The Coder's `Write` and `Edit` tools are scoped to **`experiments/<this_trial>/model.py` only**.

**Allowed reads:**
- parent's `model.py`
- `projects/<project_name>/PROJECT_INSTRUCTIONS.md`
- `projects/<project_name>/config.py`
- `projects/<project_name>/data/queries/base_data.sql` when present
- `projects/<project_name>/data/queries/training_data.sql` when present
- `projects/<project_name>/data/pipeline.py` when present
- `projects/<project_name>/eval/metrics.py` when present
- Task-provided `data_context` from MLflow context
- `automl/model/base.py`
- `automl/data/features.py`

**Forbidden:**
- past trial artifacts or broad MLflow leaderboard context (would leak metrics into Coder reasoning)
- test data (with labels) directly
- writing or editing any file outside `experiments/<this_trial>/`

## Evaluation tool boundary

`run.py` is an immutable shim that calls `automl.runner.run_trial()`. The runner
passes labeled `df_test` only to the runner-owned evaluation step after
prediction; the LLM-written `model.py` receives `df_test_features_only` with the
target column dropped.

## Escalation

If the Coder believes a change outside the trial directory is needed, it must halt and report. The user (or a future Manager-with-broader-scope turn) handles it.
