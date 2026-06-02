---
name: propose
description: Manual proposal wrapper that asks automl-proposer for one validated Proposal. Ensures the routed MLflow overview run exists before delegating to the read-only proposer agent.
disable-model-invocation: true
---

# Propose

Manual manager turn. Use the `automl-proposer` agent to return exactly one `Proposal` JSON object or a stop JSON.

## Inputs

- `projects/<project_name>/config.py`: experiment ID, model routing, per-trial timeout, target,
  data source declaration, and evaluation metric (`RUN_CONFIG`, `TASK`, `DATA`, `EVAL`).
- `projects/<project_name>/PROJECT_INSTRUCTIONS.md`: user-editable direction, refreshed each turn.
- `projects/<project_name>/data/queries/`: active project SQL context when present.
- MLflow context from `uv run automl --project <project_name> experiment proposer-context`.
- `allowed_dependencies`: dependency names parsed from project `pyproject.toml`.
- `dry_run`: selects the dry-run MLflow route when true.

## Steps

This wrapper is not purely read-only. The delegated `automl-proposer` agent is read-only, but the MLflow context command below ensures routed overview runs exist. Do not refresh data, materialize datasets, launch trials, or write proposal artifacts from `propose`.

1. Read the project config and project instructions.
2. Run allowed-dependency context before asking the proposer:
   ```bash
   uv run automl --project <project_name> --project-root <project_root> project deps
   ```
3. Render MLflow context:
   ```bash
   uv run automl --project <project_name> --project-root <project_root> experiment proposer-context
   ```
   Place `--dry-run` before `experiment` for dry-run proposals.
4. Ask `automl-proposer` for one JSON response.
5. If the response has `action="stop"`, return the stop reason and do not validate it as a `Proposal`.
6. Otherwise validate required `Proposal` fields before passing it to implementation.

## Reasoning Protocol For The Proposer

1. Cold start: if the MLflow context has no successful trials, propose a baseline:
   ```json
   {
     "schema_version": 2,
     "slug": "baseline",
     "strategy": "baseline",
     "hypothesis": "Establish reference using a simple model on the current feature pool.",
     "implementation_plan": [
       "Train a simple baseline model using registry-selected feature columns.",
       "Do not add feature engineering in this trial."
     ],
     "constraints": [
       "Do not read test data directly.",
       "Do not change target or primary metric."
     ],
     "required_dependencies": [
       "pandas",
       "numpy",
       "scikit-learn"
     ]
   }
   ```
2. The seed (starting `model.py`) is selected by `trial.create` via a metric
   query at creation time. Do not pick a parent. You may include `seed_hint` as
   `auto`, `best`, `latest`, or a strategy selector such as
   `"strategy:feature_engineering"` only when the proposal explicitly wants a
   non-default seed. Never include a run ID.
3. Ensembles are opt-in for latency-sensitive runs. Do not propose `strategy:
   "ensemble"` unless the user explicitly asks, `projects/<project_name>/PROJECT_INSTRUCTIONS.md`
   explicitly permits latency-heavy models, or constraints state that online
   latency is not a concern.
4. Diagnose gaps across `baseline`, `hyperparameter_tuning`, `feature_engineering`, `new_model`, and `debug`.
5. Use `data_context` to identify unexploited signal: predictive missingness, high-cardinality categoricals, suspicious correlations, weak feature groups, or profile observations.
6. Form one atomic, falsifiable hypothesis. Avoid proposals that change model family, feature engineering, tuning, and ensembling all at once.
7. Budget, time, and failure-stop decisions belong to the AutoML orchestrator, not the proposer. Return a stop JSON only when the rendered context explicitly asks for a stop recommendation.

## Proposal Shape

Required:

```json
{
  "schema_version": 2,
  "slug": "xgb_missingness_flags",
  "strategy": "feature_engineering",
  "hypothesis": "Missingness in EXT_SOURCE fields carries predictive signal.",
  "implementation_plan": [
    "Add binary missingness indicators for EXT_SOURCE_2 and EXT_SOURCE_3 before imputation.",
     "Keep the seeded model family unchanged."
  ],
  "constraints": [
    "Do not read test data directly.",
    "Do not change target or primary metric."
  ],
  "required_dependencies": [
    "pandas",
    "numpy",
    "xgboost"
  ]
}
```

Optional:

```json
{
  "rationale": "Profile context shows target spread by EXT_SOURCE missingness.",
  "evidence": ["EXT_SOURCE_3 missingness differs by target in profile observations."],
  "data_checks": ["Confirm EXT_SOURCE_2 and EXT_SOURCE_3 exist in the active dataset registry."],
  "risk_notes": ["Avoid label-derived encodings."],
  "seed_hint": "strategy:feature_engineering"
}
```

## Rules

- The proposer is read-only.
- Do not read past trials' `model.py` or local trial artifacts.
- Use MLflow-owned `data_context.active_dataset` as the authoritative dataset description.
- Every proposed package in `required_dependencies` must appear in `allowed_dependencies`; choose a supported alternative instead of proposing an uninstalled package.
- `slug` must be short lowercase snake_case, start with a letter, and must not
  include the numeric trial prefix.
- Ensembles are opt-in; avoid them by default because latency is critical unless the user explicitly asks for one.
- Keep hypotheses atomic and falsifiable.
- Do not make budget, time, or failure-stop decisions unless the rendered context explicitly asks for a stop recommendation.
