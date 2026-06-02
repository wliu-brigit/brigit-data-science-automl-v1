# Model contract — automl/model/base.py

The runbook ships a shared PyFunc-native `BaseModel` contract in `automl/model/base.py`. **You usually do not need to edit this file.**

## What it documents

`automl/model/base.py` is the reference for what every trial's `model.py` must satisfy. Trial code subclasses `BaseModel` directly:

```python
from automl.model import BaseModel
```

Projects do not carry a project-local model contract file. If a project needs
reusable trial helpers, put shared framework-level helpers under `automl/` and
project-specific recipe code under `projects/<project_name>/`.

Projects that provide a default project-baseline model expose it through
`MODEL_CLASS` in `projects/<project_name>/model/__init__.py`. When that route
is used, the runner imports `projects.<project_name>.model` and requires
`MODEL_CLASS` to be a class.

## When to override

Change the shared contract only if the AutoML framework itself needs a non-standard shape — e.g.:

- Pipelines that produce multi-output predictions.
- Models that need a custom `transform` interface (e.g. text + tabular fusion).
- Project-specific constraints on what `fit` may receive (e.g. cohort-aware splitting).

## What `BaseModel` already provides

The base class defines:

- `fit(df_train, registry, seed=0)` for fitting preprocessing and estimator
  state from the dataset `FeatureRegistry`.
- `transform(df)` for projecting raw features into the estimator matrix.
- `_predict(X)` as the trial-owned hook for scoring an already-transformed
  estimator matrix.
- `predict_transformed(X)` as the base-owned public helper for already
  transformed matrices.
- `predict(context, model_input, params=None)` as the base-owned MLflow PyFunc
  serving entry point.

Trial models implement `fit`, `transform`, and `_predict`. They must not
override public `predict` or `predict_transformed`; those are owned by the
shared contract so MLflow serving, runner evaluation, and diagnostics use the
same path. `predict` filters inputs with the trial `FeatureRegistry`, casts
them through that same registry, then calls `transform` and
`predict_transformed`. `_predict` is where the model decides whether to call
`predict_proba(X)[:, 1]`, `predict(X)`, a margin score, or another
model-specific scoring function.

Trial models must deep-copy the `registry` argument, select candidate inputs
from `registry.get_by_flag("feature")`, and set `model=True` on the columns
actually used by the estimator. Do not rebuild `FeatureRegistry` from
`df_train`; the dataset registry is the audit source of truth. If a model
removes a dataset feature, preserve existing comments and append a concise
model-side reason.

The root `automl/model/base.py` file is the source of truth. Read
that full code before writing a trial model; it carries the detailed comments
and examples that agents should follow.

The runner logs an MLflow model signature only as a shape/UI hint using feature
names from `FeatureRegistry`; it is not the semantic serving schema. Feature
types, feature availability, model-side removals, and serving casts are owned by
`FeatureRegistry`.

## If you're unsure

Leave it alone. The default contract works for binary classification,
regression, and multi-class as long as `_predict` returns the score vector that
the project's `projects/<project_name>/config.py` `EVAL` expects.
