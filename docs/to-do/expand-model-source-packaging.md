# Expand Model Source Packaging

## Status

Priority: P2. This is valuable for native human and agent authoring once the
core loop contract is tightened, but it is not blocking the current strict
`project.py` cutover or the example Home Credit workflow.

This note is a future-work assessment. It captures the goal, scope, and current reasoning so a future implementer has useful context, but it is not a substitute for code inspection and due diligence before implementation.

The goal and scope below should be treated as stable unless the product direction changes. The implementation details are recommendations based on the current code shape and should be re-validated before coding.

## Problem

The current human authoring path promotes one Python file as the model source.
The CLI promotion workflow calls `trial.create(model_source=...)` and then
`runner.run_trial(...)`; trial itself only authors the draft. This works for
simple models, but it pushes users toward one large `model.py` file or brittle
notebook source extraction.

That is limiting for both human and AutoML-authored models:

- Humans often want to split model code into helpers, preprocessing utilities, and the main model class.
- AutoML agents can produce cleaner and more maintainable code when they can use a small source tree instead of a single file.
- Notebook-first workflows become brittle if we rely on `inspect.getsource()` to extract a class definition from kernel state.
- Local scratch helpers may work during exploration but fail after promotion or MLflow packaging if import semantics are not explicit.

The current system also excludes `experiments/` from the MLflow code bundle, which means helper files placed beside a promoted trial's `model.py` are not a reliable deployment path.

## Goal

Support a native model source unit that can contain more than one file, while preserving one clear entrypoint:

```text
source/
  model.py
  helpers.py
  preprocessing.py
```

`source/model.py` remains the required entrypoint and must expose `Model`. Model validation, pre-fit checks, full trial execution, MLflow logging, and deployment validation should continue to operate through that entrypoint.

The desired user-facing flow is:

```text
create or edit a model source directory
validate source/model.py
run optional local notebook experiments
promote the source directory
runner executes the same source directory
MLflow logs the same source directory
```

This should create one path for human and AutoML models. AutoML can write a source directory. Humans can write the same kind of source directory from a notebook, editor, or generated draft.

## Non-Goals

Do not make arbitrary notebook state part of the promoted model.

Do not support imports from notebook or scratch locations after promotion.

Do not make the entire experiment directory part of the model source. Experiment directories contain generated outputs such as `data/`, `eval/`, `features/`, `validation/`, `.cache/`, `code_bundle/`, `model.pkl`, `manifest.json`, and `result.json`. Treating the whole experiment directory as source would make packaging unclear and fragile.

Do not add compatibility layers for old trial layouts unless explicitly re-scoped. This project is still moving forward-only.

## Current Behavior To Re-Check

At the time this note was written:

- The CLI promotion workflow copies only one file through `trial.create(model_source=...)`.
- Local trial drafts live under `projects/<project>/experiments/<slug>/`.
- The runner expects a model file and materializes it into a hashed module under trial cache before import.
- MLflow code bundling includes `automl/`, `projects/`, and a single hashed trial model module.
- `experiments/` is excluded from the bundled `projects/` tree.
- Sibling helper files next to trial `model.py` are not a reliable path for either local runner import or MLflow deployment.

Future implementers should verify these points before starting because the runner and MLflow packaging code may have changed.

## Recommended Direction

Introduce an explicit model source directory instead of expanding the meaning of the experiment root.

Recommended promoted layout:

```text
projects/<project>/experiments/<slug>/
  run.py
  metadata.json
  source/
    model.py
    helpers.py
    preprocessing.py
```

Recommended scratch layout:

```text
projects/<project>/notebooks/scratch/<draft_name>/
  model.py
  helpers.py
```

The source directory is the unit being validated, promoted, run, and logged. `source/model.py` is the entrypoint. Everything else in `source/` is model-local support code.

## Import Rules

Allow package-relative imports inside the source directory:

```python
from .helpers import make_features
from .preprocessing import build_preprocessor
```

Allow stable absolute imports from committed project or library modules:

```python
from automl.core.base_model import BaseModel
from projects.example_homecredit.modeling_utils import normalize_homecredit_features
```

Reject or strongly discourage scratch, notebook, experiment, and `sys.path` dependent imports:

```python
from helpers import make_features
from notebooks.scratch.helpers import make_features
from projects.example_homecredit.experiments.some_slug.source.helpers import make_features
import sys; sys.path.append(...)
```

The core rule is that model-local helpers are relative to the source package, and reusable project helpers are absolute imports from stable committed project modules.

## Implementation Areas

### Trial Creation And Promotion

`trial.create` should accept a model source path. The CLI promotion workflow may
pass either a single file or a directory, but the internal promoted layout should
normalize to `source/model.py`.

If a single file is provided, copy it to:

```text
experiments/<slug>/source/model.py
```

If a directory is provided, copy the directory to:

```text
experiments/<slug>/source/
```

Require:

```text
source/model.py
```

and require that it exposes `Model`.

### Runner Import

The runner should import `source/model.py` as part of a generated package, not as a loose top-level module. This is what makes package-relative imports like `from .helpers import ...` reliable.

The generated package name should be deterministic enough for debugging and unique enough to avoid `sys.modules` collisions, for example using trial id plus source hash.

### Validation

The validation contract should stay conceptually the same:

```python
validate.model(Model, sample_from=project)
```

The loading step changes. Instead of validating a class extracted from a notebook or a single loose file, helper APIs can load `source/model.py` from a source directory, retrieve `Model`, and pass it into the existing validation contract.

### MLflow Code Bundle

The MLflow bundle must include the full model source directory, not only the entrypoint file. The pyfunc load path must preserve the same package-relative imports used during local validation and trial execution.

The source tree should also be logged as artifacts under a clear path, likely:

```text
source/
```

### Manifest And Inspection

The trial manifest should record source entrypoint and source tree metadata. Recommended fields:

```json
{
  "model": {
    "source_entrypoint": "source/model.py",
    "source_tree_artifact": "source/",
    "source_hash": "...",
    "source_files": [
      "source/model.py",
      "source/helpers.py"
    ]
  }
}
```

This helps later inspection answer which code was actually used, not just which trial folder ran.

### Notebook Authoring

The notebook authoring path should favor writing a real source directory, not extracting a class from notebook kernel state.

Recommended notebook pattern:

```python
%%writefile scratch/my_model/model.py
from automl.core.base_model import BaseModel
from .helpers import make_features

class Model(BaseModel):
    ...
```

Then import and validate the source directory. This gives notebook users a native workflow while keeping the promoted artifact explicit and reproducible.

Tagged notebook cell extraction could be considered later, but it has hidden-state risks because the notebook must be saved before extraction and cell tags are less visible in VS Code/Jupyter.

## Benefits

This change would make the model authoring surface more native:

- Human and AutoML models share one source directory abstraction.
- Complex model code can be split into small files.
- Notebook workflows can remain convenient without relying on kernel source extraction.
- MLflow artifacts can preserve the actual source tree.
- Promotion becomes more reproducible because validation, runner execution, and MLflow packaging all consume the same source unit.
- Project-level reusable helpers remain separate from model-local helpers.

## Risks

The main risk is import behavior. If source directories are imported inconsistently across notebook validation, runner execution, and MLflow pyfunc loading, a model may validate locally but fail after logging.

The second risk is over-broad packaging. Copying the experiment root or scratch root would pull generated files, caches, data, or notebook artifacts into the model bundle. The source directory must be explicit and bounded.

The third risk is confusing public API shape. The API should make it obvious whether a user is promoting one file or a source directory, and it should normalize both into the same internal layout.

## Suggested Acceptance Criteria

- A model source directory with `model.py` and `helpers.py` can be validated from a notebook.
- `source/model.py` can use `from .helpers import ...`.
- Promoting that directory creates `projects/<project>/experiments/<slug>/source/`.
- The runner imports and runs the model through `source/model.py`.
- The MLflow pyfunc model can be loaded in a fresh process and can predict successfully.
- The MLflow run logs the full source tree as artifacts.
- The manifest records the entrypoint and source files.
- Imports from scratch/notebook/experiment absolute paths are rejected or fail with a clear message.

## Recommended Next Step

Before implementation, inspect the current trial promotion, runner model import, validation helper, and MLflow code bundle paths together. Then make this as one coordinated change rather than incrementally supporting helper files in only one layer.
