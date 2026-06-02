---
name: validate
description: Run AutoML project validation before triggering trials.
disable-model-invocation: true
---

# Validate

Use before `/brigit-automl:automl experiment run --project <project_name>` and any time
`projects/<project_name>/config.py`, `projects/<project_name>/PROJECT_INSTRUCTIONS.md`,
`projects/<project_name>/data/pipeline.py` (optional subclass), or
`projects/<project_name>/eval/metrics.py` (optional custom metrics) changes.

From the repo root:

```bash
uv run automl --project <project_name> validate project
```

From inside `projects/<project_name>/...`, `uv run automl validate project`
uses the current project automatically.

The command prints a JSON `ValidationReport` and exits non-zero on errors.
If issues are reported, surface them verbatim (do not auto-fix project files).
The check IDs (`config.*`, `contracts.*`, `<project>.*`) point at the failing
surface; consult the matching reference doc:

- `config.*` -> [RUN_CONFIG](../../references/setup/run-config.md)
- `contracts.data_*` -> [data-pipeline](../../references/setup/data-pipeline.md)
- `contracts.eval_*` -> [evaluation-metric](../../references/setup/evaluation-metric.md)
- `model.*` -> [model-contract](../../references/setup/model-contract.md)

For a deeper dry-run sanity check:

```text
/brigit-automl:automl experiment run --project <project_name> --dry-run --max-iter 1
```

Tenets: idempotent, surface errors verbatim, point to docs instead of
auto-fixing project files.
