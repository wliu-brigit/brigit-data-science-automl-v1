# Project Entry Point Hard Cut-Over Design

**Date:** 2026-05-15
**Status:** Approved direction, pending implementation plan

## Goal

Complete the project entry-point redesign as a strict forward change. A runnable
project is defined only by `projects/<name>/project.py` plus
`PROJECT_INSTRUCTIONS.md`. There is no backward compatibility, no YAML fallback,
and no project-root `data.py` or `evaluation.py` entry point.

The implementation must make the framework, tests, docs, validation, and smoke
fixtures agree on that one rule.

## Design Boundary

`project.py` is the only project sentinel and source of typed project metadata.
It must define these module-level constants:

- `TASK`
- `DATA`
- `EVAL`
- `RUN_CONFIG`

`PROJECT_INSTRUCTIONS.md` remains the natural-language guidance file. Optional
custom code lives under module paths that mirror core structure, such as
`data/pipeline.py` and `eval/metrics.py`.

The retired files are not supported in active code paths:

- `automl_config.yaml`
- `projects/<name>/data.py`
- `projects/<name>/evaluation.py`
- `projects/<name>/metrics.py` at the project root
- `automl.eval.EvaluationSpec`
- `automl.core.config_schema` and `AutoMLConfig`

## Architecture

`ProjectContext` owns project discovery and metadata access. It should discover
projects by scanning only `projects/*/project.py`, expose a `project_path`
metadata field for diagnostics, and fail early when an explicit project lacks
`project.py`.

Runtime code should not check for `automl_config.yaml`. Commands such as data
preparation, cleanup, profiling, and trial creation should resolve a
`ProjectContext` and then use typed properties from `project.py`.

Validation should be the user-facing contract checker. It should import
`project.py` once per project and verify all four required constants with
concrete types. Placeholder detection should remain focused on values that make
a project incomplete, such as `<TBD_target_column>` or `<TBD_base_table>`.

Tests and fixtures should model the new shape. Integration and e2e fixtures
must create typed `project.py` projects rather than YAML-only projects.

## Data Flow

Project resolution:

1. Start from an explicit `--project-root`, current working directory, or active
   project context.
2. Locate a repo root by finding `projects/*/project.py`.
3. Resolve the active project by explicit name, cwd inference, or the single
   configured project.
4. Return a `ProjectContext` whose project metadata points at `project.py`.

Runtime loading:

1. `ProjectContext` imports `projects.<name>.project`.
2. `ctx.task_object`, `ctx.data_spec`, `ctx.evaluation_spec`, and
   `ctx._run_config()` read typed constants from that module.
3. `build_pipeline(ctx)` uses `TASK`, `DATA`, and `RUN_CONFIG.split`.
4. Runner and data-prep paths use `build_pipeline(ctx)`; no project-local
   forwarding functions are called.

## Error Handling

Missing `project.py` is a configuration error, not a fallback trigger. Messages
should name the required path: `projects/<name>/project.py`.

Missing or mistyped constants should fail with direct messages:

- `projects.<name>.project must define TASK`
- `projects.<name>.project DATA must be a DataSpec`
- `projects.<name>.project EVAL must be an EvalSpec`
- `projects.<name>.project RUN_CONFIG must be a RunConfig`

Invalid model route effort should fail during `RunConfig` construction. Allowed
values are `low`, `medium`, and `high`.

## Testing Requirements

The implementation is complete only when these pass:

- `uv run python -m automl.data.prepare --project-root . --project example_homecredit --dry-run`
  no longer fails because YAML is missing.
- `uv run pytest tests/unit tests/contracts -q`
- `uv run pytest tests/integration -q`

The e2e fixture should import and inspect `projects.test_homecredit.project`.
It should not depend on `data.py`, `evaluation.py`, or `automl_config.yaml`.

Add or update ratchet tests so active runtime, setup, validation, launcher, and
fixture paths cannot silently reintroduce the retired entry-point files.

## Out Of Scope

Do not add a YAML-to-Python converter. Do not add compatibility aliases. Do not
introduce a new project definition abstraction unless it removes existing code;
`ProjectContext` is sufficient for this cleanup.
