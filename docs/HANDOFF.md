# Handoff — continue here

The running note of where the last working session left off. **Read this first
when resuming**, then [`README.md`](README.md) for the docs lifecycle. Keep it to
*current state + next actions* (git history is the changelog, not this).

**Last updated:** 2026-06-01

## Where things stand

- Branch `refactor/four-layer` is **316 commits ahead of `main`, 0 behind** — a
  clean **fast-forward merge** is available. `main` is not checked out anywhere.
- **This clone (`automl_dev-refactor/`) is canonical.** `automl/` and `automl_dev/`
  are stale snapshots — ignore them.
- Full local suite green:
  `uv run pytest tests/unit tests/integration tests/contracts` → **471 passed**.
  The live notebook e2e is gated behind `AUTOML_E2E_NOTEBOOKS=1`.
- Working tree carries local run-state in the two `example_homecredit` example
  notebooks (`0_` outputs only; `1_` outputs + a `use_project()` edit) — intentional,
  not yet committed.

## Next actions

1. **Merge to `main`** — fast-forward (316 ahead, 0 behind).
2. **Parked:** untangle the MLflow-seam import cycles — extract shared value-types
   into a low contracts layer so `mlflow ↔ eval/project/trial` stops cycling.
3. **Forward work:** `docs/to-do/agent-orchestration/`.

## Gotchas (don't relitigate)

- Repo-local plugins don't flag-free auto-load in Claude Code (v2.1.159). Load via
  **`--plugin-dir agent-skills`** (the loop does this) or symlink `agent-skills` into
  `~/.claude/skills/`. The `agents` custom-path manifest field isn't honored — agents
  must live at the plugin's default `agents/`.
- **The Jupyter kernel must point at THIS clone's `.venv`.** VSCode can launch a
  sibling clone's venv even when you pick "Brigit AutoML (.venv)" — confirm with
  `import sys; print(sys.executable)` in cell 1 (`from automl import experiment`
  failing means the kernel is bound to the wrong clone).
