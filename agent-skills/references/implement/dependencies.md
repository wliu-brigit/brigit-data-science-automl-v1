# Dependency Rules

Three rules govern what packages a trial may use:

## Rule A: Project venv is locked

`pyproject.toml` + `uv.lock` define the universe. All trials use identical pinned versions. Pickle compatibility across trials is guaranteed by construction.

## Rule B: LLM may only `import` dependencies already in `pyproject.toml`

The Coder receives an `allowed_dependencies` list derived from the project's lock file. Enforcement is at runtime — `uv run` fails with `ModuleNotFoundError` if an import is missing. The runner records `status="missing_dependency"`.

## Rule C: Adding a package is a Manager-level decision

If the manager decides a new package is needed, the run halts and surfaces the
exact `uv add <pkg>` command. After the user installs,
`/brigit-automl:automl experiment run --project <project_name>` resumes; MLflow records
the trial status and missing package.

## Missing-dependency flow

```
Coder runs trial -> ImportError
   ↓
MLflow trial: status="missing_dependency", missing_package="<name>"
   ↓
Coder returns to Manager: "Trial NN needs <pkg>. Halt requested."
   ↓
Manager surfaces clear `uv add <pkg>` command to user
   ↓
User installs, re-runs `/brigit-automl:automl experiment run --project <project_name>`
   ↓
Manager dispatches a fresh Task subagent for the same hypothesis
```
