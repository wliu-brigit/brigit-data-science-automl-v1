# Loop reference docs

Per-topic guides for the AutoML iteration loop (`profile -> propose -> implement -> validate -> review`). Each file is a constraint, schema, or protocol that shapes how the loop behaves at runtime.

| Topic | Read when |
|---|---|
| [protocol.md](protocol.md) | Authoring or debugging the loop's turn structure (Manager / Coder / runner roles) |
| [leakage.md](leakage.md) | Profiling features or proposing transforms (`profile`, `propose`, `implement`) |
| [timeouts.md](timeouts.md) | Configuring per-trial timeouts and per-session budgets |
| [mlflow-context.md](mlflow-context.md) | Reading MLflow-backed trial, overview, and learning context (`/brigit-automl:automl experiment run --project <project_name>`, `review`) |

Implement-specific rules (Coder constraints during `implement`) live in [`../implement/`](../implement/).

Setup-time guides live in [`../setup/`](../setup/).
