# Setup reference docs

Per-topic guides for the `setup` skill walkthrough. Each file is loaded by the orchestrator only when the corresponding step is active - progressive disclosure.

| Topic | Read when |
|---|---|
| [snowflake.md](snowflake.md) | Configuring Snowflake credentials in `.env` |
| [gcs.md](gcs.md) | Configuring GCS auth (Application Default Credentials) |
| [mlflow.md](mlflow.md) | Configuring the MLflow tracking URI + auth |
| [run-config.md](run-config.md) | Filling in `RUN_CONFIG` in `projects/<project_name>/config.py` |
| [data-pipeline.md](data-pipeline.md) | Wiring `DATA` in `projects/<project_name>/config.py` to your data source |
| [evaluation-metric.md](evaluation-metric.md) | Defining or customizing `EVAL` in `projects/<project_name>/config.py` |
| [model-contract.md](model-contract.md) | Overriding the default model contract (rare) |
| [project-instructions.md](project-instructions.md) | Filling in `projects/<project_name>/PROJECT_INSTRUCTIONS.md` |
