"""Typed project config for fraud_anomaly_detection."""

from __future__ import annotations

from pathlib import Path

from automl.data import DataSpec, GCSParquetSource, LocalCSVSource, SnowflakeSource
from automl.eval import Auc, EvalSpec
from automl.project import (
    BinaryClassification,
    ModelRoute,
    ModelsConfig,
    ProjectConfig,
    RunConfig,
    Splits,
    Where,
)


PROJECT_DIR = Path(__file__).resolve().parent

TASK = BinaryClassification(target="<TBD_target_column>")

source = SnowflakeSource(
    base_table="<TBD_base_table>",
    base_table_sql="data/queries/base_table.sql",
    training_data_sql="data/queries/training_data.sql",
    unique_key="<TBD_unique_key>",
)

DATA = DataSpec(
    source=source,
    metadata_cols=[],
    exclude_cols=[],
    dry_run_rows=10_001,
)

EVAL = EvalSpec(primary=Auc())

RUN_CONFIG = RunConfig(
    experiment_id="TBD_experiment_id",
    splits=Splits(train=Where("SPLIT_PCT") < 80, test=Where("SPLIT_PCT") >= 80),
    models=ModelsConfig(
        manager=ModelRoute("sonnet", "medium"),
        proposer=ModelRoute("sonnet", "medium"),
        coder=ModelRoute("sonnet", "medium"),
    ),
    per_trial_seconds=600,
)

PROJECT_CONFIG = ProjectConfig.partial(
    task=TASK,
    data_spec=DATA,
    eval_spec=EVAL,
    run_config=RUN_CONFIG,
)
