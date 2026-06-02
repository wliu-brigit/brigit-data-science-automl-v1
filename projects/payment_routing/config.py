"""Typed project config for the payment-routing starter.

TASK.target and DATA.source.base_table are intentional <TBD_...> placeholders
held over until the project is configured for a specific Snowflake table.
Run ``uv run automl validate project --project payment_routing`` to see what needs
filling in. The detector in ``automl/validate/builtin/contract_checks.py``
keys on the ``<TBD`` prefix.
"""

from __future__ import annotations

from automl.data import DataSpec, SnowflakeSource
from automl.eval import Auc, EvalSpec
from automl.project import (
    BinaryClassification,
    ModelRoute,
    ModelsConfig,
    ProjectConfig,
    RunConfig,
    Splits,
)


TASK = BinaryClassification(target="<TBD_target_column>")

DATA = DataSpec(
    source=SnowflakeSource(
        base_table="<TBD_base_table>",
        base_data_sql="data/queries/base_data.sql",
        training_data_sql="data/queries/training_data.sql",
    ),
    exclude_cols=[],
    metadata_cols=[],
)

EVAL = EvalSpec(primary=Auc())

RUN_CONFIG = RunConfig(
    experiment_id="2026-05-07-payment-routing",
    splits=Splits(train=[(0, 80)], test=[(80, 100)]),
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
