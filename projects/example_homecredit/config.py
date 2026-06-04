"""Home Credit example project config for the four-layer refactor."""

from __future__ import annotations

from pathlib import Path

from automl.data import DataSpec, LocalCSVSource
from automl.eval import Auc, EvalSpec
from automl.model import RequiredTransformer
from automl.project import (
    BinaryClassification,
    ModelRoute,
    ModelsConfig,
    ProjectConfig,
    RunConfig,
    Splits,
)
from projects.example_homecredit.model.preprocessing import WOEEncoder


PROJECT_DIR = Path(__file__).resolve().parent
SAMPLE_CSV = PROJECT_DIR / "data" / "application_train_sample.csv"
HASH_KEY = "SK_ID_CURR"

TASK = BinaryClassification(target="TARGET")
DATA = DataSpec(
    source=LocalCSVSource(csv_path=SAMPLE_CSV, hash_key=HASH_KEY),
    metadata_cols=(HASH_KEY,),
    dry_run_rows=100,
)
EVAL = EvalSpec(primary=Auc())
RUN_CONFIG = RunConfig(
    experiment_id="example-homecredit",
    splits=Splits(train=[(0, 80)], test=[(80, 100)]),
    models=ModelsConfig(
        manager=ModelRoute("opus", "high"),
        proposer=ModelRoute("opus", "high"),
        coder=ModelRoute("opus", "high"),
    ),
    per_trial_seconds=600,
    train_split="train",
    eval_split="test",
)
REQUIRED_TRANSFORMERS = [
    RequiredTransformer(
        name="homecredit_organization_woe",
        transformer=WOEEncoder(),
        input_cols=["organization_type"],
    )
]

# ProjectConfig is loaded into Session.config by automl.use_project(...).
# Domain calls should receive the Session, not re-read config globals.
PROJECT_CONFIG = ProjectConfig.partial(
    task=TASK,
    data_spec=DATA,
    eval_spec=EVAL,
    run_config=RUN_CONFIG,
    required_transformers=REQUIRED_TRANSFORMERS,
)
