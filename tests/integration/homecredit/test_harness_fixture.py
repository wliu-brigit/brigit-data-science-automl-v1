from pathlib import Path

import pytest

from automl.data import DataSpec, LocalCSVSource, build_dataset
from automl.eval import Auc, EvalSpec
from automl.model import BaseModel
from automl.project import ProjectConfig, Session
from automl.validate import model as validate_model
from projects.example_homecredit.config import PROJECT_CONFIG
from projects.example_homecredit.model import MODEL_CLASS

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_homecredit_project_config_imports_new_four_layer_contract():
    assert isinstance(PROJECT_CONFIG, ProjectConfig)
    assert isinstance(PROJECT_CONFIG.data_spec, DataSpec)
    assert isinstance(PROJECT_CONFIG.data_spec.source, LocalCSVSource)
    assert isinstance(PROJECT_CONFIG.eval_spec, EvalSpec)
    assert isinstance(PROJECT_CONFIG.eval_spec.primary, Auc)
    assert PROJECT_CONFIG.run_config.train_split == "train"
    assert PROJECT_CONFIG.run_config.eval_split == "test"
    assert "application_train_sample.csv" in str(PROJECT_CONFIG.data_spec.source.csv_path)


def test_homecredit_model_class_conforms_to_base_model_contract():
    assert issubclass(MODEL_CLASS, BaseModel)


def test_homecredit_model_validates_against_real_built_dataset_with_registry(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "file:///tmp/mlruns")
    config = ProjectConfig.load("example_homecredit", repo_root=REPO_ROOT)
    loaded = build_dataset(session=Session(config=config, dry_run=True))

    report = validate_model(MODEL_CLASS, df=loaded.df, registry=loaded.registry)

    assert report.passed
    assert report.issues == []
