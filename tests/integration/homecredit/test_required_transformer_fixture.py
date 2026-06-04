import copy
from pathlib import Path

import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from automl.data import build_dataset
from automl.model import BaseModel
from automl.project import clear_session, use_project
from automl.model import validate_model
from projects.example_homecredit.model import HomeCreditLogisticModel

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[3]


class OmittingRequiredTransformerModel(BaseModel):
    target_column = "target"

    def fit(self, df_train: pd.DataFrame, registry=None, seed: int = 0):
        del seed
        self.feature_registry = copy.deepcopy(registry)
        self.feature_columns = [
            column
            for column in df_train.select_dtypes(include="number").columns
            if column not in {self.target_column, "SPLIT_PCT", "SK_ID_CURR", "sk_id_curr"}
        ]
        self.feature_cols = list(self.feature_columns)
        self.feature_registry.set_flag(self.feature_registry.get_by_flag("feature"), "model", False)
        self.feature_registry.set_flag(self.feature_cols, "model", True)
        self.imputer = SimpleImputer()
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(self.imputer.fit_transform(df_train[self.feature_columns]))
        self.model = LogisticRegression(max_iter=200, random_state=0)
        self.model.fit(X, df_train[self.target_column])
        self.preprocessor = {"imputer": self.imputer, "scaler": self.scaler}
        self.name = "omitting_required_transformer"
        return self

    def transform(self, df: pd.DataFrame):
        return self.scaler.transform(self.imputer.transform(df[self.feature_columns]))

    def _predict(self, X):
        return self.model.predict_proba(X)[:, 1]


def test_homecredit_model_validates_with_declared_woe_requirement(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "file:///tmp/mlruns")
    active = use_project("example_homecredit", repo_root=REPO_ROOT, dry_run=True)
    try:
        loaded = build_dataset(session=active)
        report = validate_model(HomeCreditLogisticModel, df=loaded.df, registry=loaded.registry)
    finally:
        clear_session()

    assert report.passed
    assert report.issues == []


def test_homecredit_model_omitting_required_transformer_fails_validation(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "file:///tmp/mlruns")
    active = use_project("example_homecredit", repo_root=REPO_ROOT, dry_run=True)
    try:
        loaded = build_dataset(session=active)
        report = validate_model(
            OmittingRequiredTransformerModel,
            df=loaded.df,
            registry=loaded.registry,
        )
    finally:
        clear_session()

    assert not report.passed
    assert any(issue.check == "model.required_transformers" for issue in report.issues)
