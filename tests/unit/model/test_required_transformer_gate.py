import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from automl.data import FeatureRegistry
from automl.model import BaseModel, RequiredTransformer
from automl.project import ProjectConfig, Session
from automl.model import validate_model

pytestmark = pytest.mark.unit


class RequiredCategoryEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.fitted_ = True
        return self

    def transform(self, X):
        return np.zeros((len(X), 1), dtype=float)


class OtherEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.fitted_ = True
        return self

    def transform(self, X):
        return np.ones((len(X), 1), dtype=float)


def _fixture():
    df = pd.DataFrame(
        {
            "target": [0, 1, 0, 1],
            "category": ["a", "b", "a", "c"],
            "other": ["x", "y", "z", "x"],
            "value": [1.0, 2.0, 3.0, 4.0],
        }
    )
    registry = FeatureRegistry().build_from_df(df, target_column="target")
    return df, registry


def _session(requirements) -> Session:
    return Session(
        config=ProjectConfig(
            project_name="demo",
            required_transformers=requirements,
        )
    )


class BaseRequiredModel(BaseModel):
    required_name = "required_category"
    transformer_cls = RequiredCategoryEncoder
    transformer_columns = ["category"]
    wrap_pipeline = False
    no_column_transformer = False

    def fit(self, df_train, registry, seed=0):
        del seed
        self.feature_registry = registry
        self.name = "required_fixture"
        self.model = "constant"
        if self.no_column_transformer:
            self.preprocessor = "identity"
            return self
        column_transformer = ColumnTransformer(
            [
                (
                    self.required_name,
                    self.transformer_cls(),
                    self.transformer_columns,
                ),
                ("num", StandardScaler(), ["value"]),
            ]
        )
        self.preprocessor = (
            Pipeline([("prep", column_transformer)]) if self.wrap_pipeline else column_transformer
        )
        self.preprocessor.fit(df_train, df_train["target"])
        return self

    def transform(self, df):
        if isinstance(self.preprocessor, str):
            return df[["value"]].to_numpy()
        return self.preprocessor.transform(df)

    def _predict(self, X):
        return np.zeros(len(X), dtype=float)


class CompliantModel(BaseRequiredModel):
    pass


class MissingColumnTransformerModel(BaseRequiredModel):
    no_column_transformer = True


class WrappedColumnTransformerModel(BaseRequiredModel):
    wrap_pipeline = True


class WrongTransformerClassModel(BaseRequiredModel):
    transformer_cls = OtherEncoder


class MissingRequiredColumnsModel(BaseRequiredModel):
    transformer_columns = ["other"]


def _issues_for(cls, requirements):
    df, registry = _fixture()
    return validate_model(cls, df=df, registry=registry, session=_session(requirements)).issues


def _required_issue_checks(issues):
    return [
        issue.check for issue in issues if issue.check.startswith("model.required_transformers")
    ]


def test_required_transformer_gate_passes_compliant_top_level_column_transformer():
    issues = _issues_for(
        CompliantModel,
        [
            RequiredTransformer(
                name="required_category",
                transformer=RequiredCategoryEncoder(),
                input_cols=["category"],
            )
        ],
    )

    assert _required_issue_checks(issues) == []


@pytest.mark.parametrize(
    ("cls", "message"),
    [
        (MissingColumnTransformerModel, "ColumnTransformer"),
        (WrappedColumnTransformerModel, "top-level"),
        (WrongTransformerClassModel, "RequiredCategoryEncoder"),
        (MissingRequiredColumnsModel, "category"),
    ],
)
def test_required_transformer_gate_reports_noncompliant_models(cls, message):
    issues = _issues_for(
        cls,
        [
            RequiredTransformer(
                name="required_category",
                transformer=RequiredCategoryEncoder(),
                input_cols=["category"],
            )
        ],
    )

    assert _required_issue_checks(issues) == ["model.required_transformers"]
    assert message in issues[-1].message


def test_required_transformer_gate_noops_when_project_has_no_requirements():
    issues = _issues_for(MissingColumnTransformerModel, [])

    assert _required_issue_checks(issues) == []


def test_required_transformer_gate_reads_requirements_once(monkeypatch):
    calls = []
    requirements = [
        RequiredTransformer(
            name="required_category",
            transformer=RequiredCategoryEncoder(),
            input_cols=["category"],
        )
    ]

    def fake_requirements(session):
        calls.append(session)
        return requirements

    monkeypatch.setattr("automl.model.preprocessing._requirements", fake_requirements)

    issues = _issues_for(CompliantModel, requirements)

    assert _required_issue_checks(issues) == []
    assert calls == [_session(requirements)]
