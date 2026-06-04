import pandas as pd
import pytest

from automl.data import FeatureRegistry
from automl.model import BaseModel, validate_model
from automl.validate import ValidationReport

pytestmark = pytest.mark.unit


class GoodModel(BaseModel):
    def fit(self, df_train, registry, seed=0):
        self.feature_registry = registry
        self.preprocessor = "identity"
        self.model = "constant"
        self.name = "good"
        return self

    def transform(self, df):
        return df[["value"]].to_numpy()

    def _predict(self, X):
        return X[:, 0] * 0


class FitBrokenModel(GoodModel):
    def fit(self, df_train, registry, seed=0):
        raise RuntimeError("fit exploded")


class PredictBrokenModel(GoodModel):
    def _predict(self, X):
        raise RuntimeError("predict exploded")


class MissingAttrsModel(GoodModel):
    def fit(self, df_train, registry, seed=0):
        return self


class NotAModel:
    pass


def _fixture():
    df = pd.DataFrame({"target": [0, 1, 0, 1], "value": [1.0, 2.0, 3.0, 4.0]})
    registry = FeatureRegistry().build_from_df(df, target_column="target")
    return df, registry


def test_validate_model_passes_good_basemodel():
    df, registry = _fixture()

    report = validate_model(GoodModel, df=df, registry=registry)

    assert isinstance(report, ValidationReport)
    assert report.passed
    assert report.issues == []


def test_validate_model_reports_subclass_fit_predict_and_post_fit_failures():
    df, registry = _fixture()

    not_model = validate_model(NotAModel, df=df, registry=registry)
    fit_broken = validate_model(FitBrokenModel, df=df, registry=registry)
    predict_broken = validate_model(PredictBrokenModel, df=df, registry=registry)
    missing_attrs = validate_model(MissingAttrsModel, df=df, registry=registry)

    assert not not_model.passed
    assert [issue.check for issue in not_model.issues] == ["model.subclass_basemodel"]
    assert any(issue.check == "model.fit_succeeds" for issue in fit_broken.issues)
    assert any("fit exploded" in issue.message for issue in fit_broken.issues)
    assert any(issue.check == "model.predict_succeeds" for issue in predict_broken.issues)
    assert any("predict exploded" in issue.message for issue in predict_broken.issues)
    assert any(issue.check == "model.post_fit_attrs_set" for issue in missing_attrs.issues)
