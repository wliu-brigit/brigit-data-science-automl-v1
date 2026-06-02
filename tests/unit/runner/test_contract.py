from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from automl.data import FeatureRegistry
from automl.errors import RunnerError
from automl.model import BaseModel
from automl.runner.contract import validate_fitted_model

pytestmark = pytest.mark.unit


class GoodFittedModel(BaseModel):
    name = "good"
    preprocessor = object()
    model = object()

    def __init__(self, registry: FeatureRegistry) -> None:
        self.feature_registry = copy.deepcopy(registry)
        self.feature_registry.set_flag(self.feature_registry.get_by_flag("feature"), "model", False)
        self.feature_registry.set_flag(["score"], "model", True)
        self.feature_cols = ["score"]

    def fit(self, df_train, registry, seed=0):
        del df_train, registry, seed
        return self

    def transform(self, df):
        return df[["score"]].to_numpy()

    def _predict(self, X):
        return np.asarray(X[:, 0], dtype=float)


class OverridesPredictModel(GoodFittedModel):
    def predict(self, context=None, model_input=None):
        del context
        return np.asarray(model_input["score"], dtype=float)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target": [0, 1, 0],
            "score": [0.1, 0.8, 0.3],
            "amount": [10.0, 20.0, 30.0],
        }
    )


def _registry(df: pd.DataFrame | None = None) -> FeatureRegistry:
    return FeatureRegistry().build_from_df(
        df if df is not None else _frame(), target_column="target"
    )


def test_validate_fitted_model_accepts_model_with_registry_and_predict_contract():
    df = _frame()
    registry = _registry(df)
    model = GoodFittedModel(registry)

    validate_fitted_model(model, sample=df.head(2), registry=registry)


def test_validate_fitted_model_rejects_required_attrs_that_are_missing_or_none():
    df = _frame()
    registry = _registry(df)
    model = GoodFittedModel(registry)
    model.preprocessor = None

    with pytest.raises(RunnerError, match="preprocessor"):
        validate_fitted_model(model, sample=df.head(2), registry=registry)


def test_validate_fitted_model_exercises_transform_and_predict_contract():
    df = _frame()
    registry = _registry(df)
    model = GoodFittedModel(registry)

    def broken_transform(_df):
        raise RuntimeError("cannot transform")

    model.transform = broken_transform

    with pytest.raises(RunnerError, match="transform/predict"):
        validate_fitted_model(model, sample=df.head(2), registry=registry)


def test_validate_fitted_model_rejects_public_predict_override():
    df = _frame()
    registry = _registry(df)
    model = OverridesPredictModel(registry)

    with pytest.raises(RunnerError, match="BaseModel.predict"):
        validate_fitted_model(model, sample=df.head(2), registry=registry)


def test_validate_fitted_model_rejects_registry_that_drops_original_entries():
    df = _frame()
    fit_registry = _registry(df)
    model_registry = _registry(df[["target", "score"]])
    model = GoodFittedModel(model_registry)

    with pytest.raises(RunnerError, match="missing registry entr"):
        validate_fitted_model(model, sample=df.head(2), registry=fit_registry)


def test_validate_fitted_model_rejects_registry_that_is_not_model_annotated():
    df = _frame()
    fit_registry = _registry(df)
    model = GoodFittedModel(fit_registry)
    model.feature_registry = fit_registry

    with pytest.raises(RunnerError, match="deep-copy"):
        validate_fitted_model(model, sample=df.head(2), registry=fit_registry)


def test_validate_fitted_model_requires_model_flags_to_match_feature_cols():
    df = _frame()
    fit_registry = _registry(df)
    model = GoodFittedModel(fit_registry)
    model.feature_registry.set_flag(["amount"], "model", True)

    with pytest.raises(RunnerError, match="model=True"):
        validate_fitted_model(model, sample=df.head(2), registry=fit_registry)


def test_validate_fitted_model_requires_exactly_one_target():
    df = _frame()
    registry = _registry(df)
    registry.get("amount").target = True
    model = GoodFittedModel(registry)

    with pytest.raises(RunnerError, match="target columns must match"):
        validate_fitted_model(model, sample=df.head(2), registry=registry)
