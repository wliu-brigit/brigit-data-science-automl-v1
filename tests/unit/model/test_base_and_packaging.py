from pathlib import Path

import cloudpickle
import numpy as np
import pandas as pd
import pytest

import automl.model.packaging as packaging
from automl.data import FeatureRegistry
from automl.model import BaseModel, required_transformer_entries, save_model

pytestmark = pytest.mark.unit


class TinyModel(BaseModel):
    def fit(self, df_train, registry, seed=0):
        self.feature_registry = registry
        self.preprocessor = "identity"
        self.model = {"seed": seed}
        self.name = "tiny"
        self.columns = [column for column in df_train.columns if column != "target"]
        return self

    def transform(self, df):
        return df[self.columns].to_numpy()

    def _predict(self, X):
        return X.sum(axis=1)


def _registry(df: pd.DataFrame) -> FeatureRegistry:
    return FeatureRegistry().build_from_df(df, target_column="target")


def test_base_model_predict_uses_transform_and_predict_transformed():
    df = pd.DataFrame({"target": [0, 1], "a": [1.0, 2.0], "b": [10.0, 20.0]})
    model = TinyModel().fit(df, _registry(df), seed=12)

    transformed = model.transform(df)
    direct = model.predict_transformed(transformed)
    pyfunc = model.predict(context=None, model_input=df)

    assert direct.tolist() == [11.0, 22.0]
    assert np.asarray(pyfunc).tolist() == direct.tolist()
    assert model.feature_importances() is None
    assert model.training_report() is None


def test_save_model_cloudpickle_round_trips_model(tmp_path: Path):
    df = pd.DataFrame({"target": [0, 1], "a": [1.0, 2.0]})
    model = TinyModel().fit(df, _registry(df))
    path = tmp_path / "nested" / "model.pkl"

    save_model(model, path)
    with path.open("rb") as file_obj:
        restored = cloudpickle.load(file_obj)

    assert path.exists()
    assert isinstance(restored, TinyModel)
    assert restored.predict(None, df).tolist() == [1.0, 2.0]


def test_model_packaging_does_not_expose_path_load_api():
    assert not hasattr(save_model, "load")
    assert not hasattr(packaging, "load")
    assert packaging.__all__ == ["save_model"]


def test_required_transformer_entries_is_inert_for_phase_one():
    assert required_transformer_entries(session=None) == []
    assert TinyModel().required_transformer_entries(session=None) == []
