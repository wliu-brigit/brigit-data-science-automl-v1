from __future__ import annotations

import copy

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from automl.model import BaseModel


class HomeCreditLogisticModel(BaseModel):
    """Home Credit fixture model with project-required preprocessing support."""

    target_column = "target"

    def __init__(self, *, max_iter: int = 200, random_state: int = 0) -> None:
        self.max_iter = max_iter
        self.random_state = random_state
        self.feature_columns: list[str] = []
        self.feature_cols: list[str] = []
        self.model = LogisticRegression(max_iter=max_iter, random_state=random_state)
        self.feature_registry = None
        self.preprocessor = None
        self.name = "homecredit_logistic"

    def fit(self, df_train: pd.DataFrame, registry=None, seed: int = 0) -> "HomeCreditLogisticModel":
        del seed
        self.feature_registry = copy.deepcopy(registry)
        target = self._target_column(df_train)
        required_entries = self.required_transformer_entries()
        required_columns = [
            column
            for _, _, columns in required_entries
            for column in columns
        ]
        numeric_columns = self._numeric_feature_columns(
            df_train,
            self.feature_registry,
            exclude={target, "SPLIT_PCT", *required_columns},
        )
        self.feature_cols = [*required_columns, *numeric_columns]
        self.feature_columns = list(self.feature_cols)
        if self.feature_registry is not None:
            self.feature_registry.set_flag(self.feature_registry.get_by_flag("feature"), "model", False)
            self.feature_registry.set_flag(self.feature_cols, "model", True)
        transformers = [
            *required_entries,
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer()),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            ),
        ]
        self.preprocessor = ColumnTransformer(transformers)
        X = self.preprocessor.fit_transform(df_train, df_train[target])
        self.model.fit(X, df_train[target])
        return self

    def transform(self, df: pd.DataFrame):
        return self.preprocessor.transform(df)

    def _predict(self, X):
        return self.model.predict_proba(X)[:, 1]

    def predict_proba(self, df: pd.DataFrame):
        return self.predict(context=None, model_input=df)

    def _target_column(self, df: pd.DataFrame) -> str:
        if self.target_column in df.columns:
            return self.target_column
        return "TARGET"

    def _numeric_feature_columns(
        self,
        df: pd.DataFrame,
        registry,
        *,
        exclude: set[str],
    ) -> list[str]:
        numeric_columns = list(df.select_dtypes(include="number").columns)
        numeric_column_set = set(numeric_columns)
        if registry is not None:
            selected = registry.get_by_flag("model")
            if selected:
                return [
                    column
                    for column in selected
                    if column in numeric_column_set and column not in exclude
                ]
        return [column for column in numeric_columns if column not in exclude]


# The runner imports projects.example_homecredit.model.MODEL_CLASS for
# project-baseline trial execution.
MODEL_CLASS = HomeCreditLogisticModel


__all__ = ["HomeCreditLogisticModel", "MODEL_CLASS"]
