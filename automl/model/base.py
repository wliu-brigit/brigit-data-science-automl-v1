"""Pure Python model contract for trial models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseModel(ABC):
    """Minimal estimator contract used by the runner and eval thin path."""

    @abstractmethod
    def fit(self, df_train, registry, seed: int = 0):
        """Fit the model and return ``self``."""

    @abstractmethod
    def transform(self, df):
        """Transform an input frame into model-ready features."""

    @abstractmethod
    def _predict(self, X):
        """Score model-ready features."""

    def required_transformer_entries(self, session: Any | None = None) -> list[Any]:
        from automl.model.preprocessing import required_transformer_entries

        return required_transformer_entries(session=session)

    def predict_transformed(self, X):
        return self._predict(X)

    def predict(self, context=None, model_input=None, params: dict[str, Any] | None = None):
        """MLflow-compatible prediction entry point without depending on MLflow."""

        del params
        if model_input is None:
            model_input = context
        model_input = self._prepare_model_input(model_input)
        return self.predict_transformed(self.transform(model_input))

    def _prepare_model_input(self, model_input):
        registry = getattr(self, "feature_registry", None)
        if registry is None or not hasattr(model_input, "copy"):
            return model_input
        prepared = model_input.copy()
        if hasattr(registry, "cast"):
            prepared = registry.cast(prepared, inplace=True)
        if hasattr(registry, "select"):
            return registry.select(prepared, flag="model")
        return prepared

    def feature_importances(self):
        return None

    def training_report(self):
        return None


__all__ = ["BaseModel"]
