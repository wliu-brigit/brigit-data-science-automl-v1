import pytest
from sklearn.preprocessing import StandardScaler

from automl.model import BaseModel, RequiredTransformer
from automl.model.preprocessing import (
    describe_required_transformers,
    required_transformer_entries,
)
from automl.project import ProjectConfig, Session

pytestmark = pytest.mark.unit


class TinyModel(BaseModel):
    def fit(self, df_train, registry, seed=0):
        self.feature_registry = registry
        self.preprocessor = "identity"
        self.model = "constant"
        self.name = "tiny"
        return self

    def transform(self, df):
        return df

    def _predict(self, X):
        return [0.0] * len(X)


def _session_with(requirements) -> Session:
    return Session(
        config=ProjectConfig(
            project_name="demo",
            required_transformers=requirements,
        )
    )


def test_required_transformer_stores_columns_as_tuple():
    transformer = StandardScaler()

    requirement = RequiredTransformer(
        name="homecredit_organization_woe",
        transformer=transformer,
        input_cols=["organization_type"],
    )

    assert requirement.name == "homecredit_organization_woe"
    assert requirement.transformer is transformer
    assert requirement.input_cols == ("organization_type",)


def test_describe_required_transformers_returns_stable_handoff_dicts():
    active = _session_with(
        [
            RequiredTransformer(
                name="amount_scaler",
                transformer=StandardScaler(),
                input_cols=["amount"],
            )
        ]
    )

    assert describe_required_transformers(session=active) == [
        {
            "name": "amount_scaler",
            "type": "StandardScaler",
            "import_path": "sklearn.preprocessing._data.StandardScaler",
            "columns": ["amount"],
        }
    ]


def test_empty_required_transformers_return_empty_lists():
    assert describe_required_transformers(session=_session_with(None)) == []
    assert describe_required_transformers(session=_session_with([])) == []
    assert required_transformer_entries(session=_session_with([])) == []
    assert required_transformer_entries(session=None) == []


def test_required_transformer_entries_clone_config_transformers():
    declared = StandardScaler()
    active = _session_with(
        [
            RequiredTransformer(
                name="amount_scaler",
                transformer=declared,
                input_cols=["amount"],
            )
        ]
    )

    entries = TinyModel().required_transformer_entries(session=active)

    assert len(entries) == 1
    name, cloned, columns = entries[0]
    assert name == "amount_scaler"
    assert isinstance(cloned, StandardScaler)
    assert cloned is not declared
    assert columns == ["amount"]


def test_project_config_rejects_invalid_required_transformer_declarations():
    with pytest.raises(TypeError, match="required_transformers"):
        ProjectConfig(required_transformers="bad")
    with pytest.raises(TypeError, match="RequiredTransformer"):
        ProjectConfig(required_transformers=[object()])
