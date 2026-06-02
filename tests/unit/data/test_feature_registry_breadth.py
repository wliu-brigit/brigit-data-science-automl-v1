import pandas as pd
import pytest

from automl.data import FeatureEntry, FeatureRegistry

pytestmark = pytest.mark.unit


def _registry() -> FeatureRegistry:
    return FeatureRegistry().build_from_df(
        pd.DataFrame(
            {
                "row_id": [1, 2, 3],
                "amount": [10.0, 20.5, 30.0],
                "flag": [True, False, True],
                "segment": ["a", "b", "a"],
                "target": [0, 1, 0],
            }
        ),
        target_column="target",
        metadata_cols=("row_id",),
    )


def test_feature_entry_exposes_derived_lineage_defaults():
    entry = FeatureEntry(name="amount", dtype=FeatureRegistry.NUM)

    assert entry.derived is False
    assert entry.source_columns == ()


def test_add_derived_tracks_model_lineage_and_source_validation():
    registry = _registry()

    registry.add_derived(
        "amount_log",
        FeatureRegistry.NUM,
        ("amount",),
        model=True,
        comments="log feature",
    )

    entry = registry.get("amount_log")
    assert entry.available is True
    assert entry.feature is True
    assert entry.model is True
    assert entry.derived is True
    assert entry.source_columns == ("amount",)
    assert entry.comments == "log feature"

    with pytest.raises(ValueError, match="amount_log"):
        registry.add_derived("amount_log", FeatureRegistry.NUM, ("amount",))
    with pytest.raises(KeyError, match="missing"):
        registry.add_derived("missing_source_feature", FeatureRegistry.NUM, ("missing",))


def test_registry_round_trips_derived_lineage_without_learning_flags():
    registry = _registry()
    registry.add_derived("amount_log", FeatureRegistry.NUM, ("amount",))

    frame = registry.to_dataframe()
    restored = FeatureRegistry.from_dataframe(frame)

    assert "golden" not in frame.columns
    assert "weak" not in frame.columns
    assert frame.loc[frame["name"] == "amount_log", "source_columns"].item() == '["amount"]'
    assert restored.get("amount_log").derived is True
    assert restored.get("amount_log").source_columns == ("amount",)


def test_registry_selection_flags_columns_and_dtype_helpers_are_model_facing_contract():
    registry = _registry()
    registry.add_derived("amount_log", FeatureRegistry.NUM, ("amount",))

    assert len(registry) == 6
    assert "amount" in registry
    assert "missing" not in registry
    assert registry.columns == ["amount", "amount_log", "flag", "row_id", "segment", "target"]
    assert "FeatureRegistry(total=6" in repr(registry)

    assert registry.get_by_flag("target") == ["target"]
    assert registry.get_by_flag("derived") == ["amount_log"]
    assert registry.get_by_dtype(FeatureRegistry.NUM) == [
        "amount",
        "amount_log",
        "row_id",
        "target",
    ]
    assert registry.get_by_dtype(FeatureRegistry.NUM, flag="model") == ["amount", "amount_log"]
    with pytest.raises(ValueError, match="golden"):
        registry.get_by_flag("golden")
    with pytest.raises(ValueError, match="datetime"):
        registry.get_by_dtype("datetime")

    selected = registry.select(
        pd.DataFrame(
            {
                "target": [0],
                "segment": ["z"],
                "amount": [1.0],
                "flag": [False],
                "row_id": [99],
                "amount_log": [0.0],
            }
        )
    )
    assert list(selected.columns) == registry.get_by_flag("feature")
    assert "target" not in selected.columns
    assert "row_id" not in selected.columns


def test_registry_set_flag_updates_existing_columns_and_rejects_bad_inputs():
    registry = _registry()

    registry.set_flag(registry.get_by_flag("feature"), "model", False)
    registry.set_flag(["amount", "flag"], "model", True)

    assert registry.get_by_flag("model") == ["amount", "flag"]
    with pytest.raises(KeyError, match="missing"):
        registry.set_flag(["missing"], "model", True)
    with pytest.raises(ValueError, match="golden"):
        registry.set_flag(["amount"], "golden", True)


def test_registry_cast_returns_copy_when_requested_and_preserves_null_categories():
    registry = _registry()
    raw = pd.DataFrame(
        {
            "amount": ["1.5", "bad"],
            "flag": ["true", "false"],
            "segment": ["x", None],
            "target": [0, 1],
        }
    )

    casted = registry.cast(raw, inplace=False)

    assert raw["amount"].tolist() == ["1.5", "bad"]
    assert casted["amount"].tolist()[0] == 1.5
    assert pd.isna(casted["amount"].tolist()[1])
    assert casted["flag"].tolist() == [1.0, 0.0]
    assert casted["segment"].tolist()[0] == "x"
    assert pd.isna(casted["segment"].tolist()[1])


def test_registry_comments_append_and_missing_comment_is_empty():
    registry = _registry()

    registry.add_comment("amount", "first")
    registry.add_comment(["amount", "segment"], "second")

    assert registry.get_comment("amount") == "first\nsecond"
    assert registry.get_comment("segment") == "second"
    assert registry.get_comment("missing") == ""
