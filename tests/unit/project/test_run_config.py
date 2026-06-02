import pytest

from automl.project import ModelRoute, ModelsConfig, RunConfig, Splits

pytestmark = pytest.mark.unit


def test_splits_default_to_train_and_test_ranges():
    splits = Splits()

    assert splits.resolve("train") == ((0, 80),)
    assert splits.resolve("test") == ((80, 100),)
    assert splits.train_buckets() == frozenset(range(0, 80))
    assert splits.test_buckets() == frozenset(range(80, 100))


def test_splits_support_free_form_named_ranges():
    splits = Splits({"fit": [(0, 60)], "holdout": [(60, 100)]})

    assert splits.resolve("fit") == ((0, 60),)
    assert splits.buckets("holdout") == frozenset(range(60, 100))


def test_splits_reject_overlap_within_a_split_name():
    with pytest.raises(ValueError, match="overlaps"):
        Splits({"train": [(0, 50), (25, 80)]})


def test_splits_reject_cross_name_overlap_and_name_both_slices():
    with pytest.raises(ValueError, match="train.*test|test.*train"):
        Splits({"train": [(0, 80)], "test": [(79, 100)]})


def test_splits_round_trip_through_dict_payload():
    splits = Splits({"fit": [(0, 50), (70, 80)], "holdout": [(80, 100)]})

    payload = splits.to_dict()
    restored = Splits.from_dict(payload)

    assert payload == {
        "ranges": {
            "fit": [[0, 50], [70, 80]],
            "holdout": [[80, 100]],
        }
    }
    assert restored.ranges == splits.ranges


def test_splits_raise_for_missing_pointer():
    splits = Splits()

    with pytest.raises(KeyError, match="validation"):
        splits.resolve("validation")


def test_run_config_exposes_train_and_eval_split_names():
    run_config = RunConfig(
        experiment_id="example-homecredit",
        splits=Splits(train=[(0, 80)], test=[(80, 100)]),
        models=ModelsConfig(
            manager=ModelRoute("sonnet", "medium"),
            proposer=ModelRoute("sonnet", "medium"),
            coder=ModelRoute("sonnet", "medium"),
        ),
        per_trial_seconds=600,
    )

    assert run_config.train_split == "train"
    assert run_config.eval_split == "test"
    assert run_config.splits.resolve(run_config.train_split) == ((0, 80),)
    assert run_config.splits.resolve(run_config.eval_split) == ((80, 100),)


def test_run_config_defaults_to_train_test_splits():
    run_config = RunConfig(
        experiment_id="example-homecredit",
        models=ModelsConfig(
            manager=ModelRoute("sonnet", "medium"),
            proposer=ModelRoute("sonnet", "medium"),
            coder=ModelRoute("sonnet", "medium"),
        ),
        per_trial_seconds=600,
    )

    assert run_config.splits.resolve("train") == ((0, 80),)
    assert run_config.splits.resolve("test") == ((80, 100),)


def test_run_config_rejects_retired_split_keyword():
    with pytest.raises(TypeError, match="split"):
        RunConfig(
            experiment_id="example-homecredit",
            split=Splits(train=[(0, 80)], test=[(80, 100)]),
            models=ModelsConfig(
                manager=ModelRoute("sonnet", "medium"),
                proposer=ModelRoute("sonnet", "medium"),
                coder=ModelRoute("sonnet", "medium"),
            ),
            per_trial_seconds=600,
        )
