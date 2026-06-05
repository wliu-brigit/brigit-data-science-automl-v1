import pytest

from automl.project import ModelRoute, ModelsConfig, RunConfig, Splits, Where

pytestmark = pytest.mark.unit


def test_splits_default_to_train_and_test_predicates():
    splits = Splits()

    assert splits.resolve("train").to_dict() == {
        "op": "<",
        "column": "SPLIT_PCT",
        "value": 80,
    }
    assert splits.resolve("test").to_dict() == {
        "op": ">=",
        "column": "SPLIT_PCT",
        "value": 80,
    }


def test_splits_support_free_form_named_predicates():
    splits = Splits(
        {
            "fit": Where("SPLIT_PCT") < 60,
            "holdout": Where("application_date") >= "2026-03-01",
        }
    )

    assert splits.resolve("fit").to_dict() == {
        "op": "<",
        "column": "SPLIT_PCT",
        "value": 60,
    }
    assert splits.resolve("holdout").columns() == frozenset({"application_date"})


def test_splits_do_not_police_overlap():
    # Record, don't police (design §12): overlapping splits are legitimate
    # methodology — full-data views, progressive train sets, deliberate reuse.
    splits = Splits(
        {
            "train": Where("SPLIT_PCT") < 80,
            "full": Where("SPLIT_PCT") >= 0,
        }
    )

    assert sorted(splits.predicates) == ["full", "train"]


def test_splits_reject_bucket_ranges():
    with pytest.raises(TypeError, match="bucket ranges were removed"):
        Splits(train=[(0, 80)], test=[(80, 100)])


def test_splits_round_trip_through_dict_payload():
    splits = Splits(
        {
            "fit": (Where("SPLIT_PCT") < 50) | (Where("SPLIT_PCT") >= 70) & (Where("SPLIT_PCT") < 80),
            "holdout": Where("SPLIT_PCT") >= 80,
        }
    )

    payload = splits.to_dict()
    restored = Splits.from_dict(payload)

    assert payload == {
        "predicates": {
            "fit": {
                "op": "or",
                "items": [
                    {"op": "<", "column": "SPLIT_PCT", "value": 50},
                    {
                        "op": "and",
                        "items": [
                            {"op": ">=", "column": "SPLIT_PCT", "value": 70},
                            {"op": "<", "column": "SPLIT_PCT", "value": 80},
                        ],
                    },
                ],
            },
            "holdout": {"op": ">=", "column": "SPLIT_PCT", "value": 80},
        }
    }
    assert restored.predicates == splits.predicates


def test_splits_raise_for_missing_pointer():
    splits = Splits()

    with pytest.raises(KeyError, match="validation"):
        splits.resolve("validation")


def test_run_config_exposes_train_and_eval_split_names():
    run_config = RunConfig(
        experiment_id="example-homecredit",
        splits=Splits(train=Where("SPLIT_PCT") < 80, test=Where("SPLIT_PCT") >= 80),
        models=ModelsConfig(
            manager=ModelRoute("sonnet", "medium"),
            proposer=ModelRoute("sonnet", "medium"),
            coder=ModelRoute("sonnet", "medium"),
        ),
        per_trial_seconds=600,
    )

    assert run_config.train_split == "train"
    assert run_config.eval_split == "test"
    assert run_config.splits.resolve(run_config.train_split) == (Where("SPLIT_PCT") < 80)
    assert run_config.splits.resolve(run_config.eval_split) == (Where("SPLIT_PCT") >= 80)


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

    assert run_config.splits.resolve("train") == (Where("SPLIT_PCT") < 80)
    assert run_config.splits.resolve("test") == (Where("SPLIT_PCT") >= 80)


def test_run_config_rejects_retired_split_keyword():
    with pytest.raises(TypeError, match="split"):
        RunConfig(
            experiment_id="example-homecredit",
            split=Splits(train=Where("SPLIT_PCT") < 80, test=Where("SPLIT_PCT") >= 80),
            models=ModelsConfig(
                manager=ModelRoute("sonnet", "medium"),
                proposer=ModelRoute("sonnet", "medium"),
                coder=ModelRoute("sonnet", "medium"),
            ),
            per_trial_seconds=600,
        )
