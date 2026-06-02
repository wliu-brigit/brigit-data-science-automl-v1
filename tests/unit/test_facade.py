import pytest

import automl

pytestmark = pytest.mark.unit


def test_experiment_is_single_top_level_name():
    assert hasattr(automl, "Experiment")
    assert "ExperimentOverview" not in automl.__all__


def test_experiment_overview_still_reachable_in_domain():
    from automl.experiment import ExperimentOverview  # noqa: F401
