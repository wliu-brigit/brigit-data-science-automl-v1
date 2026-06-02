import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import ConvergenceWarning

from projects.example_homecredit.model import MODEL_CLASS, HomeCreditLogisticModel

pytestmark = pytest.mark.integration

SAMPLE_PATH = (
    Path(__file__).resolve().parents[3]
    / "projects"
    / "example_homecredit"
    / "data"
    / "application_train_sample.csv"
)


def test_homecredit_logistic_model_fits_and_scores_tiny_frame():
    df = pd.DataFrame(
        {
            "TARGET": [0, 1, 0, 1],
            "EXT_SOURCE_1": [0.1, 0.8, np.nan, 0.6],
            "AMT_CREDIT": [100_000.0, np.nan, 150_000.0, 80_000.0],
            "NAME_CONTRACT_TYPE": ["Cash loans", "Revolving loans", "Cash loans", "Cash loans"],
        }
    )

    model = HomeCreditLogisticModel()
    fitted = model.fit(df)

    predictions = model.predict(df)
    probabilities = model.predict_proba(df)

    assert fitted is model
    assert MODEL_CLASS is HomeCreditLogisticModel
    assert predictions.shape == (len(df),)
    assert probabilities.shape == (len(df),)
    assert np.all((0.0 <= probabilities) & (probabilities <= 1.0))


def test_homecredit_logistic_model_converges_on_committed_sample():
    df = pd.read_csv(SAMPLE_PATH)

    model = HomeCreditLogisticModel()
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(df)

    predictions = model.predict(df)
    probabilities = model.predict_proba(df)

    assert predictions.shape == (len(df),)
    assert probabilities.shape == (len(df),)
    assert np.all((0.0 <= probabilities) & (probabilities <= 1.0))
