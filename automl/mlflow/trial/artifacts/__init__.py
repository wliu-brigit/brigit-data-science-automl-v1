"""Flat re-exports for trial typed artifact writers."""

from automl.mlflow.trial.artifacts import runner
from automl.mlflow.trial.artifacts.data import (
    TrialDataContractRef,
    load_trial_data_contract,
    write_trial_data_contract,
)
from automl.mlflow.trial.artifacts.eval import (
    EvalIndexRef,
    EvalRef,
    list_eval,
    load_eval,
    load_eval_index,
    validate_eval_label,
    write_eval,
    write_eval_index,
)
from automl.mlflow.trial.artifacts.failure import (
    ERROR_REPORT_ARTIFACT,
    ERROR_TRACEBACK_ARTIFACT,
    load_error_report,
)
from automl.mlflow.trial.artifacts.manifest import ManifestRef, write_manifest
from automl.mlflow.trial.artifacts.model import (
    ModelRef,
    load_model,
    load_model_source,
    write_model,
    write_pickle_model,
)
from automl.mlflow.trial.artifacts.predictions import (
    PredictionsRef,
    list_predictions,
    load_predictions,
    write_predictions,
)

__all__ = [
    "EvalIndexRef",
    "EvalRef",
    "ERROR_REPORT_ARTIFACT",
    "ERROR_TRACEBACK_ARTIFACT",
    "ManifestRef",
    "ModelRef",
    "PredictionsRef",
    "TrialDataContractRef",
    "runner",
    "list_eval",
    "list_predictions",
    "load_eval",
    "load_eval_index",
    "load_error_report",
    "load_model",
    "load_model_source",
    "load_predictions",
    "load_trial_data_contract",
    "validate_eval_label",
    "write_eval",
    "write_eval_index",
    "write_manifest",
    "write_model",
    "write_pickle_model",
    "write_predictions",
    "write_trial_data_contract",
]
