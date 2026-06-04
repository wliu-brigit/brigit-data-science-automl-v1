"""Experiment-level MLflow seam API."""

from automl.mlflow.experiment import eval_datasets
from automl.mlflow.experiment.artifacts import (
    list_dataset_records,
    log_source_trace,
    read_dataset_frame,
    read_dataset_record,
    read_profile,
    read_registry,
    write_dataset_frame,
    write_dataset_record,
    write_profile,
    write_registry,
)
from automl.mlflow.experiment.lifecycle import (
    ensure,
    ensure_overview,
    get_active_dataset,
    mlflow_experiment_id,
    read_overview,
    set_active_dataset,
    write_overview,
)
from automl.mlflow.experiment.logging import log_json
from automl.mlflow.experiment.queries import (
    find_trial_run_id,
    list_trials,
    next_trial_number,
    search_trials,
    top_n_by_metric,
)

__all__ = [
    "ensure",
    "ensure_overview",
    "eval_datasets",
    "find_trial_run_id",
    "get_active_dataset",
    "list_dataset_records",
    "list_trials",
    "log_json",
    "log_source_trace",
    "mlflow_experiment_id",
    "next_trial_number",
    "read_dataset_frame",
    "read_dataset_record",
    "read_overview",
    "read_profile",
    "read_registry",
    "search_trials",
    "set_active_dataset",
    "top_n_by_metric",
    "write_dataset_frame",
    "write_dataset_record",
    "write_overview",
    "write_profile",
    "write_registry",
]
