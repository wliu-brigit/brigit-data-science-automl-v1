"""Experiment-level MLflow seam API."""

from automl.mlflow.experiment import eval_datasets
from automl.mlflow.experiment.artifacts import (
    dataset_index_uri,
    log_dataset_catalog,
    log_source_trace,
    read_dataset_frame,
    read_dataset_index,
    read_dataset_manifest,
    read_profile,
    read_registry,
    write_dataset_frame,
    write_dataset_index,
    write_dataset_manifest,
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
    "dataset_index_uri",
    "ensure",
    "ensure_overview",
    "eval_datasets",
    "find_trial_run_id",
    "get_active_dataset",
    "list_trials",
    "log_dataset_catalog",
    "log_json",
    "log_source_trace",
    "mlflow_experiment_id",
    "next_trial_number",
    "read_dataset_frame",
    "read_dataset_index",
    "read_dataset_manifest",
    "read_overview",
    "read_profile",
    "read_registry",
    "search_trials",
    "set_active_dataset",
    "top_n_by_metric",
    "write_dataset_frame",
    "write_dataset_index",
    "write_dataset_manifest",
    "write_overview",
    "write_profile",
    "write_registry",
]
