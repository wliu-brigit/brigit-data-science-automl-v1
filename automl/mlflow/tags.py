"""Canonical MLflow keys grouped by AutoML noun domain."""

CREATED_BY = "created_by"
RUN_KIND = "run.kind"
CREATED_AT = "run.created_at"

ACTIVE_DATASET_ID = "data.active_dataset_id"

EXPERIMENT_ID = "experiment.id"
EXPERIMENT_NAME = "experiment.name"
EXPERIMENT_OVERVIEW_RUN_ID = "experiment.overview_run_id"
PROJECT_NAME = "project.name"
PROJECT_OVERVIEW_RUN_ID = "project.overview_run_id"

TRIAL_SLUG = "trial.slug"
TRIAL_STRATEGY = "trial.strategy"
TRIAL_STATUS = "trial.status"
TRIAL_ID = "trial.id"
TRIAL_NUMBER = "trial.number"
TRIAL_HYPOTHESIS = "trial.hypothesis"
TRIAL_TRAINING_ORIGIN = "trial.origin"
TRIAL_ISSUE_COUNT = "trial.issue_count"

DATA_CONTRACT_URI = "data.contract_artifact"
MODEL_URI = "model.uri"
MODEL_SOURCE_URI = "model.source_artifact"
# MLflow 3 stores models as standalone "logged model" entities under
# ``mlflow-artifacts:/<exp>/models/<model_id>/`` rather than in the run's
# artifact tree. Persisting the id makes the model location deterministically
# discoverable from the run alone: ``get_run(run_id)`` -> tag -> ``models:/<id>``.
# Resolving via ``models:/<id>`` also avoids the legacy ``runs:/<run>/model``
# probe, which 500s (and is then retried for ~254s) when the model is not in the
# run artifact tree.
MODEL_LOGGED_ID = "model.logged_model_id"
MANIFEST_URI = "trial.manifest_artifact"
EVAL_INDEX_URI = "eval.manifest_artifact"
EVAL_PRIMARY_LABEL = "eval.primary_label"

# Serving-validation outcome. ``success`` means the round-tripped model loaded,
# predicted, and matched within tolerance. The single source of truth for
# "is this trial's model usable" — derive any deployability gate from it
# (``== "success"``) rather than a separate flag. No ``deployment.*`` noun until
# a serving/deployment module actually exists.
VALIDATION_STATUS = "validation.status"


def eval_uri(label: str) -> str:
    return f"eval.{label}.report_artifact"


def eval_dataset_id(label: str) -> str:
    return f"eval.{label}.dataset_id"


def eval_predictions_uri(label: str) -> str:
    return f"eval.{label}.predictions_uri"


def eval_predictions_manifest_uri(label: str) -> str:
    return f"eval.{label}.predictions_artifact"


__all__ = [
    "ACTIVE_DATASET_ID",
    "CREATED_BY",
    "CREATED_AT",
    "DATA_CONTRACT_URI",
    "EVAL_INDEX_URI",
    "EVAL_PRIMARY_LABEL",
    "EXPERIMENT_ID",
    "EXPERIMENT_NAME",
    "EXPERIMENT_OVERVIEW_RUN_ID",
    "MANIFEST_URI",
    "MODEL_URI",
    "MODEL_LOGGED_ID",
    "MODEL_SOURCE_URI",
    "VALIDATION_STATUS",
    "PROJECT_NAME",
    "PROJECT_OVERVIEW_RUN_ID",
    "RUN_KIND",
    "TRIAL_ID",
    "TRIAL_HYPOTHESIS",
    "TRIAL_ISSUE_COUNT",
    "TRIAL_NUMBER",
    "TRIAL_SLUG",
    "TRIAL_STATUS",
    "TRIAL_STRATEGY",
    "TRIAL_TRAINING_ORIGIN",
    "eval_dataset_id",
    "eval_predictions_manifest_uri",
    "eval_predictions_uri",
    "eval_uri",
]
