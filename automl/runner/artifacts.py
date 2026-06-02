"""Compatibility facade for runner-owned MLflow artifact publishing helpers."""

from __future__ import annotations

from automl.runner.data_artifacts import (
    log_data_contract,
    log_feature_artifacts,
)
from automl.runner.failure_artifacts import log_failure_artifacts
from automl.runner.manifest_artifacts import log_manifest
from automl.runner.model_artifacts import log_agent_proposal, log_model
from automl.runner.serving_validation import log_validation_artifacts
from automl.runner.timing import TimingRecorder
from automl.runner.timing_artifacts import log_timing

__all__ = [
    "TimingRecorder",
    "log_agent_proposal",
    "log_data_contract",
    "log_feature_artifacts",
    "log_failure_artifacts",
    "log_manifest",
    "log_model",
    "log_timing",
    "log_validation_artifacts",
]
