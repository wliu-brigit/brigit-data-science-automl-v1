from __future__ import annotations

import pytest

from automl.trial.manifest import TrialRunManifest

pytestmark = pytest.mark.unit


def test_trial_run_manifest_round_trips_current_runner_payload_shape():
    payload = {
        "schema_version": 1,
        "run": {
            "run_id": "run-123",
            "project_name": "home_credit",
            "experiment_id": "baseline",
            "experiment_overview_run_id": "overview-456",
            "trial_id": "trial-789",
            "trial_number": 4,
            "trial_slug": "baseline_lr",
            "trial_strategy": "baseline",
        },
        "data": {
            "dataset_id": "dataset-v1",
            "identity_hash": "sha256:abc",
            "record_uri": "runs:/overview-run/datasets/dataset-v1/dataset.json",
            "contract_artifact": "data/contract.json",
        },
        "model": {"pyfunc_uri": "runs:/run-123/model"},
        "evaluation": {
            "primary_label": "holdout",
            "report_artifact": "eval/holdout/report.json",
            "manifest_artifact": "eval/manifest.json",
            "predictions_uri": "gs://bucket/predictions.parquet",
        },
        "validation": {
            "status": "success",
            "report_artifact": "validation/report.json",
        },
        "timing": {
            "schema_version": 1,
            "unit": "seconds",
            "total_seconds": 12.5,
            "phases": {"fit": 4.0, "evaluation": 3.5},
        },
        "artifacts": [
            {
                "path": "data/contract.json",
                "content_type": "application/json",
                "schema_version": 1,
            },
            {
                "path": "eval/manifest.json",
                "content_type": "application/json",
                "schema_version": 1,
            },
            {
                "path": "features/dataset_feature_registry.csv",
                "content_type": "text/csv",
                "schema_version": 1,
            },
            {
                "path": "features/feature_registry.csv",
                "content_type": "text/csv",
                "schema_version": 1,
            },
            {
                "path": "model/MLmodel",
                "content_type": "text/yaml",
                "schema_version": 1,
            },
            {
                "path": "timing/summary.json",
                "content_type": "application/json",
                "schema_version": 1,
            },
            {
                "path": "validation/data/input.csv",
                "content_type": "text/csv",
                "schema_version": 1,
            },
            {
                "path": "validation/data/input.parquet",
                "content_type": "application/parquet",
                "schema_version": 1,
            },
            {
                "path": "validation/data/expected.parquet",
                "content_type": "application/parquet",
                "schema_version": 1,
            },
            {
                "path": "validation/latency_detail.json",
                "content_type": "application/json",
                "schema_version": 1,
            },
            {
                "path": "validation/report.json",
                "content_type": "application/json",
                "schema_version": 1,
            },
        ],
    }

    assert TrialRunManifest.from_dict(payload).to_dict() == payload
