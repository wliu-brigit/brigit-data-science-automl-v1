"""Runner trial manifest artifact publishing helpers."""

from __future__ import annotations

from automl.data import TrialDataContract
from automl.mlflow import tags as mlflow_tags
from automl.mlflow import trial as mlflow_trial
from automl.mlflow.trial import artifacts as trial_artifacts
from automl.project import Session
from automl.trial.manifest import TrialRunManifest


def log_manifest(
    *,
    run_id: str,
    active: Session,
    trial_id: str,
    trial_number: int | None,
    slug: str,
    strategy: str,
    contract: TrialDataContract,
    eval_result,
    validation_report: dict[str, object],
    timing: dict[str, object],
    has_agent_proposal: bool = False,
) -> None:
    run_tags = mlflow_trial.get_tags(run_id)
    payload = {
        "schema_version": 1,
        "run": {
            "run_id": run_id,
            "project_name": active.project_name,
            "experiment_id": active.active_experiment_id,
            "experiment_overview_run_id": run_tags.get(
                mlflow_tags.EXPERIMENT_OVERVIEW_RUN_ID,
                "",
            ),
            "trial_id": trial_id,
            "trial_number": int(trial_number or 0),
            "trial_slug": slug,
            "trial_strategy": strategy,
        },
        "data": {
            "dataset_id": contract.dataset.id,
            "identity_hash": contract.dataset.identity_hash,
            "record_uri": contract.dataset.record_uri,
            "contract_artifact": "data/contract.json",
        },
        "model": {"pyfunc_uri": f"runs:/{run_id}/model"},
        "evaluation": {
            "primary_label": eval_result.label,
            "report_artifact": f"eval/{eval_result.label}/report.json",
            "manifest_artifact": "eval/manifest.json",
            "predictions_uri": eval_result.predictions_uri,
        },
        "validation": {
            "status": validation_report.get("status"),
            "report_artifact": "validation/report.json",
        },
        "timing": timing,
        "artifacts": sorted(
            _artifact_manifest_entries(has_agent_proposal=has_agent_proposal),
            key=lambda item: item["path"],
        ),
    }
    manifest = TrialRunManifest.from_dict(payload).to_dict()
    trial_artifacts.write_manifest(run_id, manifest)


def _artifact_manifest_entries(*, has_agent_proposal: bool) -> list[dict[str, object]]:
    paths = [
        ("data/contract.json", "application/json"),
        ("eval/manifest.json", "application/json"),
        ("features/dataset_feature_registry.csv", "text/csv"),
        ("features/feature_registry.csv", "text/csv"),
        ("model/MLmodel", "text/yaml"),
        ("timing/summary.json", "application/json"),
        ("trial/issues.json", "application/json"),
        ("validation/data/input.csv", "text/csv"),
        ("validation/data/input.parquet", "application/parquet"),
        ("validation/data/expected.parquet", "application/parquet"),
        ("validation/latency_detail.json", "application/json"),
        ("validation/report.json", "application/json"),
    ]
    if has_agent_proposal:
        paths.append(("agent/proposer/proposal.json", "application/json"))
    return [
        {
            "path": path,
            "content_type": content_type,
            "schema_version": 1,
        }
        for path, content_type in paths
    ]


__all__ = ["log_manifest"]
