"""Runner data and feature artifact publishing helpers."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

from automl.data import TrialDataContract
from automl.mlflow import trial as mlflow_trial
from automl.mlflow.trial import artifacts as trial_artifacts
from automl.mlflow.trial.artifacts import runner as runner_artifacts


def log_data_contract(run_id: str, contract: TrialDataContract) -> None:
    trial_artifacts.write_trial_data_contract(run_id, contract)
    tag_payload: dict[str, object] = {
        "data.dataset_id": contract.dataset.id,
        "data.identity_hash": contract.dataset.identity_hash,
        "data.manifest_uri": contract.dataset.manifest_uri,
    }
    for slice_contract in contract.slices:
        if slice_contract.name is None:
            continue
        tag_payload[f"data.slice.{slice_contract.name}.content_hash"] = (
            slice_contract.content_hash
        )
        tag_payload[f"data.slice.{slice_contract.name}.n_rows"] = slice_contract.n_rows
    mlflow_trial.set_tags(run_id, tag_payload)


def log_feature_artifacts(
    *,
    run_id: str,
    dataset_registry,
    model_registry,
    model,
) -> None:
    _log_registry_csv(run_id, "features/dataset_feature_registry.csv", dataset_registry)
    _log_registry_csv(run_id, "features/feature_registry.csv", model_registry)
    try:
        importances = model.feature_importances()
    except Exception:
        importances = None
    if not importances:
        return
    rows = []
    for name, value in dict(importances).items():
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            rows.append({"feature": str(name), "importance": numeric})
    if rows:
        import pandas as pd

        frame = pd.DataFrame(rows).sort_values("importance", ascending=False)
        _log_frame_csv(run_id, "features/importance.csv", frame)


def _log_registry_csv(run_id: str, path: str, registry) -> None:
    if registry is None or not hasattr(registry, "to_dataframe"):
        return
    _log_frame_csv(run_id, path, registry.to_dataframe())


def _log_frame_csv(run_id: str, path: str, frame) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        local_path = Path(tmp_dir) / Path(path).name
        frame.to_csv(local_path, index=False)
        runner_artifacts.write_local_file(run_id, path, local_path)


__all__ = ["log_data_contract", "log_feature_artifacts"]
