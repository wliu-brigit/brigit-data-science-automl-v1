"""Experiment-scoped dataset artifact helpers.

Datasets are produced by an experiment's data pipeline, so their catalog,
profile, and source-trace artifacts live on the experiment overview run and
their heavy bytes under the experiment's GCS route — not on the project.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Mapping

from automl.errors import StorageError
from automl.mlflow import client
from automl.mlflow.experiment import logging as experiment_logging
from automl.mlflow.experiment.lifecycle import ensure_overview
from automl.utils.io import gcs


def write_dataset_record(
    payload: dict,
    *,
    dataset_id: str,
    experiment_id: str | None = None,
) -> str:
    """Log datasets/<id>/dataset.json on the experiment overview run; return its runs:/ URI."""
    segment = _clean_artifact_segment(dataset_id)
    experiment_logging.log_json(f"datasets/{segment}/dataset", payload, experiment_id=experiment_id)
    run_id = _ensure_overview_run_id(experiment_id)
    return f"runs:/{run_id}/datasets/{segment}/dataset.json"


def read_dataset_record(dataset_id: str, experiment_id: str | None = None) -> dict | None:
    """Read one version's dataset.json; None when the record doesn't exist."""
    run_id = _overview_run_id_or_none(experiment_id)
    if run_id is None:
        return None
    segment = _clean_artifact_segment(dataset_id)
    try:
        local_path = client.download_artifact(run_id, f"datasets/{segment}/dataset.json")
    except Exception as exc:
        raise StorageError(f"Failed to read dataset record for {dataset_id!r}") from exc
    if local_path is None:
        return None
    with open(local_path, encoding="utf-8") as handle:
        record = json.load(handle)
    # The record never stores a pointer to itself; the reader derives it so
    # Dataset.record_uri is always populated on anything read back.
    record["record_uri"] = f"runs:/{run_id}/datasets/{segment}/dataset.json"
    return record


def list_dataset_records(experiment_id: str | None = None) -> list[dict]:
    """All version records, sorted by id. The folder structure IS the index."""
    run_id = _overview_run_id_or_none(experiment_id)
    if run_id is None:
        return []
    # Verified against the live prod proxy AND file-backed MLflow
    # (2026-06-04): a missing datasets/ folder and an entirely empty
    # artifact root both return [] from list_artifacts without raising —
    # so an exception here is a genuine transport/auth failure and must
    # propagate loudly, never read as "no datasets yet". (Resolves the
    # step-2 swallow, which assumed missing paths 500 like downloads do.)
    try:
        entries = client.raw().list_artifacts(run_id, "datasets")
    except Exception as exc:
        raise StorageError(
            f"Failed to list dataset records for experiment run {run_id!r}"
        ) from exc
    records: list[dict] = []
    for entry in entries:
        if not entry.is_dir:
            continue
        dataset_id = Path(entry.path).name
        record = read_dataset_record(dataset_id, experiment_id)
        if record is not None:
            records.append(record)
    return sorted(records, key=lambda record: str(record.get("id", "")))


def write_dataset_frame(uri: str, df) -> None:
    try:
        gcs.write_parquet(uri, df, overwrite=True)
    except Exception as exc:
        raise StorageError(f"Failed to write dataset frame {uri!r}") from exc


def read_dataset_frame(uri: str):
    try:
        return gcs.read_parquet(uri)
    except Exception as exc:
        raise StorageError(f"Failed to read dataset frame {uri!r}") from exc


def write_registry(uri: str, registry_frame) -> None:
    try:
        gcs.write_csv(uri, registry_frame, overwrite=True)
    except Exception as exc:
        raise StorageError(f"Failed to write feature registry {uri!r}") from exc


def read_registry(uri: str):
    try:
        return gcs.read_csv(uri)
    except Exception as exc:
        raise StorageError(f"Failed to read feature registry {uri!r}") from exc


def write_profile(
    dataset_id: str,
    *,
    local_dir: str | Path,
    experiment_id: str | None = None,
) -> dict:
    local_path = Path(local_dir)
    artifact_base = f"datasets/{dataset_id}/profile"
    run_id = _ensure_overview_run_id(experiment_id)
    uris = _profile_uris(run_id, artifact_base, local_path)
    manifest_path = local_path / "profile_manifest.json"
    manifest = _read_local_manifest(manifest_path)
    manifest.update(
        {
            "schema_version": int(manifest.get("schema_version", 1)),
            "dataset_id": dataset_id,
            **uris,
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    try:
        client.raw().log_artifacts(run_id, str(local_path), artifact_path=artifact_base)
    except Exception as exc:
        raise StorageError("Failed to log profile artifacts") from exc
    return uris


def read_profile(dataset_id: str, experiment_id: str | None = None):
    from automl.data.profile import Profile

    run_id = _overview_run_id_or_none(experiment_id)
    if run_id is None:
        return None
    artifact_path = f"datasets/{dataset_id}/profile/profile_manifest.json"
    try:
        local_path = client.download_artifact(run_id, artifact_path)
    except Exception:
        return None
    if local_path is None:
        # No profile was published for this dataset — a normal state (e.g. a
        # freshly materialized dataset), not a storage failure.
        return None
    try:
        with open(local_path, encoding="utf-8") as handle:
            return Profile.from_dict(json.load(handle))
    except Exception as exc:
        raise StorageError(f"Failed to read profile manifest for {dataset_id!r}") from exc


def log_source_trace(
    dataset_id: str,
    files: Mapping[str, Path],
    experiment_id: str | None = None,
) -> dict[str, str]:
    """Log source-specific trace files under the Dataset on the experiment overview run."""
    if not files:
        return {}
    artifact_base = f"datasets/{_clean_artifact_segment(dataset_id)}/source_trace"
    run_id = _ensure_overview_run_id(experiment_id)
    uris: dict[str, str] = {}
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_root = Path(tmp_dir)
        for name, source_path in files.items():
            file_name = _clean_artifact_file_name(name)
            staged = tmp_root / file_name
            shutil.copyfile(Path(source_path), staged)
            try:
                client.raw().log_artifact(run_id, str(staged), artifact_path=artifact_base)
            except Exception as exc:
                raise StorageError(f"Failed to log source trace artifact {file_name!r}") from exc
            uris[name] = f"runs:/{run_id}/{artifact_base}/{file_name}"
    return uris


def _ensure_overview_run_id(experiment_id: str | None = None) -> str:
    ensure_overview(experiment_id)
    return experiment_logging._overview_run_id(experiment_id)


def _overview_run_id_or_none(experiment_id: str | None = None) -> str | None:
    try:
        return experiment_logging._overview_run_id(experiment_id)
    except StorageError:
        return None


def _profile_uris(run_id: str, artifact_base: str, local_dir: Path) -> dict:
    charts_dir = local_dir / "charts"
    chart_uris = {
        path.stem: f"runs:/{run_id}/{artifact_base}/charts/{path.name}"
        for path in sorted(charts_dir.glob("*.png"))
    }
    return {
        "data_card_uri": f"runs:/{run_id}/{artifact_base}/data_card.json",
        "data_observations_uri": f"runs:/{run_id}/{artifact_base}/data_observations.json",
        "profile_manifest_uri": f"runs:/{run_id}/{artifact_base}/profile_manifest.json",
        "chart_uris": chart_uris,
    }


def _read_local_manifest(path: Path) -> dict:
    if not path.exists():
        return {"schema_version": 1}
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {"schema_version": 1}


def _clean_artifact_segment(value: str) -> str:
    cleaned = value.strip("/")
    if not cleaned or "/" in cleaned:
        raise ValueError("artifact segment must be a non-empty file-safe name")
    return cleaned


def _clean_artifact_file_name(value: str) -> str:
    cleaned = Path(value.strip("/")).name
    if not cleaned:
        raise ValueError("source trace artifact name required")
    return cleaned


__all__ = [
    "list_dataset_records",
    "log_source_trace",
    "read_dataset_frame",
    "read_dataset_record",
    "read_registry",
    "read_profile",
    "write_dataset_frame",
    "write_dataset_record",
    "write_registry",
    "write_profile",
]
