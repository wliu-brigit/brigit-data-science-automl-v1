"""Project-scoped artifact helpers."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Mapping

from automl.errors import StorageError
from automl.mlflow import _routing
from automl.mlflow import client
from automl.mlflow.artifact_paths import json_artifact_path
from automl.mlflow.project import overview as project_overview
from automl.utils.io import gcs


def dataset_index_uri() -> str:
    return _project_uri("data/dataset_index.json")


def read_dataset_index() -> dict:
    uri = dataset_index_uri()
    try:
        if not gcs.blob_exists(uri):
            return {"schema_version": 1, "datasets": []}
    except Exception as exc:
        raise StorageError(f"Failed to read dataset index {uri!r}") from exc
    try:
        return gcs.read_json(uri)
    except Exception as exc:
        raise StorageError(f"Failed to read dataset index {uri!r}") from exc


def write_dataset_index(payload: dict) -> None:
    uri = dataset_index_uri()
    try:
        gcs.write_json(uri, payload, overwrite=True)
    except Exception as exc:
        raise StorageError(f"Failed to write dataset index {uri!r}") from exc


def log_dataset_catalog(payload: dict, *, active_dataset_id: str) -> None:
    """Mirror the small dataset catalog pointers onto the project overview run."""
    document = dict(payload)
    document["schema_version"] = int(document.get("schema_version", 1))
    document["active_dataset_id"] = active_dataset_id
    log_json("datasets/index", document)
    latest = _latest_dataset_payload(document, active_dataset_id)
    log_json("datasets/latest", latest)


def read_dataset_manifest(uri: str) -> dict:
    try:
        return gcs.read_json(uri)
    except Exception as exc:
        raise StorageError(f"Failed to read dataset manifest {uri!r}") from exc


def write_dataset_manifest(uri: str, payload: dict) -> None:
    try:
        gcs.write_json(uri, payload, overwrite=True)
    except Exception as exc:
        raise StorageError(f"Failed to write dataset manifest {uri!r}") from exc


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


def write_profile(dataset_id: str, *, local_dir: str | Path) -> dict:
    local_path = Path(local_dir)
    artifact_base = f"datasets/{dataset_id}/profile"
    run_id = project_overview._ensure_overview_run_id()
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


def read_profile(dataset_id: str):
    from automl.data.profile import Profile

    run = project_overview._overview_run()
    if run is None:
        return None
    artifact_path = f"datasets/{dataset_id}/profile/profile_manifest.json"
    try:
        local_path = client.raw().download_artifacts(run.info.run_id, artifact_path)
    except Exception:
        return None
    try:
        with open(local_path, encoding="utf-8") as handle:
            return Profile.from_dict(json.load(handle))
    except Exception as exc:
        raise StorageError(f"Failed to read profile manifest for {dataset_id!r}") from exc


def log_json(name: str, payload: dict) -> None:
    """Log a loose-tier JSON artifact on the project overview run."""
    artifact_path = json_artifact_path(name)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / Path(artifact_path).name
        tmp_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        _log_artifact(
            tmp_path,
            artifact_path=str(Path(artifact_path).parent)
            if Path(artifact_path).parent.as_posix() != "."
            else None,
        )


def log_source_trace(dataset_id: str, files: Mapping[str, Path]) -> dict[str, str]:
    """Log source-specific trace files under the Dataset on the project overview run."""
    if not files:
        return {}
    artifact_base = f"datasets/{_clean_artifact_segment(dataset_id)}/source_trace"
    run_id = project_overview._ensure_overview_run_id()
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


def _project_uri(path: str) -> str:
    bound = client.bound()
    if not bound.bucket:
        raise StorageError("GCS bucket required for project artifacts")
    return f"gs://{bound.bucket}/{_routing.project_route_prefix()}/{path.strip('/')}"


def _log_artifact(local_path: Path, *, artifact_path: str | None) -> None:
    run_id = project_overview._ensure_overview_run_id()
    try:
        client.raw().log_artifact(run_id, str(local_path), artifact_path=artifact_path)
    except Exception as exc:
        raise StorageError(f"Failed to log project artifact {local_path.name!r}") from exc


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


def _latest_dataset_payload(index_payload: dict, active_dataset_id: str) -> dict:
    datasets = index_payload.get("datasets")
    if not isinstance(datasets, list):
        datasets = []
    active = None
    for item in datasets:
        if isinstance(item, dict) and item.get("id") == active_dataset_id:
            active = item
            break
    return {
        "schema_version": int(index_payload.get("schema_version", 1)),
        "dataset_id": active_dataset_id,
        "dataset": active or {},
    }


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
    "dataset_index_uri",
    "log_dataset_catalog",
    "log_json",
    "log_source_trace",
    "read_dataset_frame",
    "read_dataset_index",
    "read_dataset_manifest",
    "read_registry",
    "read_profile",
    "write_dataset_frame",
    "write_dataset_index",
    "write_dataset_manifest",
    "write_registry",
    "write_profile",
]
