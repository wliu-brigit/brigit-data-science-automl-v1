"""Trial manifest artifact writer."""

from __future__ import annotations

from dataclasses import dataclass

from automl.errors import StorageError
from automl.mlflow import tags
from automl.mlflow.trial.artifacts.data import _payload_to_dict, _write_payload
from automl.mlflow.trial.logging import set_tag


@dataclass(frozen=True)
class ManifestRef:
    run_id: str
    uri: str
    path: str = "manifest.json"


def write_manifest(run_id: str, payload: object) -> ManifestRef:
    path = "manifest.json"
    uri = _write_payload(run_id, path, _payload_to_dict(payload))
    try:
        set_tag(run_id, tags.MANIFEST_URI, path)
    except Exception as exc:
        raise StorageError("Failed to commit manifest artifact") from exc
    return ManifestRef(run_id=run_id, uri=uri)


__all__ = ["ManifestRef", "write_manifest"]
