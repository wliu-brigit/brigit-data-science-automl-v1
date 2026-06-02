"""MLflow artifact path normalization helpers."""

from __future__ import annotations


def json_artifact_path(name: str) -> str:
    cleaned = name.strip("/")
    if not cleaned:
        raise ValueError("JSON artifact name required")
    if not cleaned.endswith(".json"):
        cleaned = f"{cleaned}.json"
    return cleaned


__all__ = ["json_artifact_path"]
