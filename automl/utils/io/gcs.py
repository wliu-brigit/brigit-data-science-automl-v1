"""Small GCS URI and object helpers."""

from __future__ import annotations

import io
import json
import os
import warnings
from typing import Any

import pandas as pd


GCS_SCHEME = "gs://"


def is_gcs_uri(value: str) -> bool:
    """Return whether ``value`` is a GCS URI."""
    return isinstance(value, str) and value.startswith(GCS_SCHEME)


def parse_gcs_uri(uri: str) -> tuple[str, str]:
    """Parse ``gs://bucket/blob`` into ``(bucket, blob)``."""
    if not is_gcs_uri(uri):
        raise ValueError(f"Expected GCS URI starting with gs://, got {uri!r}")

    remainder = uri.removeprefix(GCS_SCHEME)
    bucket, separator, blob = remainder.partition("/")
    if not bucket:
        raise ValueError(f"GCS URI must include a bucket: {uri!r}")
    if not separator or not blob:
        raise ValueError(f"GCS URI must include an object path: {uri!r}")
    return bucket, blob


def join_uri(base_uri: str, *parts: str) -> str:
    """Join path components onto a GCS URI without losing the bucket."""
    if not is_gcs_uri(base_uri):
        raise ValueError(f"Expected GCS URI starting with gs://, got {base_uri!r}")

    remainder = base_uri.removeprefix(GCS_SCHEME)
    bucket, separator, base_path = remainder.partition("/")
    if not bucket:
        raise ValueError(f"GCS URI must include a bucket: {base_uri!r}")

    segments = []
    if separator and base_path.strip("/"):
        segments.append(base_path.strip("/"))
    segments.extend(part.strip("/") for part in parts if part.strip("/"))
    if not segments:
        return f"{GCS_SCHEME}{bucket}"
    return f"{GCS_SCHEME}{bucket}/{'/'.join(segments)}"


def _gcs_client() -> Any:
    """Create a google-cloud-storage client with the configured project."""
    from google.cloud import storage

    warnings.filterwarnings(
        "ignore",
        message="Your application has authenticated using end user credentials.*",
        category=UserWarning,
    )
    project = os.environ.get("GCP_PROJECT")
    if not project:
        raise EnvironmentError("Missing required environment variable: 'GCP_PROJECT'")
    return storage.Client(project=project)


def _client_or_default(client: Any | None) -> Any:
    return client if client is not None else _gcs_client()


def blob_exists(uri: str, *, client: Any | None = None) -> bool:
    """Return whether the GCS object exists."""
    bucket, blob_name = parse_gcs_uri(uri)
    return bool(_client_or_default(client).bucket(bucket).blob(blob_name).exists())


def list_blob_names(
    uri_or_bucket: str,
    *,
    prefix: str | None = None,
    client: Any | None = None,
) -> list[str]:
    """Return sorted blob names under a bucket/prefix."""
    bucket, blob_prefix = _bucket_and_prefix(uri_or_bucket, prefix)
    blobs = _client_or_default(client).list_blobs(bucket, prefix=blob_prefix)
    return sorted(blob.name for blob in blobs)


def list_prefixes(
    uri_or_bucket: str,
    *,
    prefix: str | None = None,
    client: Any | None = None,
) -> list[str]:
    """Return sorted child prefixes as full ``gs://`` URIs with trailing slashes."""
    bucket, blob_prefix = _bucket_and_prefix(uri_or_bucket, prefix)
    iterator = _client_or_default(client).list_blobs(
        bucket,
        prefix=blob_prefix,
        delimiter="/",
    )
    for _ in iterator:
        pass
    return sorted(f"gs://{bucket}/{item}" for item in iterator.prefixes)


def delete_prefix(uri: str, *, client: Any | None = None) -> int:
    """Delete all blobs under a GCS URI prefix and return the deleted count."""
    bucket, blob_prefix = parse_gcs_uri(uri.rstrip("/") + "/_")
    blob_prefix = blob_prefix.removesuffix("_")
    deleted = 0
    try:
        blobs = _client_or_default(client).list_blobs(bucket, prefix=blob_prefix)
    except Exception as exc:
        raise RuntimeError(f"failed to list GCS prefix {uri!r}: {exc}") from exc
    for blob in blobs:
        try:
            blob.delete()
            deleted += 1
        except Exception as exc:
            name = getattr(blob, "name", "<unknown>")
            raise RuntimeError(f"failed to delete {name!r} under {uri!r}: {exc}") from exc
    return deleted


def move_prefix(source_uri: str, destination_uri: str, *, client: Any | None = None) -> int:
    """Move all blobs under one GCS prefix to another prefix.

    GCS has no atomic directory rename, so this performs copy-then-delete per
    object. The destination prefix must be different from the source prefix.
    """
    source_bucket, source_prefix = parse_gcs_uri(source_uri.rstrip("/") + "/_")
    destination_bucket, destination_prefix = parse_gcs_uri(destination_uri.rstrip("/") + "/_")
    source_prefix = source_prefix.removesuffix("_")
    destination_prefix = destination_prefix.removesuffix("_")
    if source_bucket != destination_bucket:
        raise ValueError("GCS prefix move requires source and destination in the same bucket")
    if source_prefix == destination_prefix:
        raise ValueError("GCS prefix move source and destination must differ")
    moved = 0
    storage_client = _client_or_default(client)
    bucket = storage_client.bucket(source_bucket)
    try:
        blobs = list(storage_client.list_blobs(source_bucket, prefix=source_prefix))
    except Exception as exc:
        raise RuntimeError(f"failed to list GCS prefix {source_uri!r}: {exc}") from exc
    for blob in blobs:
        name = getattr(blob, "name", "<unknown>")
        try:
            suffix = name.removeprefix(source_prefix)
            bucket.copy_blob(blob, bucket, f"{destination_prefix}{suffix}")
            blob.delete()
            moved += 1
        except Exception as exc:
            raise RuntimeError(
                f"failed to move {name!r} from {source_uri!r} to {destination_uri!r}: {exc}"
            ) from exc
    return moved


def read_json(uri: str, *, client: Any | None = None) -> dict[str, Any]:
    """Read a JSON object from GCS."""
    bucket, blob_name = parse_gcs_uri(uri)
    raw = _client_or_default(client).bucket(bucket).blob(blob_name).download_as_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {uri}")
    return payload


def download_to_file(
    uri: str,
    dest: "str | os.PathLike[str]",
    *,
    client: Any | None = None,
) -> None:
    """Robust whole-object download: streamed to disk, checksummed, retried.

    Unlike ``read_bytes``/``read_parquet`` (single-shot into RAM), this uses the
    client's chunked transfer with explicit crc32c validation and an extended
    retry window, for multi-GB objects where a long-lived connection sees
    transients (design: trial-reliability §5).
    """
    from google.cloud.storage.retry import DEFAULT_RETRY

    bucket, blob_name = parse_gcs_uri(uri)
    blob = _client_or_default(client).bucket(bucket).blob(blob_name)
    blob.download_to_filename(
        str(dest),
        checksum="crc32c",
        retry=DEFAULT_RETRY.with_timeout(300.0),
    )


def read_bytes(uri: str, *, client: Any | None = None) -> bytes:
    """Read raw bytes from GCS."""
    bucket, blob_name = parse_gcs_uri(uri)
    return _client_or_default(client).bucket(bucket).blob(blob_name).download_as_bytes()


def write_json(
    uri: str,
    payload: dict[str, Any],
    *,
    client: Any | None = None,
    overwrite: bool = False,
) -> None:
    """Write a JSON object to GCS."""
    bucket, blob_name = parse_gcs_uri(uri)
    _client_or_default(client).bucket(bucket).blob(blob_name).upload_from_string(
        json.dumps(payload, indent=2),
        content_type="application/json",
        if_generation_match=None if overwrite else 0,
    )


def write_bytes(
    uri: str,
    payload: bytes,
    *,
    content_type: str = "application/octet-stream",
    client: Any | None = None,
    overwrite: bool = False,
) -> None:
    """Write raw bytes to GCS."""
    bucket, blob_name = parse_gcs_uri(uri)
    _client_or_default(client).bucket(bucket).blob(blob_name).upload_from_string(
        payload,
        content_type=content_type,
        if_generation_match=None if overwrite else 0,
    )


def read_parquet(uri: str, *, client: Any | None = None) -> pd.DataFrame:
    """Read a parquet DataFrame from GCS."""
    bucket, blob_name = parse_gcs_uri(uri)
    raw = _client_or_default(client).bucket(bucket).blob(blob_name).download_as_bytes()
    return pd.read_parquet(io.BytesIO(raw))


def write_parquet(
    uri: str,
    df: pd.DataFrame,
    *,
    client: Any | None = None,
    overwrite: bool = False,
) -> None:
    """Write a DataFrame as snappy parquet to GCS."""
    bucket, blob_name = parse_gcs_uri(uri)
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False, compression="snappy")
    buffer.seek(0)
    _client_or_default(client).bucket(bucket).blob(blob_name).upload_from_file(
        buffer,
        content_type="application/octet-stream",
        if_generation_match=None if overwrite else 0,
    )


def read_csv(uri: str, *, client: Any | None = None) -> pd.DataFrame:
    """Read a CSV DataFrame from GCS."""
    return pd.read_csv(io.BytesIO(read_bytes(uri, client=client)))


def write_csv(
    uri: str,
    df: pd.DataFrame,
    *,
    client: Any | None = None,
    overwrite: bool = False,
) -> None:
    """Write a DataFrame as CSV bytes to GCS."""
    write_bytes(
        uri,
        df.to_csv(index=False).encode("utf-8"),
        content_type="text/csv",
        client=client,
        overwrite=overwrite,
    )


def _bucket_and_prefix(uri_or_bucket: str, prefix: str | None) -> tuple[str, str]:
    if prefix is not None:
        return uri_or_bucket, prefix.strip("/")
    if is_gcs_uri(uri_or_bucket):
        bucket, blob_prefix = parse_gcs_uri(uri_or_bucket.rstrip("/") + "/_")
        return bucket, blob_prefix.removesuffix("_")
    return uri_or_bucket, ""


__all__ = [
    "blob_exists",
    "delete_prefix",
    "download_to_file",
    "is_gcs_uri",
    "join_uri",
    "list_blob_names",
    "list_prefixes",
    "move_prefix",
    "parse_gcs_uri",
    "read_bytes",
    "read_csv",
    "read_json",
    "read_parquet",
    "write_bytes",
    "write_csv",
    "write_json",
    "write_parquet",
]
