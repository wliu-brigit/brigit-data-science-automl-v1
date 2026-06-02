#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "google-cloud-storage>=2.16",
#   "kaggle>=1.6.17",
#   "pandas>=2.2",
#   "pyarrow>=15.0",
# ]
# ///
"""Prepare the full Home Credit training file for this example project.

This is intentionally a one-off script with PEP 723 inline dependencies. It is
not imported by the AutoML loop.
"""
from __future__ import annotations

import argparse
import os
import tempfile
import zipfile
from pathlib import Path
from urllib.parse import urlparse


COMPETITION = "home-credit-default-risk"
TRAIN_CSV = "application_train.csv"
ENV_VAR = "EXAMPLE_HOMECREDIT_GCS_URI"
KAGGLE_API_TOKEN_ENV = "KAGGLE_API_TOKEN"


def _default_work_dir() -> Path:
    return Path(__file__).resolve().parent / ".cache" / "full_homecredit"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _repo_env_path() -> Path:
    return _repo_root() / ".env"


def _load_env_file(env_path: Path) -> None:
    if not env_path.is_file():
        return
    for line in env_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key:
            os.environ.setdefault(key, value)


def _load_repo_env() -> None:
    _load_env_file(_repo_env_path())


def _parse_gcs_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "gs" or not parsed.netloc or not parsed.path.lstrip("/"):
        raise ValueError(f"--gcs-uri must look like gs://bucket/path/file.parquet, got {uri!r}")
    return parsed.netloc, parsed.path.lstrip("/")


def _download_competition(work_dir: Path) -> Path:
    csv_path = work_dir / TRAIN_CSV
    if csv_path.exists():
        return csv_path

    archive = work_dir / f"{TRAIN_CSV}.zip"
    if not archive.exists():
        _load_repo_env()
        api_token = os.environ.get(KAGGLE_API_TOKEN_ENV)
        if not api_token:
            raise SystemExit(
                f"Set {KAGGLE_API_TOKEN_ENV} in {_repo_env_path()} or the process "
                "environment before running this helper."
            )

        try:
            from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore[reportMissingImports]

            with tempfile.TemporaryDirectory(prefix="kaggle_cfg_") as cfg_dir:
                os.environ["KAGGLE_CONFIG_DIR"] = cfg_dir
                if api_token:
                    token_path = Path(cfg_dir) / "access_token"
                    token_path.write_text(api_token + "\n")
                    token_path.chmod(0o600)

                api = KaggleApi()
                api.authenticate()
                api.competition_download_file(
                    COMPETITION,
                    file_name=TRAIN_CSV,
                    path=str(work_dir),
                )
        except Exception as exc:
            raise SystemExit(
                f"Kaggle download failed. Confirm {KAGGLE_API_TOKEN_ENV} in "
                f"{_repo_env_path()} is valid and the {COMPETITION!r} competition "
                "rules have been accepted."
            ) from exc

    with zipfile.ZipFile(archive) as zf:
        if TRAIN_CSV not in zf.namelist():
            raise SystemExit(f"{archive} does not contain {TRAIN_CSV}")
        zf.extract(TRAIN_CSV, path=work_dir)

    if not csv_path.exists():
        raise SystemExit(f"Kaggle download did not produce {csv_path}")
    return csv_path


def _write_parquet(csv_path: Path, parquet_path: Path) -> None:
    import pandas as pd

    df = pd.read_csv(csv_path)
    df.to_parquet(parquet_path, index=False)


def _upload_to_gcs(parquet_path: Path, gcs_uri: str) -> None:
    from google.cloud import storage

    bucket_name, object_name = _parse_gcs_uri(gcs_uri)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(object_name)
    blob.upload_from_filename(str(parquet_path))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Home Credit training data and upload parquet to GCS.",
    )
    parser.add_argument(
        "--gcs-uri",
        required=True,
        help="Destination parquet URI, for example gs://bucket/path/application_train.parquet.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=_default_work_dir(),
        help="Local cache directory for the Kaggle zip, CSV, and generated parquet.",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        help=f"Use an existing {TRAIN_CSV} instead of downloading from Kaggle.",
    )
    args = parser.parse_args()

    _parse_gcs_uri(args.gcs_uri)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    csv_path = args.csv_path if args.csv_path is not None else _download_competition(args.work_dir)
    csv_path = csv_path.expanduser().resolve()
    if not csv_path.is_file():
        raise SystemExit(f"CSV not found: {csv_path}")

    parquet_path = args.work_dir / "application_train.parquet"
    _write_parquet(csv_path, parquet_path)
    _upload_to_gcs(parquet_path, args.gcs_uri)

    print(f"Uploaded {parquet_path} to {args.gcs_uri}")
    print(f"Run with: export {ENV_VAR}={args.gcs_uri}")


if __name__ == "__main__":
    main()
