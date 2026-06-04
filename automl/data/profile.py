"""Deterministic dataset profile generation."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from automl.data.dataset import LoadedDataset
from automl.errors import DataError
from automl.mlflow import client as mlflow_client
from automl.project import Session
from automl.project import session as active_project_session


@dataclass(frozen=True)
class Profile:
    dataset_id: str
    target_column: str
    data_card_uri: str
    data_observations_uri: str
    profile_manifest_uri: str
    chart_uris: dict[str, str] = field(default_factory=dict)
    created_at: str = ""
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "target_column": self.target_column,
            "data_card_uri": self.data_card_uri,
            "data_observations_uri": self.data_observations_uri,
            "profile_manifest_uri": self.profile_manifest_uri,
            "chart_uris": dict(self.chart_uris),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Profile":
        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            dataset_id=str(payload.get("dataset_id", "")),
            target_column=str(payload.get("target_column", "")),
            data_card_uri=str(payload.get("data_card_uri", "")),
            data_observations_uri=str(payload.get("data_observations_uri", "")),
            profile_manifest_uri=str(payload.get("profile_manifest_uri", "")),
            chart_uris={
                str(key): str(value)
                for key, value in dict(payload.get("chart_uris", {})).items()
            },
            created_at=str(payload.get("created_at", "")),
        )


StatsCheck = Callable[[LoadedDataset], list[dict[str, Any]]]
ChartFn = Callable[[pd.DataFrame, str, Path], None]


def profile(dataset_id: str | None = None, *, session: Session | None = None) -> Profile:
    from automl.data.registry import list_datasets, load_dataset_by_id
    from automl.mlflow.experiment import artifacts as experiment_artifacts

    active = _session(session)
    with mlflow_client.bound_for(active, experiment_id=active.active_experiment_id):
        resolved_dataset_id = dataset_id
        if resolved_dataset_id is None:
            index = list_datasets(session=active)
            dataset = index.active or (index.datasets[-1] if index.datasets else None)
            if dataset is None:
                raise DataError("no dataset is available to profile")
            resolved_dataset_id = dataset.id
        loaded = load_dataset_by_id(resolved_dataset_id, session=active)
        if not isinstance(loaded, LoadedDataset):
            raise DataError("profile requires a full loaded dataset")
        with tempfile.TemporaryDirectory(prefix="automl-profile-") as tmp_dir:
            local_dir = Path(tmp_dir)
            manifest = _write_profile_artifacts(loaded, local_dir)
            uris = experiment_artifacts.write_profile(loaded.id, local_dir=local_dir)
            return Profile(
                dataset_id=loaded.id,
                target_column=manifest["target_column"],
                data_card_uri=uris["data_card_uri"],
                data_observations_uri=uris["data_observations_uri"],
                profile_manifest_uri=uris["profile_manifest_uri"],
                chart_uris=uris["chart_uris"],
                created_at=manifest["created_at"],
            )


def get_profile(dataset_id: str | None = None, *, session: Session | None = None) -> Profile | None:
    from automl.data.registry import list_datasets
    from automl.mlflow.experiment import artifacts as experiment_artifacts

    active = _session(session)
    with mlflow_client.bound_for(active, experiment_id=active.active_experiment_id):
        resolved_dataset_id = dataset_id
        if resolved_dataset_id is None:
            index = list_datasets(session=active)
            dataset = index.active or (index.datasets[-1] if index.datasets else None)
            if dataset is None:
                return None
            resolved_dataset_id = dataset.id
        return experiment_artifacts.read_profile(resolved_dataset_id)


def _write_profile_artifacts(loaded: LoadedDataset, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(exist_ok=True)
    observations: list[dict[str, Any]] = []
    for check in _STATS_CHECKS:
        try:
            observations.extend(check(loaded))
        except Exception as exc:  # noqa: BLE001 - profile should degrade per check
            observations.append(_issue_observation(check.__name__, exc))

    chart_files: dict[str, str] = {}
    for name, chart_fn in _CHARTS:
        out_path = charts_dir / f"{name}.png"
        try:
            chart_fn(loaded.df, loaded.dataset.target_column, out_path)
        except Exception as exc:  # noqa: BLE001 - profile should degrade per chart
            observations.append(_issue_observation(name, exc))
            continue
        if out_path.exists():
            chart_files[name] = str(out_path)

    created_at = datetime.now(UTC).isoformat()
    data_card = _data_card(loaded)
    (output_dir / "data_card.json").write_text(
        json.dumps(data_card, indent=2, default=str),
        encoding="utf-8",
    )
    (output_dir / "data_observations.json").write_text(
        json.dumps(
            {"schema_version": 1, "dataset_id": loaded.id, "observations": observations},
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "dataset_id": loaded.id,
        "target_column": loaded.dataset.target_column,
        "created_at": created_at,
        "chart_files": chart_files,
    }
    (output_dir / "profile_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str),
        encoding="utf-8",
    )
    return manifest


def _session(explicit: Session | None) -> Session:
    return explicit if explicit is not None else active_project_session()


def _data_card(loaded: LoadedDataset) -> dict[str, Any]:
    df = loaded.df
    target = loaded.dataset.target_column
    return {
        "schema_version": 1,
        "dataset_id": loaded.id,
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "target": target,
        "target_distribution": df[target].value_counts(dropna=False).to_dict()
        if target in df.columns
        else {},
        "columns": {
            column: {
                "dtype": str(df[column].dtype),
                "null_pct": float(df[column].isna().mean()) if len(df) else 0.0,
                "nunique": int(df[column].nunique(dropna=True)),
            }
            for column in df.columns
        },
    }


def _basic_observations(loaded: LoadedDataset) -> list[dict[str, Any]]:
    df = loaded.df
    observations: list[dict[str, Any]] = []
    target = loaded.dataset.target_column
    if target in df.columns and len(df):
        counts = df[target].value_counts(dropna=False)
        majority = float((counts / len(df)).max()) if len(counts) else 0.0
        observations.append(
            {
                "kind": "label_distribution",
                "text": f"Majority target share is {majority:.0%}.",
                "source": "profile_deterministic",
            }
        )
    high_missing = df.isna().mean()
    high_missing = high_missing[high_missing > 0.5]
    if len(high_missing):
        observations.append(
            {
                "kind": "missingness",
                "text": f"{len(high_missing)} column(s) have more than 50% missing values.",
                "source": "profile_deterministic",
            }
        )
    return observations


def _unique_key_cardinality(loaded: LoadedDataset) -> list[dict[str, Any]]:
    key = list(loaded.dataset.unique_key)
    if not key or any(column not in loaded.df.columns for column in key):
        return []
    n_distinct = int(len(loaded.df)) - int(loaded.df.duplicated(subset=key).sum())
    return [
        {
            "kind": "unique_key_cardinality",
            "text": (
                f"unique_key {key}: {n_distinct} distinct of {len(loaded.df)} rows "
                f"({'1:1' if n_distinct == len(loaded.df) else 'DUPLICATES PRESENT'})."
            ),
            "source": "profile_deterministic",
        }
    ]


def _plot_label_distribution(df: pd.DataFrame, target: str, out_path: Path) -> None:
    if target not in df.columns:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    df[target].value_counts(dropna=False).sort_index().plot.bar(ax=ax)
    ax.set_title("Label distribution")
    ax.set_xlabel(target)
    ax.set_ylabel("rows")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_missingness(df: pd.DataFrame, target: str, out_path: Path) -> None:
    del target
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    missing = (df.isna().mean() * 100).sort_values(ascending=False).head(20)
    if len(missing) == 0:
        return
    fig, ax = plt.subplots(figsize=(7, max(3, len(missing) * 0.25)))
    missing.iloc[::-1].plot.barh(ax=ax)
    ax.set_xlabel("null %")
    ax.set_title("Missingness")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _issue_observation(name: str, exc: Exception) -> dict[str, str]:
    return {
        "kind": "profile_issue",
        "text": f"{name} failed: {type(exc).__name__}: {exc}",
        "source": "profile_deterministic",
    }


_STATS_CHECKS: list[StatsCheck] = [_basic_observations, _unique_key_cardinality]
_CHARTS: list[tuple[str, ChartFn]] = [
    ("label_distribution", _plot_label_distribution),
    ("missingness", _plot_missingness),
]


__all__ = ["Profile", "get_profile", "profile"]
