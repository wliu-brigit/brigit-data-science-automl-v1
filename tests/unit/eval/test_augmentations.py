from pathlib import Path

import pandas as pd
import pytest

from automl.eval import eval_dataset as eval_dataset_module
from automl.project import (
    BinaryClassification,
    ModelRoute,
    ModelsConfig,
    ProjectConfig,
    RunConfig,
    Session,
    Splits,
)

pytestmark = pytest.mark.unit


def _session(tmp_path: Path, *, namespace: str = "", dry_run: bool = False) -> Session:
    route = ModelRoute("sonnet", "medium")
    return Session(
        config=ProjectConfig(
            project_name="demo",
            repo_root=tmp_path,
            project_dir=tmp_path / "projects" / "demo",
            config_path=tmp_path / "projects" / "demo" / "config.py",
            task=BinaryClassification(target="target"),
            run_config=RunConfig(
                experiment_id="baseline",
                splits=Splits({"train": ((0, 50),), "test": ((50, 100),)}),
                models=ModelsConfig(manager=route, proposer=route, coder=route),
                per_trial_seconds=120,
            ),
            gcs_bucket="bucket",
            gcs_prefix="root",
        ),
        namespace=namespace,
        dry_run=dry_run,
    )


def test_augmentation_identity_is_content_based():
    frame = pd.DataFrame({"row_id": [1, 2], "risk_weight": [1.0, 2.0]})
    changed = frame.assign(risk_weight=[1.0, 3.0])

    first = eval_dataset_module.compute_augmentation_identity(
        "ev_123", "risk_weight", frame, ("row_id",)
    )
    second = eval_dataset_module.compute_augmentation_identity(
        "ev_123", "risk_weight", frame, ("row_id",)
    )
    third = eval_dataset_module.compute_augmentation_identity(
        "ev_123", "risk_weight", changed, ("row_id",)
    )

    assert first == second
    assert first != third


def test_augmentation_manifest_round_trips_route_fields(tmp_path):
    active = _session(tmp_path)
    frame = pd.DataFrame({"row_id": [1, 2], "risk_weight": [1.0, 2.0]})

    augmentation = eval_dataset_module.Augmentation.create(
        session=active,
        eval_dataset_id="ev_123",
        name="risk_weight",
        frame=frame,
        unique_key=("row_id",),
    )
    restored = eval_dataset_module.Augmentation.from_dict(
        {**augmentation.to_dict(), "future": "ignored"}
    )

    assert restored == augmentation
    assert augmentation.data_gcs_uri.endswith(
        f"eval/datasets/ev_123/augmentations/risk_weight__{augmentation.hash8}/data.parquet"
    )
    assert augmentation.manifest_gcs_uri.endswith(
        f"eval/datasets/ev_123/augmentations/risk_weight__{augmentation.hash8}/manifest.json"
    )


def test_augmentation_route_uris_preserve_current_gcs_prefix_layout(tmp_path):
    active = _session(tmp_path, namespace="qa", dry_run=True)
    frame = pd.DataFrame({"row_id": [1, 2], "risk_weight": [1.0, 2.0]})

    augmentation = eval_dataset_module.Augmentation.create(
        session=active,
        eval_dataset_id="ev_123",
        name="risk_weight",
        frame=frame,
        unique_key=("row_id",),
    )

    base = (
        f"gs://bucket/root/qa/dry_run/demo/baseline/eval/datasets/"
        f"ev_123/augmentations/risk_weight__{augmentation.hash8}"
    )
    assert augmentation.route_prefix == "root/qa/dry_run/demo/baseline"
    assert augmentation.base_gcs_uri == base
    assert augmentation.data_gcs_uri == f"{base}/data.parquet"
    assert augmentation.manifest_gcs_uri == f"{base}/manifest.json"
    assert eval_dataset_module.augmentation_root_uri("ev_123", session=active) == (
        "gs://bucket/root/qa/dry_run/demo/baseline/eval/datasets/ev_123/augmentations/"
    )


def test_augmentation_validation_rejects_bad_names_unique_keys_and_empty_payload(tmp_path):
    active = _session(tmp_path)
    frame = pd.DataFrame({"row_id": [1, 1], "risk_weight": [1.0, 2.0]})

    with pytest.raises(ValueError, match="name"):
        eval_dataset_module.Augmentation.create(
            session=active,
            eval_dataset_id="ev_123",
            name="RiskWeight",
            frame=frame,
            unique_key=("row_id",),
        )
    with pytest.raises(ValueError, match="duplicate"):
        eval_dataset_module.Augmentation.create(
            session=active,
            eval_dataset_id="ev_123",
            name="risk_weight",
            frame=frame,
            unique_key=("row_id",),
        )
    with pytest.raises(ValueError, match="non-unique-key"):
        eval_dataset_module.Augmentation.create(
            session=active,
            eval_dataset_id="ev_123",
            name="risk_weight",
            frame=pd.DataFrame({"row_id": [1, 2]}),
            unique_key=("row_id",),
        )
    with pytest.raises(ValueError, match="unique_key"):
        eval_dataset_module.Augmentation.create(
            session=active,
            eval_dataset_id="ev_123",
            name="risk_weight",
            frame=pd.DataFrame({"missing": [1, 2], "risk_weight": [1.0, 2.0]}),
            unique_key=("row_id",),
        )
