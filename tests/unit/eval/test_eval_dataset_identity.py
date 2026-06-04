from pathlib import Path

import pandas as pd
import pytest

from automl.eval import EvalDataset
from automl.eval import eval_dataset as eval_dataset_module
from automl.eval import registry as registry_module
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


def _session(tmp_path: Path, *, namespace: str = "qa", dry_run: bool = False) -> Session:
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


def test_split_view_identity_is_recipe_based_not_frame_based():
    first = eval_dataset_module.compute_eval_dataset_identity(
        kind="split_view",
        of_dataset_id="dataset-v1",
        split_pct_col="SPLIT_PCT",
        buckets=((80, 90), (95, 100)),
        target_column="target",
        hash_key=("row_id",),
    )
    second = eval_dataset_module.compute_eval_dataset_identity(
        kind="split_view",
        of_dataset_id="dataset-v1",
        split_pct_col="SPLIT_PCT",
        buckets=((80, 90), (95, 100)),
        target_column="target",
        hash_key=("row_id",),
    )
    same_rows_different_recipe = eval_dataset_module.compute_eval_dataset_identity(
        kind="split_view",
        of_dataset_id="dataset-v1",
        split_pct_col="SPLIT_PCT",
        buckets=((80, 100),),
        target_column="target",
        hash_key=("row_id",),
    )

    assert first == second
    assert first != same_rows_different_recipe


def test_split_view_identity_sorts_and_rejects_overlapping_buckets():
    ordered = eval_dataset_module.compute_eval_dataset_identity(
        kind="split_view",
        of_dataset_id="dataset-v1",
        split_pct_col="SPLIT_PCT",
        buckets=((10, 20), (30, 40)),
        target_column="target",
        hash_key=("row_id",),
    )
    reversed_order = eval_dataset_module.compute_eval_dataset_identity(
        kind="split_view",
        of_dataset_id="dataset-v1",
        split_pct_col="SPLIT_PCT",
        buckets=((30, 40), (10, 20)),
        target_column="target",
        hash_key=("row_id",),
    )

    assert reversed_order == ordered
    with pytest.raises(ValueError, match="overlap"):
        eval_dataset_module.compute_eval_dataset_identity(
            kind="split_view",
            of_dataset_id="dataset-v1",
            split_pct_col="SPLIT_PCT",
            buckets=((10, 20), (15, 30)),
            target_column="target",
            hash_key=("row_id",),
        )


def test_external_identity_changes_with_frame_content():
    frame = pd.DataFrame({"row_id": [1, 2], "target": [0, 1], "score": [0.1, 0.9]})
    changed = frame.assign(score=[0.2, 0.9])

    first = eval_dataset_module.compute_eval_dataset_identity(
        kind="external",
        frame=frame,
        target_column="target",
        hash_key=("row_id",),
    )
    second = eval_dataset_module.compute_eval_dataset_identity(
        kind="external",
        frame=changed,
        target_column="target",
        hash_key=("row_id",),
    )

    assert first != second


def test_eval_dataset_manifest_round_trips_route_and_kind(tmp_path):
    active = _session(tmp_path)
    dataset = EvalDataset.split_view(
        session=active,
        of_dataset_id="dataset-v1",
        split="test",
        split_pct_col="SPLIT_PCT",
        buckets=((50, 100),),
        target_column="target",
        hash_key=("row_id",),
    )

    payload = dataset.to_dict()
    restored = EvalDataset.from_dict({**payload, "future": "ignored"})

    assert restored == dataset
    assert payload["data_gcs_uri"] is None
    assert "schema_hash" not in payload
    assert "content_hash" not in payload
    assert restored.manifest_gcs_uri == (
        f"gs://bucket/root/qa/demo/baseline/eval/datasets/{dataset.id}/manifest.json"
    )


def test_external_manifest_carries_hashes_and_data_uri(tmp_path):
    active = _session(tmp_path)
    frame = pd.DataFrame({"row_id": [1, 2], "target": [0, 1], "score": [0.1, 0.9]})
    dataset = EvalDataset.external(
        session=active,
        frame=frame,
        target_column="target",
        hash_key=("row_id",),
        provenance={"source": "unit"},
    )

    payload = dataset.to_dict()

    assert payload["kind"] == "external"
    assert payload["schema_hash"].startswith("sha256:")
    assert payload["content_hash"].startswith("sha256:")
    assert payload["data_gcs_uri"].endswith(f"eval/datasets/{dataset.id}/data.parquet")


def test_eval_dataset_route_uris_preserve_current_gcs_prefix_layout(tmp_path):
    active = _session(tmp_path, namespace="qa", dry_run=True)
    frame = pd.DataFrame({"row_id": [1, 2], "target": [0, 1], "score": [0.1, 0.9]})

    dataset = EvalDataset.external(
        session=active,
        frame=frame,
        target_column="target",
        hash_key=("row_id",),
    )

    assert dataset.route_prefix == "root/qa/dry_run/demo/baseline"
    assert dataset.manifest_gcs_uri == (
        f"gs://bucket/root/qa/dry_run/demo/baseline/eval/datasets/{dataset.id}/manifest.json"
    )
    assert dataset.data_gcs_uri == (
        f"gs://bucket/root/qa/dry_run/demo/baseline/eval/datasets/{dataset.id}/data.parquet"
    )
    assert eval_dataset_module.manifest_uri_for(dataset.id, session=active) == (
        f"gs://bucket/root/qa/dry_run/demo/baseline/eval/datasets/{dataset.id}/manifest.json"
    )


@pytest.mark.parametrize(
    ("namespace", "dry_run", "expected"),
    [
        ("", False, "gs://bucket/root/demo/baseline/eval/datasets/"),
        ("", True, "gs://bucket/root/dry_run/demo/baseline/eval/datasets/"),
        ("qa", False, "gs://bucket/root/qa/demo/baseline/eval/datasets/"),
        ("qa", True, "gs://bucket/root/qa/dry_run/demo/baseline/eval/datasets/"),
    ],
)
def test_eval_registry_dataset_root_uri_pins_route_matrix(tmp_path, namespace, dry_run, expected):
    active = _session(tmp_path, namespace=namespace, dry_run=dry_run)

    assert registry_module._eval_dataset_root(active) == expected


def test_external_identity_validates_target_hash_key_and_duplicates(tmp_path):
    active = _session(tmp_path)
    frame = pd.DataFrame({"row_id": [1, 1], "target": [0, 1]})

    with pytest.raises(ValueError, match="target"):
        EvalDataset.external(
            session=active,
            frame=frame.drop(columns=["target"]),
            target_column="target",
            hash_key=("row_id",),
        )
    with pytest.raises(ValueError, match="hash_key"):
        EvalDataset.external(
            session=active,
            frame=frame,
            target_column="target",
            hash_key=("missing",),
        )
    with pytest.raises(ValueError, match="duplicate"):
        EvalDataset.external(
            session=active,
            frame=frame,
            target_column="target",
            hash_key=("row_id",),
        )
