# Data Snapshot Run Contract Cutover Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Store each materialized data snapshot once, then derive train/test views per run from deterministic `SPLITID` and the run's recorded `Split`.

**Architecture:** Separate the immutable data snapshot from the run-specific data contract. A data snapshot owns `data.parquet`, `feature_registry.csv`, source lineage, target, schema, and content hashes. Each trial logs one `data/contract.json` artifact that points to the snapshot manifest and owns the exact train/test split view used by that run. No old train/test snapshot manifest compatibility is required.

**Tech Stack:** Python, pandas, MLflow, GCS helpers, current `DataPipeline`, `RunConfig.Split`, existing snapshot/manifest and runner artifact systems.

---

## Current Problem

The code currently validates deterministic `SPLITID`, but snapshot identity and storage still depend on the train/test split:

- `automl/data/snapshot.py` computes `snapshot_identity_hash` from `df_train`, `df_test`, registry, target, and split.
- `snapshot_gcs_paths()` writes `train.parquet` and `test.parquet`.
- `DataPipeline.prepare_data()` materializes, splits, and writes train/test snapshot files.
- `DataPipeline.load_data_snapshot()` reads train/test files directly.
- `automl/runner/_execute.py` logs MLflow train/test inputs from separate train/test URIs.
- `automl/data/run_snapshot.py` loads historical runs from `train_uri` and `test_uri`.

That means changing 80/20 to 70/30 creates a different snapshot even when the materialized data is identical. The refactor below makes deterministic split the actual source of train/test reproducibility.

## Target Invariants

- A data snapshot is split-independent.
- A split view is run-specific and historical.
- Dry-run and full-run remain separate namespaces.
- Changing `RUN_CONFIG.split` reuses the same `data.parquet` snapshot when materialized data and registry are unchanged.
- The snapshot is content-addressed. Source identity, canonical data bytes, registry bytes, schema, target column, and split-id column define snapshot identity.
- Pipeline class or quality-threshold changes create a new snapshot only when they change canonical data, registry, schema, target, split-id column, or source identity. Do not hash pipeline code or config classes into the snapshot identity.
- Historical run load-back uses the split view recorded on that run, not current `project.py`.
- No fallback reader for old manifests.

## File Structure

- Modify `automl/data/snapshot.py`
  - Own data-snapshot identity, GCS path schema, manifest validation, and split-view helper functions.
- Modify `automl/data/pipeline.py`
  - Materialize one canonical data frame, write/read one snapshot file, and apply split views at load time.
- Modify `automl/data/sources.py`
  - Provide stable source identity payloads for snapshot identity and richer source event payloads for traceability.
- Modify `automl/mlflow/store.py`
  - Store active data snapshot metadata without split tags.
- Modify `automl/runner/_execute.py`
  - Log one run data contract plus compact searchable tags.
- Modify `automl/runner/_stages.py`
  - Update snapshot hash and split-view helper expectations.
- Modify `automl/mlflow/artifacts/data.py`
  - Add `write_data_contract()` and remove old per-run data mirror writers.
- Modify `automl/data/run_snapshot.py`
  - Load historical runs from `data/contract.json` plus the referenced snapshot manifest.
- Modify `automl/profile/snapshot.py`, `automl/validate/targets.py`, `automl/inspect/views.py` only if signatures or artifact names change.
- Modify `automl/data/sources.py`, `automl/data/spec.py`, `automl/data/loader.py`, notebooks as part of the same strict cutover so snapshot lineage uses the final public names.
- Update tests in `tests/integration/test_data_pipeline_snapshots.py`, `tests/integration/test_runner.py`, `tests/integration/test_logging_contract_smoke.py`, `tests/unit/test_inspect.py`, and source-preview tests.

## Execution Order Note

Do the source API hard cutover before implementing source identity in `DataPipeline`. Snapshot manifests should never be written with retired names like `csv_path`, `gcs_uri`, `training_data_sql`, `base_data_sql`, `rows`, or `dry_run_rows`.

## Source Identity Rule

The snapshot manifest should not accidentally carry stale source metadata. Use two source payloads:

- `source_identity`: stable source declaration that participates in `snapshot_identity_hash`.
- `source_event`: execution traceability for the prepare event.

For example:

```python
source_identity = {
    "kind": "local_csv",
    "data_path": "projects/example_homecredit/data/application_train_sample.csv",
    "hash_key": ["SK_ID_CURR"],
    "dry_run_nrows": 100,
}
source_event = {
    **source_identity,
    "refresh_source": False,
}
```

Do not include mutable execution flags such as `refresh_source` in `source_identity`; otherwise a forced refresh that produces identical data would create an unnecessary new snapshot. Do include `dry_run_nrows` when dry-run materialization is capped, because it changes the canonical data.

For Snowflake, `source_identity` should include executed SQL hashes, while `source_event` should include both the query path when present and artifact URIs for the executed SQL text. Direct ad hoc `query=` has no local path, so the executed SQL artifact is the durable source of truth.

## Run Data Contract Logging

Do not use MLflow `log_input()` for this cutover. It records dataset lineage metadata, but it does not provide the native load-back contract we need for train/test views derived from one canonical data snapshot. If we log MLflow inputs and still load from our own artifacts, we create two ways to describe the same data.

Log data lineage the same way the rest of the system logs model/eval/proposal state: a versioned MLflow artifact plus compact searchable tags.

- `data/contract.json` is the run-level data contract.
- The trial manifest `data` block points to `data/contract.json`.
- MLflow tags mirror only the fields needed for search and UI filtering.
- The loader reads `data/contract.json`, not tags and not MLflow input records.

## Unified Lineage Contract

There are exactly two authoritative lineage artifacts:

1. **GCS data snapshot manifest:** `gs://.../data/snapshots/<snapshot_name>/manifest.json`
   - Owns canonical data identity.
   - Points to `data.parquet` and `feature_registry.csv`.
   - Contains data/registry/schema/source hashes, target column, split-id column, source identity, and source event.

2. **Run data contract:** MLflow run artifact `data/contract.json`
   - Owns train/test identity for that run.
   - Points to the GCS data snapshot manifest URI.
   - Contains `snapshot_name`, `snapshot_identity_hash`, `split_id_col`, split ranges, derived train/test hashes, train/test row counts, and a compact UI summary.

Everything else is a derived index or summary:

- MLflow tags mirror selected fields for search.
- The trial run manifest may include a compact data summary, but it must not become a second loader contract.

Do not write separate per-run `data/snapshot.json`, `data/inputs.json`, `data/source.json`, `data/schema.json`, or MLflow dataset inputs in this cutover. Those duplicate the snapshot manifest, run data contract, or run manifest summary.

## Write/Read Symmetry Rule

All data reconstruction must go through the same two identifiers:

- `data_manifest_uri`: where to read canonical data and feature registry.
- `data/contract.json`: how this run used the snapshot, including the exact train/test split view.

Active reads get `data_manifest_uri` from the experiment overview's active snapshot pointer, then apply current `RUN_CONFIG.split`. Historical reads get `data_manifest_uri` and the recorded split view from the run's `data/contract.json`. No loader should infer data identity from MLflow tags, run manifest summaries, notebook state, or current `project.py` when loading a historical run.

MLflow tags are still written, but only as search/UI mirrors:

- `data.snapshot_name`
- `data.snapshot_identity_hash`
- `data.manifest_uri`
- `data.contract_artifact`
- `data.split_view_hash`

Validation compares tags against `manifest.json` and `data/contract.json`; loading never uses tags as the source of truth.

### Data Snapshot Write Path

`DataPipeline.prepare_data()` writes the active data snapshot:

- GCS: `data.parquet`
- GCS: `feature_registry.csv`
- GCS: `manifest.json`
- MLflow experiment overview: `latest_snapshot.json` pointer to the GCS manifest URI.
- MLflow experiment overview: `snapshot_index.json` entries with snapshot name, identity hash, and GCS manifest URI.

The GCS manifest is the data snapshot source of truth. MLflow overview artifacts are pointers/indexes.

### Active Data Read Path

`DataPipeline.load_data_snapshot()` reads the active snapshot for the current project run:

- Reads the active pointer from the experiment overview.
- Reads the GCS data manifest v2 from the pointer.
- Reads GCS `data.parquet`.
- Reads GCS `feature_registry.csv`.
- Validates manifest hashes with `validate_data_manifest_v2()`.
- Applies current `RUN_CONFIG.split` through `apply_split_view()`.
- Builds an in-memory split view for the current run.

This path may use current `project.py` split because it is preparing the current run.

### Trial Run Write Path

`automl/runner/_execute.py` writes run-specific lineage:

- MLflow run artifact `data/contract.json` containing the exact data contract used by that run.
- Trial run manifest data summary that points to `data/contract.json` and the GCS data manifest URI.
- MLflow tags for search and UI.

The run's `data/contract.json` is the historical train/test source of truth.

### Historical Run Read Path

`automl.data.run_snapshot.load_data_snapshot_for_run()` reads a completed run:

- Reads run artifact `data/contract.json`.
- Reads the GCS data manifest URI from `data/contract.json`.
- Reads GCS `data.parquet`.
- Reads GCS `feature_registry.csv`.
- Validates data snapshot manifest with `validate_data_manifest_v2()`.
- Applies the recorded split from `data/contract.json` through `apply_split_view()`.
- Validates the split view with `validate_split_view()`.
- Validates MLflow run tags against the same data contract.

This path must not use current `project.py` split. Historical load-back is contract-artifact driven.

## Run Data Contract Schema

`data/contract.json` is a machine-only load contract. It is not a report, not a UI view, and not an artifact inventory. It contains only the run lineage, the immutable snapshot pointer, and the exact split view needed for deterministic load-back:

```json
{
  "schema_version": 1,
  "run": {
    "project_name": "example_homecredit",
    "experiment_id": "2026-05-example",
    "trial_id": "example_notebook_2_elasticnet",
    "run_id": "<mlflow_run_id>"
  },
  "snapshot": {
    "name": "v1_abc12345",
    "manifest_uri": "gs://bucket/.../data/snapshots/v1_abc12345/manifest.json",
    "identity_hash": "sha256:...",
    "prepare_event_id": "pe_20260517T000000Z_v1_abc12345",
    "target_column": "TARGET",
    "shape": {
      "n_rows": 1000,
      "n_columns": 128
    }
  },
  "split": {
    "split_id_col": "SPLITID",
    "ranges": {
      "train": [[0, 80]],
      "test": [[80, 100]]
    },
    "view_hash": "sha256:...",
    "train_content_hash": "sha256:...",
    "test_content_hash": "sha256:...",
    "shape": {
      "train": {"n_rows": 800, "n_columns": 128},
      "test": {"n_rows": 200, "n_columns": 128}
    }
  }
}
```

The GCS snapshot manifest remains the physical snapshot source of truth: it owns `data_uri`, `feature_registry_uri`, source identity, schema hash, and content hashes. The run contract points to that manifest and owns only the run-specific split. Feature registry artifacts remain under `features/`, and the root run manifest remains the broad artifact index.

---

## Task 1: Snapshot Helper Tests For New Storage Contract

**Files:**
- Modify: `tests/integration/test_data_pipeline_snapshots.py`
- Modify: `automl/data/snapshot.py`

- [ ] **Step 1: Write failing tests for data-only paths**

Add tests asserting `snapshot_gcs_paths()` returns only canonical data paths:

```python
def test_snapshot_gcs_paths_use_single_data_file() -> None:
    paths = snapshot_gcs_paths(
        bucket="bucket",
        gcs_prefix="automl",
        project_name="payment_routing",
        experiment_id="2026-Q2",
        snapshot_name="v1_abc12345",
        dry_run=True,
        route_namespace="",
    )

    assert paths["data_path"].endswith("/data/snapshots/v1_abc12345/data.parquet")
    assert paths["data_uri"].endswith("/data/snapshots/v1_abc12345/data.parquet")
    assert paths["feature_registry_path"].endswith("/feature_registry.csv")
    assert paths["manifest_path"].endswith("/manifest.json")
    assert "train_path" not in paths
    assert "test_path" not in paths
    assert "train_uri" not in paths
    assert "test_uri" not in paths
```

- [ ] **Step 2: Write failing tests that snapshot identity ignores split**

Use one canonical `df_data` and registry dataframe. Compute snapshot identity once. Then compute two split views from the same data and assert snapshot identity does not change while split view identity does.

```python
def test_snapshot_identity_is_split_independent() -> None:
    df = pd.DataFrame({
        "ID": [1, 2, 3, 4, 5],
        "TARGET": [0, 1, 0, 1, 0],
        "SPLITID": [10, 20, 70, 85, 95],
    })
    registry_df = pd.DataFrame({
        "name": ["ID", "TARGET", "SPLITID"],
        "target": [False, True, False],
        "available": [True, True, True],
    })

    identity = compute_snapshot_identity(
        df,
        registry_df,
        target_column="TARGET",
        split_id_col="SPLITID",
        source_identity={"kind": "test", "dry_run_nrows": None},
    )
    split_80 = build_split_view(
        df,
        snapshot_identity_hash=identity.snapshot_identity_hash,
        split_id_col="SPLITID",
        split=Split(train=[(0, 80)], test=[(80, 100)]),
    )
    split_70 = build_split_view(
        df,
        snapshot_identity_hash=identity.snapshot_identity_hash,
        split_id_col="SPLITID",
        split=Split(train=[(0, 70)], test=[(70, 100)]),
    )

    assert split_80["split_view_hash"] != split_70["split_view_hash"]
    assert split_80["hashes"]["train_content_hash"] != split_70["hashes"]["train_content_hash"]
    assert identity.snapshot_identity_hash == compute_snapshot_identity(
        df,
        registry_df,
        target_column="TARGET",
        split_id_col="SPLITID",
        source_identity={"kind": "test", "dry_run_nrows": None},
    ).snapshot_identity_hash
```

- [ ] **Step 3: Run the targeted tests and confirm they fail**

Run:

```bash
uv run pytest tests/integration/test_data_pipeline_snapshots.py::test_snapshot_gcs_paths_use_single_data_file tests/integration/test_data_pipeline_snapshots.py::test_snapshot_identity_is_split_independent -q
```

Expected: failures because helpers still expose train/test paths and identity still requires `df_train`, `df_test`, and `split`.

- [ ] **Step 4: Implement new snapshot helper signatures**

Change `automl/data/snapshot.py`:

```python
@dataclass(frozen=True)
class SnapshotIdentity:
    data_content_hash: str
    feature_registry_hash: str
    schema_hash: str
    source_identity_hash: str
    snapshot_identity_hash: str
    snapshot_hash8: str
    target_column: str
    split_id_col: str


def schema_hash(df_data: pd.DataFrame) -> str:
    payload = {
        "columns": list(df_data.columns),
        "dtypes": [str(dtype) for dtype in df_data.dtypes],
    }
    return _json_hash(payload)


def compute_snapshot_identity(
    df_data: pd.DataFrame,
    registry_df: pd.DataFrame,
    *,
    target_column: str,
    split_id_col: str,
    source_identity: dict[str, Any],
) -> SnapshotIdentity:
    data_hash = dataframe_content_hash(df_data)
    registry_hash = registry_content_hash(registry_df)
    schema = schema_hash(df_data)
    source_identity_hash = _json_hash(source_identity)
    identity_payload = {
        "data_content_hash": data_hash,
        "feature_registry_hash": registry_hash,
        "schema_hash": schema,
        "source_identity_hash": source_identity_hash,
        "target_column": target_column,
        "split_id_col": split_id_col,
    }
    identity_hash = _json_hash(identity_payload)
    return SnapshotIdentity(
        data_content_hash=data_hash,
        feature_registry_hash=registry_hash,
        schema_hash=schema,
        source_identity_hash=source_identity_hash,
        snapshot_identity_hash=identity_hash,
        snapshot_hash8=identity_hash.removeprefix("sha256:")[:8],
        target_column=target_column,
        split_id_col=split_id_col,
    )
```

Add:

```python
def split_to_manifest(split: Split) -> dict[str, Any]:
    return {
        "kind": "split_id",
        "train": [list(pair) for pair in split.train],
        "test": [list(pair) for pair in split.test],
    }


def split_from_manifest(payload: dict[str, Any]) -> Split:
    if payload.get("kind") != "split_id":
        raise ValueError(f"unsupported split kind: {payload.get('kind')!r}")
    return Split(train=payload["train"], test=payload["test"])


def apply_split_view(
    df_data: pd.DataFrame,
    *,
    split_id_col: str,
    split: Split,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if split_id_col not in df_data.columns:
        raise KeyError(f"split_id_col {split_id_col!r} not in data columns")
    df_train = df_data[df_data[split_id_col].isin(split.train_buckets())].reset_index(drop=True)
    df_test = df_data[df_data[split_id_col].isin(split.test_buckets())].reset_index(drop=True)
    return df_train, df_test


def build_split_view(
    df_data: pd.DataFrame,
    *,
    snapshot_identity_hash: str,
    split_id_col: str,
    split: Split,
) -> dict[str, Any]:
    df_train, df_test = apply_split_view(df_data, split_id_col=split_id_col, split=split)
    split_payload = split_to_manifest(split)
    train_hash = dataframe_content_hash(df_train)
    test_hash = dataframe_content_hash(df_test)
    payload = {
        "schema_version": 1,
        "snapshot_identity_hash": snapshot_identity_hash,
        "split_id_col": split_id_col,
        "split": split_payload,
        "shape": {
            "n_train": int(len(df_train)),
            "n_test": int(len(df_test)),
        },
        "hashes": {
            "train_content_hash": train_hash,
            "test_content_hash": test_hash,
        },
    }
    payload["split_view_hash"] = _json_hash(payload)
    return payload
```

Add split-view validation for load-back:

```python
def validate_split_view(
    manifest: dict[str, Any],
    split_view: dict[str, Any],
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
) -> None:
    manifest_hashes = manifest.get("hashes")
    if not isinstance(manifest_hashes, dict):
        raise RuntimeError("data manifest is missing hashes")
    snapshot_identity_hash = str(manifest_hashes.get("snapshot_identity_hash") or "")
    if split_view.get("snapshot_identity_hash") != snapshot_identity_hash:
        raise RuntimeError("split view snapshot_identity_hash does not match data snapshot")
    if split_view.get("split_id_col") != manifest.get("split_id_col"):
        raise RuntimeError("split view split_id_col does not match data snapshot")
    hashes = split_view.get("hashes")
    if not isinstance(hashes, dict):
        raise RuntimeError("split view is missing hashes")
    actual_train = dataframe_content_hash(df_train)
    actual_test = dataframe_content_hash(df_test)
    if hashes.get("train_content_hash") != actual_train:
        raise RuntimeError("split view train_content_hash does not match derived train data")
    if hashes.get("test_content_hash") != actual_test:
        raise RuntimeError("split view test_content_hash does not match derived test data")
    shape = split_view.get("shape")
    if not isinstance(shape, dict):
        raise RuntimeError("split view is missing shape")
    if shape.get("n_train") != len(df_train):
        raise RuntimeError("split view n_train does not match derived train data")
    if shape.get("n_test") != len(df_test):
        raise RuntimeError("split view n_test does not match derived test data")
    expected_hash = _json_hash({
        key: value
        for key, value in split_view.items()
        if key != "split_view_hash"
    })
    if split_view.get("split_view_hash") != expected_hash:
        raise RuntimeError("split_view_hash does not match split view contents")
```

Update `snapshot_gcs_paths()` to return:

```python
{
    "bucket": bucket,
    "base_path": base,
    "data_path": f"{base}/data.parquet",
    "feature_registry_path": f"{base}/feature_registry.csv",
    "manifest_path": f"{base}/manifest.json",
    "data_uri": f"gs://{bucket}/{base}/data.parquet",
    "feature_registry_uri": f"gs://{bucket}/{base}/feature_registry.csv",
    "manifest_uri": f"gs://{bucket}/{base}/manifest.json",
}
```

- [ ] **Step 5: Run targeted tests**

Run:

```bash
uv run pytest tests/integration/test_data_pipeline_snapshots.py::test_snapshot_gcs_paths_use_single_data_file tests/integration/test_data_pipeline_snapshots.py::test_snapshot_identity_is_split_independent -q
```

Expected: PASS.

---

## Task 2: Manifest Schema V2 For Canonical Data Snapshot

**Files:**
- Modify: `automl/data/snapshot.py`
- Modify: `tests/unit/test_data_snapshot.py`
- Modify: `tests/integration/test_data_pipeline_snapshots.py`

- [ ] **Step 1: Write failing manifest tests**

Add tests that `build_data_manifest()` rejects old train/test paths and emits schema v2:

```python
def test_build_data_manifest_v2_uses_data_uri_not_train_test() -> None:
    df = pd.DataFrame({"ID": [1, 2], "TARGET": [0, 1], "SPLITID": [10, 90]})
    registry_df = pd.DataFrame({"name": ["ID", "TARGET", "SPLITID"]})
    identity = compute_snapshot_identity(
        df,
        registry_df,
        target_column="TARGET",
        split_id_col="SPLITID",
        source_identity={"kind": "local_csv", "data_path": "sample.csv", "dry_run_nrows": None},
    )
    paths = snapshot_gcs_paths(
        bucket="bucket",
        gcs_prefix="automl",
        project_name="p",
        experiment_id="e",
        snapshot_name=f"v1_{identity.snapshot_hash8}",
        dry_run=False,
    )

    manifest = build_data_manifest(
        project_name="p",
        experiment_id="e",
        snapshot_name=f"v1_{identity.snapshot_hash8}",
        paths=paths,
        identity=identity,
        df_data=df,
        registry_df=registry_df,
        target_column="TARGET",
        split_id_col="SPLITID",
        source_identity={"kind": "local_csv", "data_path": "sample.csv", "dry_run_nrows": None},
        source_event={"kind": "local_csv"},
        run_mode="full_run",
        dry_run_nrows=None,
        created_at="2026-05-17T00:00:00+00:00",
    )

    assert manifest["schema_version"] == 2
    assert manifest["gcs"]["data_uri"] == paths["data_uri"]
    assert "train_uri" not in manifest["gcs"]
    assert "test_uri" not in manifest["gcs"]
    assert manifest["hashes"]["data_content_hash"] == identity.data_content_hash
    assert manifest["hashes"]["source_identity_hash"] == identity.source_identity_hash
    assert manifest["split_id_col"] == "SPLITID"
```

- [ ] **Step 2: Update `build_data_manifest()`**

Change the signature to:

```python
def build_data_manifest(
    *,
    project_name: str,
    experiment_id: str,
    snapshot_name: str,
    paths: dict[str, str],
    identity: SnapshotIdentity,
    df_data: pd.DataFrame,
    registry_df: pd.DataFrame,
    target_column: str,
    split_id_col: str,
    source_identity: dict[str, Any],
    source_event: dict[str, Any],
    run_mode: str,
    dry_run_nrows: int | None,
    created_at: str | None = None,
) -> dict[str, Any]:
```

Return only data-level fields:

```python
return {
    "schema_version": 2,
    "project_name": project_name,
    "experiment_id": experiment_id,
    "snapshot_name": snapshot_name,
    "created_at": timestamp,
    "run_mode": run_mode,
    "dry_run_nrows": dry_run_nrows,
    "gcs": {
        "data_uri": paths["data_uri"],
        "feature_registry_uri": paths["feature_registry_uri"],
        "manifest_uri": paths["manifest_uri"],
    },
    "shape": {
        "n_rows": int(len(df_data)),
        "n_columns": int(len(df_data.columns)),
    },
    "hashes": {
        "data_content_hash": identity.data_content_hash,
        "feature_registry_hash": identity.feature_registry_hash,
        "schema_hash": identity.schema_hash,
        "source_identity_hash": identity.source_identity_hash,
        "snapshot_identity_hash": identity.snapshot_identity_hash,
    },
    "target_column": target_column,
    "split_id_col": split_id_col,
    "source_identity": _thaw_jsonable(source_identity),
    "source_event": _thaw_jsonable(source_event),
}
```

Add strict v2 validation:

```python
def validate_data_manifest_v2(
    manifest: dict[str, Any],
    df_data: pd.DataFrame,
    registry_df: pd.DataFrame,
) -> None:
    if manifest.get("schema_version") != 2:
        raise RuntimeError("data snapshot manifest schema_version must be 2")
    gcs = manifest.get("gcs")
    if not isinstance(gcs, dict) or not gcs.get("data_uri"):
        raise RuntimeError("data snapshot manifest must include gcs.data_uri")
    if "train_uri" in gcs or "test_uri" in gcs:
        raise RuntimeError("data snapshot manifest must not include train_uri/test_uri")
    source_identity = manifest.get("source_identity")
    if not isinstance(source_identity, dict):
        raise RuntimeError("data snapshot manifest must include source_identity")
    source_event = manifest.get("source_event")
    if not isinstance(source_event, dict):
        raise RuntimeError("data snapshot manifest must include source_event")
    identity = compute_snapshot_identity(
        df_data,
        registry_df,
        target_column=str(manifest.get("target_column") or ""),
        split_id_col=str(manifest.get("split_id_col") or ""),
        source_identity=source_identity,
    )
    hashes = manifest.get("hashes")
    if not isinstance(hashes, dict):
        raise RuntimeError("data snapshot manifest must include hashes")
    expected = {
        "data_content_hash": identity.data_content_hash,
        "feature_registry_hash": identity.feature_registry_hash,
        "schema_hash": identity.schema_hash,
        "source_identity_hash": identity.source_identity_hash,
        "snapshot_identity_hash": identity.snapshot_identity_hash,
    }
    for key, value in expected.items():
        if hashes.get(key) != value:
            raise RuntimeError(f"data snapshot manifest {key} does not match loaded data")
```

Replace the train/test-based validators in `DataPipeline` and `automl/data/run_snapshot.py` with this helper.

- [ ] **Step 3: Run manifest tests**

Run:

```bash
uv run pytest tests/unit/test_data_snapshot.py tests/integration/test_data_pipeline_snapshots.py::test_build_data_manifest_v2_uses_data_uri_not_train_test -q
```

Expected: PASS.

---

## Task 3: Pipeline Materialization Without Split-Coupled Snapshot Storage

**Files:**
- Modify: `automl/data/pipeline.py`
- Modify: `automl/data/sources.py`
- Modify: `tests/integration/test_data_pipeline_snapshots.py`
- Modify: `tests/integration/test_data_pipeline.py`

- [ ] **Step 1: Add failing tests for split reuse**

Add a test that prepares a snapshot with 80/20, then loads with 70/30 without creating a new snapshot name.

Use monkeypatched GCS writers/readers as the existing snapshot tests do. The assertion is:

```python
assert first_manifest["snapshot_name"] == second_manifest["snapshot_name"]
assert first_loaded.split_view["split_view_hash"] != second_loaded.split_view["split_view_hash"]
assert len(first_loaded.df_train) != len(second_loaded.df_train)
assert first_loaded.registry.to_dataframe().equals(second_loaded.registry.to_dataframe())
```

Add a write/read roundtrip test for active snapshots:

```python
def test_prepare_then_load_active_snapshot_roundtrips_v2_manifest(monkeypatch, tmp_path):
    pipeline = _pipeline_with_split(tmp_path, split=Split(train=[(0, 80)], test=[(80, 100)]))

    prepared = pipeline.prepare_data()
    loaded = pipeline.load_data_snapshot()

    assert prepared["gcs"]["data_uri"] == loaded.manifest["gcs"]["data_uri"]
    assert "train_uri" not in loaded.manifest["gcs"]
    assert "test_uri" not in loaded.manifest["gcs"]
    assert loaded.split_view["split"] == split_to_manifest(pipeline.split_spec)
    validate_data_manifest_v2(loaded.manifest, loaded.df_data, loaded.registry.to_dataframe())
    validate_split_view(loaded.manifest, loaded.split_view, loaded.df_train, loaded.df_test)
```

- [ ] **Step 2: Reject custom split overrides in snapshot-backed pipelines**

The split view is the single source of truth. Snapshot-backed pipelines must derive train/test only through `apply_split_view(df_data, split_id_col=..., split=...)`.

Add:

```python
def _assert_split_view_supported(self) -> None:
    if type(self).split is not DataPipeline.split:
        raise NotImplementedError(
            "Snapshot-backed pipelines require deterministic Split over SPLITID; "
            "custom DataPipeline.split() is not supported in this cutover."
        )
    if self.split_id_col is None:
        raise NotImplementedError(
            "Snapshot-backed pipelines require split_id_col; set a hash_key on the source "
            "or emit SPLITID from SQL."
        )
```

Call `_assert_split_view_supported()` before `preview()`, `prepare_data()`, and `load_data_snapshot()` derive train/test.

- [ ] **Step 3: Add stable source identity payloads**

Add source identity support in `automl/data/sources.py`:

```python
class DataSource:
    ...

    def identity_payload(self, pipeline: "DataPipeline") -> dict[str, object]:
        payload = dict(self.event_payload(pipeline))
        payload.pop("refresh_source", None)
        return payload
```

Override where richer stable identity is useful. For Snowflake, include hashes of the executed SQL text, because a local query path alone is not enough lineage:

```python
def _sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def identity_payload(self, pipeline: "DataPipeline") -> dict[str, object]:
    payload = dict(self.event_payload(pipeline))
    payload.pop("refresh_source", None)
    if self.query is not None or self.query_path is not None:
        payload["query_executed_sql_hash"] = _sha256_text(
            self.query_executed_sql(pipeline)
        )
    if self.refresh_query_path is not None:
        payload["refresh_query_executed_sql_hash"] = _sha256_text(
            self.refresh_query_executed_sql(pipeline)
        )
    return payload
```

- [ ] **Step 4: Split materialization into pre-split and split application**

In `DataPipeline`, replace `_materialize_source()` with a data-level helper:

```python
def _materialize_dataset(
    self,
    *,
    refresh_source: bool = True,
    apply_learnings: bool = True,
) -> tuple[pd.DataFrame, FeatureRegistry]:
    if refresh_source and self.force_refresh_source:
        self.run_base_sql()
    df = self.load_training_data()
    df = self.normalize_source_values(df)
    df, name_map = self.standardize_columns(df)
    self.validate_loaded_data(df)
    registry = self.build_feature_registry(df, name_map)
    if apply_learnings:
        learning_payloads = self._learning_payloads_from_mlflow()
        registry.apply_learning_flags(
            golden_entries=[
                *self._normalize_learning_entries(learning_payloads["golden"]),
                *self._normalize_learning_entries(learning_payloads["project_golden"]),
            ],
            weak_entries=[
                *self._normalize_learning_entries(learning_payloads["weak"]),
                *self._normalize_learning_entries(learning_payloads["project_weak"]),
            ],
        )
    df = self.apply_column_roles(df, registry)
    self.infer_dtypes(df, registry)
    df = self.apply_dtypes(df, registry)
    df = self.dedupe(df)
    df = self.apply_quality_filters(df, registry)
    self.flag_features(registry)
    self._validate_contract(registry)
    return df, registry
```

Add:

```python
def _split_dataset(
    self,
    df_data: pd.DataFrame,
    registry: FeatureRegistry,
    *,
    snapshot_identity_hash: str,
    validate_evaluation: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    self._assert_split_view_supported()
    df_train, df_test = apply_split_view(
        df_data,
        split_id_col=self.split_id_col or "",
        split=self.split_spec,
    )
    if validate_evaluation:
        self._validate_evaluation_columns(df_test, registry)
    split_view = build_split_view(
        df_data,
        snapshot_identity_hash=snapshot_identity_hash,
        split_id_col=self.split_id_col or "",
        split=self.split_spec,
    )
    return df_train, df_test, split_view
```

Always pass a real snapshot identity hash. For no-log `preview()`, compute the preview identity locally using the same `compute_snapshot_identity()` helper; do not use an empty or placeholder hash.

- [ ] **Step 5: Update `preview()`**

`preview()` should materialize once, split locally, and return the same `DataPreview` shape:

```python
df_data, registry = self._materialize_dataset(
    refresh_source=False,
    apply_learnings=False,
)
registry_df = registry.to_dataframe()
identity = compute_snapshot_identity(
    df_data,
    registry_df,
    target_column=self.target_column or "",
    split_id_col=self.split_id_col or "",
    source_identity=self.source.identity_payload(self),
)
df_train, df_test, split_view = self._split_dataset(
    df_data,
    registry,
    snapshot_identity_hash=identity.snapshot_identity_hash,
    validate_evaluation=False,
)
report = build_split_report(
    df_data,
    target=self.target_column,
    split=self.split_spec,
    split_id_col=self.split_id_col,
)
```

- [ ] **Step 6: Update `prepare_data()`**

`prepare_data()` must write only:

- `data.parquet`
- `feature_registry.csv`
- `manifest.json`

Use:

```python
df_data, registry = self._materialize_dataset()
registry_df = registry.to_dataframe()
source_identity = self.source.identity_payload(self)
source_event = self.source_event_payload()
identity = compute_snapshot_identity(
    df_data,
    registry_df,
    target_column=self.target_column or "",
    split_id_col=self.split_id_col or "",
    source_identity=source_identity,
)
```

Build the manifest with both stable source identity and prepare-event source metadata:

```python
manifest = build_data_manifest(
    project_name=self.project_name,
    experiment_id=self.ctx.experiment_id,
    snapshot_name=snapshot_name,
    paths=paths,
    identity=identity,
    df_data=df_data,
    registry_df=registry_df,
    target_column=self.target_column or "",
    split_id_col=self.split_id_col or "",
    source_identity=source_identity,
    source_event=source_event,
    run_mode="dry_run" if self.dry_run else "full_run",
    dry_run_nrows=self.dry_run_row_limit(),
    created_at=created_at,
)
```

Write:

```python
write_df_to_gcs_as_parquet(df_data, paths["bucket"], paths["data_path"])
write_df_to_gcs_as_csv(registry_df, paths["bucket"], paths["feature_registry_path"])
write_json_to_gcs(manifest, paths["bucket"], paths["manifest_path"])
```

Do not write `train.parquet` or `test.parquet`.

- [ ] **Step 7: Update active snapshot load**

`load_data_snapshot()` should read `data.parquet`, build split view from current `self.split_spec`, and return train/test:

```python
df_data, registry_df = self._read_snapshot_objects(bucket, required)
self._validate_snapshot_hashes(manifest, df_data, registry_df)
registry = FeatureRegistry.from_dataframe(registry_df)
self._assert_split_view_supported()
if manifest["split_id_col"] != self.split_id_col:
    raise RuntimeError("active data snapshot split_id_col does not match pipeline split_id_col")
df_train, df_test = apply_split_view(
    df_data,
    split_id_col=manifest["split_id_col"],
    split=self.split_spec,
)
split_view = build_split_view(
    df_data,
    snapshot_identity_hash=manifest["hashes"]["snapshot_identity_hash"],
    split_id_col=manifest["split_id_col"],
    split=self.split_spec,
)
self._validate_evaluation_columns(df_test, registry)
return LoadedDataSnapshot(
    df_train=df_train,
    df_test=df_test,
    df_data=df_data,
    registry=registry,
    manifest=manifest,
    snapshot_name=manifest["snapshot_name"],
    prepare_event_id=active.get("prepare_event_id", ""),
    split_view=split_view,
)
```

Add `df_data`, `split_view`, and `data_contract` fields to `LoadedDataSnapshot`. `data_contract` defaults to `{}` for active snapshots and is populated for historical run load-back.

- [ ] **Step 8: Run pipeline snapshot tests**

Run:

```bash
uv run pytest tests/integration/test_data_pipeline_snapshots.py tests/integration/test_data_pipeline.py -q
```

Expected: failures only in tests still expecting old train/test manifest fields. Update those assertions to v2 semantics.

---

## Task 4: MLflow Active Snapshot Store Without Split Tags

**Files:**
- Modify: `automl/mlflow/store.py`
- Modify: `tests/unit/test_mlflow_store.py`
- Modify: `tests/contracts/test_runtime_completeness.py`

- [ ] **Step 1: Update active snapshot tags**

Remove split-specific active snapshot tags from `_active_data_snapshot_tags()`. It should return:

```python
{
    "data.prepare_event_id": prepare_event_id,
    "data.snapshot_name": snapshot_name_value,
    "data.snapshot_identity_hash": _snapshot_identity_hash(manifest),
    "data.manifest_uri": data_manifest_uri,
}
```

Do not log `data.split_scheme` on the overview run. Split is not part of the active data snapshot. Do not keep the old `data.snapshot_sha256` tag name; the durable name is `data.snapshot_identity_hash`.

- [ ] **Step 2: Update `latest_snapshot.json`**

Keep only a compact pointer/index. Do not copy the full GCS manifest into MLflow overview artifacts:

```python
data_manifest_uri = _required_str(manifest["gcs"], "manifest_uri")
active_pointer = {
    "schema_version": 2,
    "snapshot_name": snapshot_name_value,
    "prepare_event_id": prepare_event_id,
    "data_manifest_uri": data_manifest_uri,
    "snapshot_identity_hash": _snapshot_identity_hash(manifest),
    "prepare_event_uri": f"runs:/{experiment_overview_run_id}/{prefix}/prepare_event.json",
}
```

`snapshot_index.json` should use the same compact fields: `snapshot_name`, `prepare_event_id`, `data_manifest_uri`, and `snapshot_identity_hash`.

- [ ] **Step 3: Run store tests**

Run:

```bash
uv run pytest tests/unit/test_mlflow_store.py tests/contracts/test_runtime_completeness.py -q
```

Expected: update tests to stop expecting active overview `data.split_scheme` or `data.snapshot_sha256`.

---

## Task 5: Runner Logs One Data Contract Per Run

**Files:**
- Modify: `automl/runner/_execute.py`
- Modify: `automl/runner/_stages.py`
- Modify: `automl/mlflow/artifacts/data.py`
- Modify: `tests/integration/test_runner.py`
- Modify: `tests/integration/test_logging_contract_smoke.py`
- Modify/Create: `tests/integration/test_data_contract_writer.py`

- [ ] **Step 1: Add `write_data_contract()`**

In `automl/mlflow/artifacts/data.py`:

```python
def write_data_contract(
    *,
    trial_dir: Path,
    contract: RunDataContract,
) -> Path:
    if not isinstance(contract, RunDataContract):
        raise TypeError("contract must be a RunDataContract")
    out_dir = trial_dir / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "contract.json"
    out.write_text(json.dumps(contract.to_dict(), indent=2, default=str))
    return out
```

Export it from `automl/mlflow/artifacts/__init__.py` and `automl/mlflow/__init__.py`.

- [ ] **Step 2: Update runner data lineage extraction**

Replace train/test URI extraction with canonical data URI:

```python
snapshot_gcs = snapshot.manifest.get("gcs") or {}
snapshot_hashes = snapshot.manifest.get("hashes") or {}
data_uri = str(snapshot_gcs.get("data_uri") or "")
data_hash = str(snapshot_hashes.get("data_content_hash") or "")
snapshot_identity_hash = str(snapshot_hashes.get("snapshot_identity_hash") or "")
split_view = dict(snapshot.split_view)
train_hash = str(split_view["hashes"]["train_content_hash"])
test_hash = str(split_view["hashes"]["test_content_hash"])
split_view_hash = str(split_view["split_view_hash"])
data_shape = {
    "data": {
        "n_rows": int(snapshot.manifest["shape"]["n_rows"]),
        "n_columns": int(snapshot.manifest["shape"]["n_columns"]),
    },
    "train": {
        "n_rows": int(split_view["shape"]["n_train"]),
        "n_columns": int(snapshot.manifest["shape"]["n_columns"]),
    },
    "test": {
        "n_rows": int(split_view["shape"]["n_test"]),
        "n_columns": int(snapshot.manifest["shape"]["n_columns"]),
    },
}
```

Set tags:

```python
_mlflow_local.set_tag("data.snapshot_name", snapshot.snapshot_name)
_mlflow_local.set_tag("data.snapshot_identity_hash", snapshot_identity_hash)
_mlflow_local.set_tag("data.manifest_uri", str(snapshot.manifest["gcs"]["manifest_uri"]))
_mlflow_local.set_tag("data.contract_artifact", "data/contract.json")
_mlflow_local.set_tag("data.split_view_hash", split_view_hash)
```

Do not set `data.train_uri`, `data.test_uri`, `data.train_hash`, or `data.test_hash`. Those are in `data/contract.json`, not tags.

- [ ] **Step 3: Build and log the run data contract**

Build one typed contract from the active snapshot and split view:

```python
contract = RunDataContract(
    run=RunRef(
        project_name=project_name,
        experiment_id=experiment_id,
        trial_id=trial_id,
        run_id=_mlflow_run.info.run_id,
    ),
    snapshot=SnapshotRef(
        name=snapshot.snapshot_name,
        manifest_uri=str(snapshot.manifest["gcs"]["manifest_uri"]),
        identity_hash=snapshot_identity_hash,
        prepare_event_id=snapshot.prepare_event_id,
        target_column=target_col,
        shape=Shape(
            n_rows=int(snapshot.manifest["shape"]["n_rows"]),
            n_columns=int(snapshot.manifest["shape"]["n_columns"]),
        ),
    ),
    split=SplitContract(
        split_id_col=str(split_view["split_id_col"]),
        ranges=split_view["split"],
        view_hash=split_view_hash,
        train_content_hash=train_hash,
        test_content_hash=test_hash,
        shape=SplitShape(
            train=Shape(
                n_rows=int(split_view["shape"]["n_train"]),
                n_columns=int(snapshot.manifest["shape"]["n_columns"]),
            ),
            test=Shape(
                n_rows=int(split_view["shape"]["n_test"]),
                n_columns=int(snapshot.manifest["shape"]["n_columns"]),
            ),
        ),
    ),
)
data_contract_path = write_data_contract(trial_dir=trial_dir, contract=contract)
_mlflow_local.log_artifact(str(data_contract_path), artifact_path="data")
```

- [ ] **Step 4: Update run manifest data block**

Replace `train_uri` and `test_uri` fields with:

```python
data={
    "snapshot_name": snapshot.snapshot_name,
    "snapshot_identity_hash": snapshot_identity_hash,
    "prepare_event_id": snapshot.prepare_event_id,
    "data_uri": data_uri,
    "data_content_hash": data_hash,
    "split_view_hash": split_view_hash,
    "train_content_hash": train_hash,
    "test_content_hash": test_hash,
    "data_manifest_uri": str(snapshot.manifest["gcs"]["manifest_uri"]),
    "target_column": target_col,
    "split_id_col": str(split_view["split_id_col"]),
    "shape": data_shape,
    "contract_artifact": "data/contract.json",
}
```

- [ ] **Step 5: Run runner/logging tests**

Run:

```bash
uv run pytest tests/integration/test_runner.py tests/integration/test_logging_contract_smoke.py -q
```

Expected: PASS after updating assertions to `data/contract.json` plus compact data tags.

---

## Task 6: Historical Run Load-Back Uses Recorded Data Contract

**Files:**
- Modify: `automl/data/run_snapshot.py`
- Modify: `tests/unit/test_inspect.py`

- [ ] **Step 1: Write failing historical load-back test**

Update `tests/unit/test_inspect.py` so the fake run artifacts include:

- `data/contract.json` with a typed `RunDataContract` and 70/30 split ranges.
- Run tags matching the compact contract fields.

Assert that `inspect.load_data_snapshot(run_id)` returns the 70/30 rows even if the current project would use 80/20.

Also assert write/read symmetry:

```python
loaded = ai.load_data_snapshot(run_id, strict=True)

assert loaded.manifest["gcs"]["data_uri"] == data_uri
assert loaded.data_contract["snapshot"]["manifest_uri"] == manifest_uri
assert loaded.split_view["split_view_hash"] == contract["split"]["view_hash"]
assert loaded.split_view["hashes"]["train_content_hash"] == contract["split"]["train_content_hash"]
assert loaded.split_view["hashes"]["test_content_hash"] == contract["split"]["test_content_hash"]
```

- [ ] **Step 2: Update `load_data_snapshot_for_run()`**

Read:

```python
contract_payload = _read_run_json(client, run_id, "data/contract.json")
contract = RunDataContract.from_dict(contract_payload)
run = client.get_run(run_id)
```

Load:

```python
manifest = _read_gcs_json_uri(contract.snapshot.manifest_uri)
gcs = manifest["gcs"]
data_bucket, data_object = _parse_gcs_uri(_required_str(gcs, "data_uri"))
registry_bucket, registry_object = _parse_gcs_uri(_required_str(gcs, "feature_registry_uri"))
df_data = read_parquet_from_gcs(data_bucket, data_object)
registry_df = get_csv_from_gcs(registry_bucket, registry_object)
split = split_from_manifest(contract.split.ranges)
df_train, df_test = apply_split_view(
    df_data,
    split_id_col=contract.split.split_id_col,
    split=split,
)
```

Validate:

```python
validate_data_manifest_v2(manifest, df_data, registry_df)
validate_run_data_contract(contract_payload, manifest)
validate_split_view(manifest, contract.to_split_view(), df_train, df_test)
_validate_run_lineage(
    run=run,
    manifest=manifest,
    contract=contract,
    strict=strict,
)
```

Return:

```python
LoadedDataSnapshot(
    df_train=df_train,
    df_test=df_test,
    df_data=df_data,
    registry=FeatureRegistry.from_dataframe(registry_df),
    manifest=manifest,
    snapshot_name=_required_str(manifest, "snapshot_name"),
    prepare_event_id=contract.snapshot.prepare_event_id,
    run_id=run_id,
    source="run",
    data_contract=contract_payload,
    split_view=contract.to_split_view(),
)
```

- [ ] **Step 3: Add contract/tag validation**

`validate_run_data_contract()` should validate:

- `schema_version == 1`
- top-level keys are exactly `schema_version`, `run`, `snapshot`, and `split`.
- `contract["run"]` matches the GCS manifest project and experiment ids.
- `contract["snapshot"]` matches the GCS manifest snapshot name, manifest URI, identity hash, target column, prepare event id, and shape.
- `contract["split"]["split_id_col"]` matches the GCS manifest split-id column.
- `contract["split"]["ranges"]` parses through the same `Split` parser the loader uses.
- `contract["split"]["view_hash"]` matches the split fields.

`_validate_run_lineage()` should validate searchable tags only:

- `data.snapshot_name`
- `data.snapshot_identity_hash`
- `data.manifest_uri`
- `data.contract_artifact`
- `data.split_view_hash`

Never read tags to decide what to load. Tags are a searchable index over `data/contract.json`.

- [ ] **Step 4: Run inspect tests**

Run:

```bash
uv run pytest tests/unit/test_inspect.py -q
```

Expected: PASS.

---

## Task 7: Source Preview Naming Cleanup

**Files:**
- Modify: `automl/data/sources.py`
- Modify: `automl/data/adapters/local_csv.py`
- Modify: `automl/data/adapters/gcs_parquet.py`
- Modify: `automl/data/adapters/snowflake.py`
- Modify: `automl/data/spec.py`
- Modify: `automl/data/loader.py`
- Modify: `automl/cli/project.py`
- Modify: `automl/validate/builtin/contract_checks.py`
- Modify: `projects/example_homecredit/project.py`
- Modify: `tests/integration/test_data_pipeline.py`
- Modify: `tests/unit/data/test_spec.py`

- [ ] **Step 1: Rename source location fields**

Use one source action: `source.preview(nrows=...)`.

Change source constructors:

```python
@dataclass(frozen=True)
class LocalCSVSource(DataSource):
    data_path: str | Path
    hash_key: HashKey | None = None


@dataclass(frozen=True)
class GCSParquetSource(DataSource):
    data_path: str
    hash_key: HashKey | None = None


@dataclass(frozen=True)
class SnowflakeSource(DataSource):
    query: str | None = None
    query_path: str | Path | None = None
    base_table: str | None = None
    refresh_query_path: str | Path | None = None
```

No aliases for old field names.

- [ ] **Step 2: Rename preview row argument**

All sources should expose:

```python
def preview(self, *, nrows: int | None = 1000) -> pd.DataFrame:
```

CSV uses `pd.read_csv(..., nrows=nrows)`.

GCS parquet uses `read_parquet_head_from_gcs(..., nrows)` when `nrows` is not `None`.

Snowflake uses direct `query` or `query_path` and applies `_limit_sql(sql, nrows)`.

- [ ] **Step 3: Rename `DataSpec.dry_run_rows`**

Change `DataSpec` field:

```python
dry_run_nrows: int = 10_001
```

Update `build_pipeline()` to pass:

```python
dry_run_rows=spec.dry_run_nrows
```

Keep `DataPipeline.dry_run_rows` internal unless doing a full naming pass. The public config surface should be `dry_run_nrows`.

- [ ] **Step 4: Add hard-cutover source API ratchet**

Run:

```bash
rg "csv_path|gcs_uri|training_data_sql|base_data_sql|dry_run_rows|preview\\(.*rows=|rows=" automl tests projects docs
```

Expected remaining hits:

- `DataPipeline.dry_run_rows` internal constructor/attribute plumbing.
- Test names or assertions that explicitly verify retired names are absent.

Every public-facing source constructor, project template, example, notebook, validator, and docs reference should use:

- `data_path`
- `query`
- `query_path`
- `refresh_query_path`
- `preview(nrows=...)`
- `dry_run_nrows`

- [ ] **Step 5: Update tests**

Run:

```bash
uv run pytest tests/integration/test_data_pipeline.py tests/unit/data/test_spec.py tests/e2e/test_test_homecredit_project.py -q
```

Expected: PASS.

---

## Task 8: Notebook 1 Discovery Flow

**Files:**
- Modify: `automl/data/pipeline.py`
- Modify: `automl/data/preview.py` or `automl/data/loader.py`
- Modify: `automl/data/__init__.py`
- Modify: `projects/example_homecredit/notebooks/1_define_project_from_data.ipynb`
- Modify or rename if approved: `projects/example_homecredit/notebooks/1_discover_data_and_draft_project.ipynb`

- [ ] **Step 1: Remove project-context setup from notebook 1**

Delete the old explicit project metadata/resolver setup cells. Notebook
cells should use `ctx = automl.context()` instead.

- [ ] **Step 2: Start from direct source preview**

Use CSV by default, with GCS and Snowflake commented examples:

```python
NROWS = 1000
PROJECT_DIR = Path.cwd().parents[0] if Path.cwd().name == "notebooks" else Path.cwd()
SAMPLE_CSV = PROJECT_DIR / "data" / "application_train_sample.csv"

source = LocalCSVSource(data_path=SAMPLE_CSV, hash_key="SK_ID_CURR")
df = source.preview(nrows=NROWS)

# source = GCSParquetSource(
#     data_path=os.environ["EXAMPLE_HOMECREDIT_GCS_URI"],
#     hash_key="SK_ID_CURR",
# )
# df = source.preview(nrows=NROWS)

# source = SnowflakeSource(query="""
# SELECT
#     *,
#     MOD(ABS(HASH(SK_ID_CURR)), 100) AS SPLITID
# FROM database.schema.table
# """)
# df = source.preview(nrows=NROWS)
```

- [ ] **Step 3: Keep exploration explicit**

Use:

```python
display(hash_key_report(df).sort_values(["is_complete_unique", "unique_rate"], ascending=False))
df_with_split = add_split_id(df, hash_key="SK_ID_CURR")
display(split_report(df_with_split, target="TARGET", split=Split(train=[(0, 80)], test=[(80, 100)])))
```

- [ ] **Step 4: Draft project contract in memory**

Show the exact objects that go into `project.py`:

```python
TASK = BinaryClassification(target="TARGET")
DATA = DataSpec(
    source=source,
    metadata_cols=["SK_ID_CURR"],
    exclude_cols=[],
    dry_run_nrows=100,
)
EVAL = EvalSpec(primary=Auc())
RUN_CONFIG = RunConfig(
    experiment_id="example-homecredit",
    split=Split(train=[(0, 80)], test=[(80, 100)]),
    models=ModelsConfig(...),
    per_trial_seconds=600,
)
```

- [ ] **Step 5: Make draft pipeline preview context-free**

Notebook 1 is pre-`project.py` discovery. The no-log preview path must not call `resolve_project_context()` or require `projects/<name>/project.py`.

Update `DataPipeline.__init__` so context resolution is lazy:

```python
self.ctx = project_context
if self.ctx is not None:
    self.project_root = self.ctx.repo_root
    self.project_dir = self.ctx.project_dir
    self.project_name = self.ctx.project_name
else:
    self.project_root = Path(project_root).resolve() if project_root else Path.cwd().resolve()
    self.project_dir = self.project_root
    self.project_name = project_name or ""
```

Add a small context guard for methods that genuinely need project routing:

```python
def _require_context(self) -> ProjectContext:
    if self.ctx is None:
        raise RuntimeError(
            "ProjectContext is required for snapshot, MLflow, GCS, and evaluation-spec operations. "
            "Use build_pipeline() from a committed project.py, or use preview_pipeline() for no-log draft preview."
        )
    return self.ctx
```

Use `_require_context()` in:

- `_learning_payloads_from_mlflow()`
- `_validate_evaluation_columns()`
- `_snapshot_paths()`
- `prepare_data()`
- `load_data_snapshot()`
- source SQL artifact writers that need `project_dir`

Do not use `_require_context()` in:

- `source.preview(nrows=...)`
- `DataPipeline.preview()`
- `preview_pipeline(...)`

- [ ] **Step 6: Add no-log preview helper**

If `build_pipeline()` still requires committed `project.py`, add a small helper:

```python
def preview_pipeline(
    *,
    task: Task,
    data: DataSpec,
    run_config: RunConfig,
    dry_run: bool = True,
) -> DataPreview:
    pipeline = data.pipeline_cls(
        source=data.source,
        raw_target_column=task.target,
        exclude_cols=list(data.exclude_cols),
        metadata_cols=list(data.metadata_cols),
        dry_run=dry_run,
        dry_run_rows=data.dry_run_nrows,
        null_drop_threshold=data.null_drop_threshold,
        constant_drop_threshold=data.constant_drop_threshold,
        split=run_config.split,
        project_root=Path.cwd(),
    )
    return pipeline.preview()
```

Place it in `automl/data/preview.py` or `automl/data/loader.py`; export from `automl.data`.

- [ ] **Step 7: Execute notebook smoke**

Run:

```bash
uv run python -m pytest tests/integration/test_notebook_driven_ds_smoke.py -q
```

If this smoke does not execute notebook 1, run the notebook execution command already used in this repo for example notebooks.

---

## Task 9: Ratchets And Repository-Wide Cleanup

**Files:**
- Modify: tests that scan retired terms if present.
- Modify: docs/notebooks references.

- [ ] **Step 1: Search for retired train/test snapshot fields**

Run:

```bash
rg "train_uri|test_uri|train_path|test_path|train.parquet|test.parquet|data.split_scheme|split_scheme" automl tests projects docs
```

Every remaining hit must be one of:

- derived train/test content hashes in split view
- train/test model input contexts
- test names that explicitly assert no train/test snapshot paths

- [ ] **Step 2: Update contract docs and artifact listings**

Update descriptions:

- GCS `data/snapshots/<snapshot_name>/manifest.json`: canonical data snapshot lineage.
- MLflow run artifact `data/contract.json`: run-specific data load contract and pointer to the GCS manifest.
- MLflow tags: searchable mirrors only.

Remove docs that describe per-run `data/snapshot.json`, `data/inputs.json`, `data/source.json`, `data/schema.json`, `data/split.json`, or MLflow dataset inputs as current artifacts.

- [ ] **Step 3: Run focused tests**

Run:

```bash
uv run pytest tests/integration/test_data_pipeline_snapshots.py tests/integration/test_data_pipeline.py tests/integration/test_runner.py tests/integration/test_logging_contract_smoke.py tests/unit/test_inspect.py tests/unit/test_mlflow_store.py -q
```

Expected: PASS.

- [ ] **Step 4: Run full test suite if focused tests pass**

Run:

```bash
uv run pytest -q
```

Expected: PASS.

---

## Self-Review Notes

- The plan deliberately does not preserve old manifest compatibility.
- The only stored parquet snapshot is `data.parquet`.
- Train/test are still first-class in model training and the run data contract, but they are derived views over `data.parquet`.
- Active snapshot pointers are split-independent.
- Historical runs record a data contract at run time so load-back is reproducible after `project.py` changes.
- Dry-run remains separate because `snapshot_gcs_paths(..., dry_run=True)` already routes to a distinct namespace.
- Feature registry remains snapshot-level because it describes the canonical materialized dataset after pipeline column-role and quality processing, before run-specific splitting.
