# Step 2 — Dataset record & lifecycle

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The recipe (config-derived identity) is computed and recorded on
the dataset record; the record relocates GCS → MLflow as
`datasets/<id>/dataset.json`; the index/latest mirrors die; `materialize()`
gains the attach-as-pinned fast path with drift warnings and explicit
`--refresh-data` / `--refresh-source`; `dataframe_content_hash` becomes
order-insensitive; eval records are renamed for their nouns.

**Architecture:** A new `automl/data/recipe.py` computes the recipe from
(spec × session) and diffs it; `automl/mlflow/experiment/artifacts.py` swaps
its GCS index/manifest helpers for `write/read/list_dataset_record` on the
experiment overview run (the folder structure *is* the index; the existing
active-dataset **experiment tag** is the only pointer); `materialize()`
becomes: pointer → record → recipe compare → attach (silent or loud) |
re-derive → content-identity attach-or-mint. Three green commits:
(1) additive primitives, (2) the record/lifecycle cutover, (3) eval record
renames.

**Tech stack:** Python 3.12, pandas, MLflow artifact API (proxied in prod —
use the list-first `client.download_artifact` helper, never raw GET), GCS
for bytes only.

**Source of truth:** `../design.md` §2–§6, §13, §14 step 2. Resolved in
conversation 2026-06-04: helper names `write/read/list_dataset_record`;
`DatasetIndex` survives as an **in-memory view only**; recipe field list
below (incl. `pipeline_cls` as dotted path; `dry_run_rows` only in dry-run
sessions); every materialize/attach **prints the version in use**.

**Prereqs:** Step 1 landed (suite green on `unique_key`/`SPLIT_PCT`
vocabulary). **wendao has wiped old MLflow/GCS state manually** — the code
never deletes or migrates anything (ground rule). Old records are
unreadable by the new code on purpose; that is the forward-only posture.

**Carried in from the step-1 review session (2026-06-04):** before keys land
on the dataset record, unify `unique_key` normalization across data and eval —
data's `_normalize_key` sorts and rejects duplicates/blanks; eval's
`_normalize_unique_key` (eval_dataset.py) preserves order and checks nothing.
One shared normalizer (eval may import from `automl.data.split`; the domains
are co-dependent core), or a documented decision that eval preserves declared
order. While in those files, consider collapsing the ~5 copies of the
present+duplicate-free key check (eval_dataset/_load/base) onto
`split.validate_unique_key`.

---

## The recipe (settled field list)

```python
{
  "schema_version": 1,
  "source": <source.recipe_identity(project_dir)>,   # defaults to identity();
                                                     # Snowflake overrides in step 3 (SQL content hashes)
  "exclude_cols": sorted(...),
  "metadata_cols": sorted(...),
  "null_drop_threshold": float,
  "constant_drop_threshold": float,
  "pipeline_cls": "automl.data.pipeline.DataPipeline",   # dotted path — catches class swaps, not body edits
  "target": TASK.target (raw name),
  # only when session.dry_run (it's only read then; otherwise phantom drift):
  "dry_run_rows": int,
}
```

Explicitly out (design §3): `EVAL`, `RUN_CONFIG`, `Splits`, GCS
bucket/prefix. The recipe is recorded as a **readable dict** on the record,
never part of content identity.

## Where every record lives after this step

```
MLflow experiment overview run artifacts:
  datasets/<id>/dataset.json        ← written by write_dataset_record (one per version folder)
  datasets/<id>/source_trace/       ← unchanged
  datasets/<id>/profile/            ← unchanged
MLflow experiment tags:
  automl.active_dataset_id          ← unchanged (the one pointer; tags.ACTIVE_DATASET_ID)
GCS:
  .../datasets/<id>/data.parquet + feature_registry.csv   ← bytes only
DELETED: datasets/index + datasets/latest overview artifacts; GCS dataset_index.json + manifest.json.
```

---

## PART 1 — additive primitives (commit 1)

### Task 1: order-insensitive `dataframe_content_hash` (TDD)

**Files:**
- Modify: `tests/unit/utils/test_hashing.py` (create if absent — check
  `ls tests/unit/utils/` first; if the module's tests live elsewhere, add there)
- Modify: `automl/utils/hashing.py`

- [ ] **Step 1: Write the failing tests**

```python
import pandas as pd
import pytest

from automl.utils.hashing import dataframe_content_hash

pytestmark = pytest.mark.unit


def test_content_hash_is_row_order_insensitive():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    shuffled = df.sample(frac=1, random_state=7).reset_index(drop=True)
    assert dataframe_content_hash(df) == dataframe_content_hash(shuffled)


def test_content_hash_still_counts_duplicates():
    once = pd.DataFrame({"a": [1, 2]})
    twice = pd.DataFrame({"a": [1, 1, 2]})
    assert dataframe_content_hash(once) != dataframe_content_hash(twice)


def test_content_hash_still_sees_column_order_and_dtypes():
    df = pd.DataFrame({"a": [1], "b": [2]})
    assert dataframe_content_hash(df) != dataframe_content_hash(df[["b", "a"]])
```

- [ ] **Step 2:** Run: `uv run pytest tests/unit/utils/test_hashing.py -v` —
Expected: the order test FAILS against today's ordered row_hashes.

- [ ] **Step 3: Implement** — in `automl/utils/hashing.py`, the one change
(design §4: a canonical multiset — sort the per-row hash list, reorder no
data, the unique key plays no role):

```python
def dataframe_content_hash(df: pd.DataFrame) -> str:
    """Hash ordered columns, dtype strings, and the multiset of row hashes.

    Row hashes are sorted before fingerprinting so identity is insensitive
    to row order (a Snowflake SELECT has no order guarantee); duplicates
    still count, and no data is reordered.
    """
    payload = {
        "columns": list(df.columns),
        "dtypes": [str(dtype) for dtype in df.dtypes],
        "row_hashes": sorted(pd.util.hash_pandas_object(df, index=False).astype("uint64").tolist()),
    }
    return json_hash(payload)
```

- [ ] **Step 4:** `uv run pytest tests/unit/utils/test_hashing.py tests/unit/data -v`
— Expected: PASS (slice-contract hashes change value but tests compute
expected values through the same function).

### Task 2: `automl/data/recipe.py` (TDD)

**Files:**
- Create: `tests/unit/data/test_recipe.py`
- Create: `automl/data/recipe.py`
- Modify: `automl/data/sources/base.py`
- Modify: `automl/data/__init__.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Recipe: the config-derived identity of a materialization."""

from pathlib import Path

import pytest

from automl.data import DataSpec, LocalCSVSource
from automl.data.recipe import compute_recipe, recipe_diff
from automl.project import BinaryClassification, ProjectConfig, Session

pytestmark = pytest.mark.unit


def _spec_and_session(tmp_path, csv_name="a.csv", exclude=(), dry_run=False):
    spec = DataSpec(
        source=LocalCSVSource(csv_path=tmp_path / csv_name, unique_key="row_id"),
        exclude_cols=tuple(exclude),
    )
    config = ProjectConfig(
        project_name="demo",
        project_dir=Path(tmp_path),
        task=BinaryClassification(target="target"),
        data_spec=spec,
    )
    return spec, Session(config=config, dry_run=dry_run)
    # If Session's constructor differs (check automl/project/session.py for
    # how the integration tests build one), mirror the pattern used in
    # tests/integration/data_pipeline/test_materialize_load.py:_session —
    # the recipe needs only config.project_dir, config.raw_target_column,
    # session.dry_run, and the spec.


def test_recipe_is_computed_without_touching_the_source(tmp_path):
    spec, session = _spec_and_session(tmp_path, csv_name="missing.csv")
    recipe = compute_recipe(spec, session)  # file does not exist; must not be read
    assert recipe["source"]["kind"] == "local_csv"
    assert recipe["target"]
    assert "dry_run_rows" not in recipe


def test_recipe_canonicalizes_cosmetic_ordering(tmp_path):
    spec_a, session = _spec_and_session(tmp_path, exclude=("b", "a"))
    spec_b, _ = _spec_and_session(tmp_path, exclude=("a", "b"))
    assert compute_recipe(spec_a, session) == compute_recipe(spec_b, session)


def test_recipe_includes_dry_run_rows_only_in_dry_run_sessions(tmp_path):
    spec, session = _spec_and_session(tmp_path, dry_run=True)
    assert compute_recipe(spec, session)["dry_run_rows"] == spec.dry_run_rows


def test_recipe_diff_names_changed_fields_with_dotted_paths():
    recorded = {"target": "y", "source": {"kind": "local_csv", "csv_path": "a.csv"}}
    current = {"target": "y", "source": {"kind": "local_csv", "csv_path": "b.csv"}}
    assert recipe_diff(recorded, current) == ["source.csv_path"]
    assert recipe_diff(recorded, recorded) == []
```

- [ ] **Step 2:** Run: `uv run pytest tests/unit/data/test_recipe.py -v` —
Expected: FAIL (module missing).

- [ ] **Step 3: Implement `automl/data/recipe.py`**

```python
"""Dataset recipe: the config-derived identity of a materialization.

The recipe answers "SHOULD the dataset be different?" from config alone,
without touching any source. Its field list is a mechanical rule — the
transitive set of inputs materialize() reads — not a curated list.
Content identity (identity_hash) remains the only dedup key; the recipe is
recorded on the dataset record so drift reports can name fields.
"""

from __future__ import annotations

from typing import Any, Mapping


def compute_recipe(spec: Any, session: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "source": spec.source.recipe_identity(project_dir=session.config.project_dir),
        "exclude_cols": sorted(spec.exclude_cols),
        "metadata_cols": sorted(spec.metadata_cols),
        "null_drop_threshold": float(spec.null_drop_threshold),
        "constant_drop_threshold": float(spec.constant_drop_threshold),
        "pipeline_cls": f"{spec.pipeline_cls.__module__}.{spec.pipeline_cls.__qualname__}",
        "target": session.config.raw_target_column,
    }
    if session.dry_run:
        payload["dry_run_rows"] = int(spec.dry_run_rows)
    return payload


def recipe_diff(
    recorded: Mapping[str, Any],
    current: Mapping[str, Any],
    *,
    prefix: str = "",
) -> list[str]:
    """Dotted paths of fields that differ — the payload of a drift warning."""
    fields: list[str] = []
    for key in sorted(set(recorded) | set(current)):
        left, right = recorded.get(key), current.get(key)
        path = f"{prefix}{key}"
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            fields.extend(recipe_diff(left, right, prefix=f"{path}."))
        elif left != right:
            fields.append(path)
    return fields


__all__ = ["compute_recipe", "recipe_diff"]
```

- [ ] **Step 4:** Add the default hook to `DataSource`
(`automl/data/sources/base.py`):

```python
    def recipe_identity(self, *, project_dir: str | Path | None = None) -> dict[str, Any]:
        """Recipe-side identity: config-only, never touches the source.

        Defaults to identity(); sources whose identity references files by
        path override this to hash file *content* (SnowflakeSource, step 3).
        """
        del project_dir
        return self.identity()
```

- [ ] **Step 5:** Export from `automl/data/__init__.py`:
`from automl.data.recipe import compute_recipe, recipe_diff` (+ `__all__`).

- [ ] **Step 6:** `uv run pytest tests/unit/data/test_recipe.py -v` — PASS.

### Task 3: `load()` gains `refresh_source`; commit 1

**Files:**
- Modify: `automl/data/sources/base.py`, `local_csv.py`, `gcs_parquet.py`, `snowflake.py`

- [ ] **Step 1:** Extend the abstract signature (file sources accept and
ignore it — layer-1 verbs are no-ops for them, design §2):

```python
    @abstractmethod
    def load(
        self,
        *,
        project_dir: str | Path | None = None,
        nrows: int | None = None,
        refresh_source: bool = False,
    ) -> pd.DataFrame:
        """Load raw rows; refresh_source asks the source to rebuild its upstream first."""
```

Each concrete `load()` adds the parameter; `local_csv`/`gcs_parquet` bodies
start with `del refresh_source`. The Snowflake stub keeps raising.

- [ ] **Step 2:** `uv run pytest tests/unit tests/contracts tests/integration` — PASS.

- [ ] **Step 3: Commit**

```bash
git add -A automl tests
git commit -m "Add recipe primitives, order-insensitive content hash, refresh_source on load()

Additive groundwork for attach-as-pinned (design step 2, part 1)."
```

---

## PART 2 — record relocation + materialize lifecycle (commit 2)

### Task 4: dataset record helpers in MLflow experiment artifacts (TDD)

**Files:**
- Modify: `tests/unit/mlflow/test_experiment_dataset_artifacts.py`
- Modify: `automl/mlflow/experiment/artifacts.py`

- [ ] **Step 1: Write the failing tests** — the module's actual fixtures
(review-verified) are `bound_file_mlflow` (file-backed MLflow) and
`bound_artifacts` (fake GCS); the record helpers ride MLflow, so use
`bound_file_mlflow`. Delete the tests pinning the deleted surface in the
same change: `test_dataset_index_uri_*`, `test_read_dataset_index_*`,
`test_log_dataset_catalog_*`.

```python
def test_write_then_read_dataset_record_round_trips(bound_file_mlflow):
    payload = {"id": "v1_ab12cd34", "identity_hash": "sha256:x", "recipe": {"target": "y"}}
    uri = experiment_artifacts.write_dataset_record(payload, dataset_id="v1_ab12cd34")
    assert uri.startswith("runs:/") and uri.endswith("datasets/v1_ab12cd34/dataset.json")
    assert experiment_artifacts.read_dataset_record("v1_ab12cd34") == payload


def test_read_dataset_record_returns_none_when_absent(bound_file_mlflow):
    assert experiment_artifacts.read_dataset_record("v9_missing") is None


def test_list_dataset_records_returns_every_version_folder(bound_file_mlflow):
    experiment_artifacts.write_dataset_record({"id": "v1_a"}, dataset_id="v1_a")
    experiment_artifacts.write_dataset_record({"id": "v2_b"}, dataset_id="v2_b")
    records = experiment_artifacts.list_dataset_records()
    assert [record["id"] for record in records] == ["v1_a", "v2_b"]
```

- [ ] **Step 2:** Run; expected FAIL (helpers missing).

- [ ] **Step 3: Implement in `automl/mlflow/experiment/artifacts.py`** —
add the three helpers, delete `dataset_index_uri`, `read_dataset_index`,
`write_dataset_index`, `log_dataset_catalog`, `_latest_dataset_payload`,
`read_dataset_manifest`, `write_dataset_manifest` (and their `__all__`
entries):

```python
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
    try:
        entries = client.raw().list_artifacts(run_id, "datasets")
    except Exception:
        return []
    records: list[dict] = []
    for entry in entries:
        if not entry.is_dir:
            continue
        dataset_id = Path(entry.path).name
        record = read_dataset_record(dataset_id, experiment_id)
        if record is not None:
            records.append(record)
    return sorted(records, key=lambda record: str(record.get("id", "")))
```

(Imports of `gcs` for the byte helpers — `write_dataset_frame` etc. —
stay; only the JSON record/index surface moves.)

- [ ] **Step 4:** `uv run pytest tests/unit/mlflow/test_experiment_dataset_artifacts.py -v` — PASS.

### Task 5: Dataset value object — recipe + record_uri, manifest property dies

**Files:**
- Modify: `automl/data/dataset.py`
- Modify: `automl/data/contract.py`

- [ ] **Step 1: `Dataset`** — add two fields and drop the GCS-manifest property:

```python
    recipe: Mapping[str, Any] = field(default_factory=dict)   # readable dict, design §3
    record_uri: str = ""    # runs:/<overview_run>/datasets/<id>/dataset.json, set at persist
```

(use `dataclasses.field`; place after `experiment_id`, before
`schema_version`). Delete the `manifest_gcs_uri` property. `from_dict` adds
`recipe=dict(payload.get("recipe", {}))`,
`record_uri=str(payload.get("record_uri", ""))`; `to_dict` adds `"recipe"`
and drops `"manifest_gcs_uri"` — but **not** `"record_uri"`: the persisted
record never stores a pointer to itself; `read_dataset_record` injects it
on the way back (Task 4).

- [ ] **Step 2: `DatasetRef`** (`automl/data/contract.py`) —
`manifest_uri` field → `record_uri`; `from_dataset` uses
`record_uri=dataset.record_uri`; `from_dict` key `"manifest_uri"` →
`"record_uri"`.

- [ ] **Step 3:** Trial tag lineage — **all four** `manifest_uri` consumers
(review found two beyond the obvious pair):
`automl/data/contract.py` has **two** `"data.manifest_uri"` literals
(`verify_trial_tag_lineage` and a second occurrence ~line 187 — both →
`"data.record_uri"`); `automl/runner/data_artifacts.py:20`
`log_data_contract` tag payload; `automl/runner/manifest_artifacts.py:46`
reads `contract.dataset.manifest_uri` → `record_uri`. Update
`tests/unit/data/test_contract_validators.py` (the L4 test pins the exact
tag keys) in the same change.

### Task 6: `materialize()` — attach-as-pinned lifecycle

**Files:**
- Modify: `automl/data/pipeline.py`
- Modify: `tests/unit/data/test_materialize_return_shape.py`
- Modify: `tests/integration/data_pipeline/test_materialize_load.py`

- [ ] **Step 1: New public signature and flow** — replace `materialize` /
`_materialize_bound` in `automl/data/pipeline.py`:

```python
import logging

logger = logging.getLogger(__name__)


def materialize(
    *,
    refresh_data: bool = False,
    refresh_source: bool = False,
    include_rows: bool = True,
    session: Session | None = None,
) -> LoadedDataset | Dataset:
    """Attach to the active pinned dataset, or (re-)derive it on explicit refresh.

    refresh_source implies refresh_data: rebuilding layer 1 only matters if
    layer 2 is re-derived from it. Neither flag is ever passed by the agent
    loop — humans ask for refreshes (design §14).
    """
    refresh_data = refresh_data or refresh_source
    active = _session(session)
    with mlflow_client.bound_for(active, experiment_id=active.active_experiment_id):
        result = _materialize_bound(
            active=active,
            refresh_data=refresh_data,
            refresh_source=refresh_source,
            include_rows=include_rows,
        )
    return result


def _materialize_bound(
    *, active: Session, refresh_data: bool, refresh_source: bool, include_rows: bool
) -> LoadedDataset | Dataset:
    spec = active.config.require_data_spec()
    recipe = compute_recipe(spec, active)

    if not refresh_data:
        attached = _attach_active(active, recipe)
        if attached is not None:
            if not include_rows:
                return attached
            from automl.data.registry import load_dataset_by_id

            return load_dataset_by_id(attached.id, session=active)

    pipeline = spec.pipeline_cls(spec, active, refresh_source=refresh_source)
    loaded = pipeline.run()
    records = experiment_artifacts.list_dataset_records()
    existing = _record_for_identity(records, loaded.dataset.identity_hash)

    if existing is not None:
        # Content unchanged: attach; update the recorded recipe last-wins
        # (the user explicitly refreshed — "this recipe currently produces
        # this content" is honest provenance, design §3).
        dataset = replace(
            Dataset.from_dict(existing),
            recipe=recipe,
        )
        record_uri = experiment_artifacts.write_dataset_record(
            dataset.to_dict(), dataset_id=dataset.id
        )
        dataset = replace(dataset, record_uri=record_uri)
        mlflow_experiment.set_active_dataset(dataset.id, experiment_id=active.active_experiment_id)
        logger.info("content unchanged — attached to %s (recipe updated)", dataset.id)
        loaded = LoadedDataset(dataset=dataset, df=loaded.df, registry=loaded.registry)
        return loaded if include_rows else dataset

    dataset = replace(
        loaded.dataset,
        id=_next_dataset_id(records, loaded.dataset),
        recipe=recipe,
    )
    object_state = _dataset_object_state(dataset)
    if any(object_state.values()):
        present = [name for name, exists in object_state.items() if exists]
        raise DataError(
            f"GCS objects already present for new dataset {dataset.id}: {present} — "
            "refusing to overwrite; wipe state manually if this is intentional"
        )
    experiment_artifacts.write_dataset_frame(dataset.data_gcs_uri, loaded.df)
    experiment_artifacts.write_registry(dataset.registry_gcs_uri, loaded.registry.to_dataframe())
    record_uri = experiment_artifacts.write_dataset_record(
        dataset.to_dict(), dataset_id=dataset.id
    )
    dataset = replace(dataset, record_uri=record_uri)  # in memory only; the reader re-derives it
    _log_source_trace(dataset, spec.source.artifact_files(pipeline))
    mlflow_experiment.set_active_dataset(dataset.id, experiment_id=active.active_experiment_id)
    logger.info("minted %s and set it active", dataset.id)
    loaded = LoadedDataset(dataset=dataset, df=loaded.df, registry=loaded.registry)
    return loaded if include_rows else dataset


def _attach_active(active: Session, recipe: dict) -> Dataset | None:
    """The default fast path: resolve pointer -> record -> recipe compare."""
    active_id = mlflow_experiment.get_active_dataset(experiment_id=active.active_experiment_id)
    if active_id is None:
        return None
    record = experiment_artifacts.read_dataset_record(active_id)
    if record is None:
        return None
    dataset = Dataset.from_dict(record)
    drift = recipe_diff(dataset.recipe, recipe)
    if drift:
        logger.warning(
            "recipe drift: %s changed since %s — running against %s as pinned; "
            "pass --refresh-data to re-derive%s",
            ", ".join(drift),
            dataset.id,
            dataset.id,
            (
                " (base_table.sql changed: only --refresh-source rebuilds the base table)"
                if any(field.startswith("source.base_table_sql") for field in drift)
                else ""
            ),
        )
    else:
        logger.info("attached to %s (pinned)", dataset.id)
    return dataset


def _record_for_identity(records: list[dict], identity_hash: str) -> dict | None:
    for record in records:
        if record.get("identity_hash") == identity_hash:
            return record
    return None


def _next_dataset_id(records: list[dict], dataset: Dataset) -> str:
    max_version = 0
    for record in records:
        match = re.match(r"^v(\d+)_", str(record.get("id", "")))
        if match:
            max_version = max(max_version, int(match.group(1)))
    return f"v{max_version + 1}_{dataset.identity_hash.removeprefix('sha256:')[:8]}"
```

Notes against today's code: `DatasetIndex` reads/writes, the
`read_dataset_manifest` re-read + `_validate_existing_dataset_matches_candidate`
cross-check, and `log_dataset_catalog` calls are gone (the record IS what we
just read/wrote); `_dataset_object_state` drops its `"manifest"` entry
(record lives in MLflow now); imports updated
(`from automl.data.recipe import compute_recipe, recipe_diff`; drop
`DatasetIndex` import). `DataPipeline.__init__` keeps `refresh_source` and
**now forwards it**: `run()`'s `source.load(...)` call gains
`refresh_source=self.refresh_source`.

- [ ] **Step 2: Behavior tests** — in
`tests/integration/data_pipeline/test_materialize_load.py`, replace the
index/latest/dataset_index assertions with the new lifecycle (reusing the
file-backed-MLflow + fake-GCS fixtures already in the module). Existing
tests that pin the deleted surface (review-enumerated, all in this module):
the `match="partial dataset objects"` test (~line 306 — the branch is
replaced by the refusing-to-overwrite guard; rewrite it against the new
message), the GCS `dataset_index.json` blob assertions (~153–175), and the
`manifest_gcs_uri` assertions (~164, 217–226, 288, 313–314). New tests:

```python
def test_first_materialize_mints_v1_and_sets_pointer(...):
    loaded = materialize(session=active)
    assert loaded.dataset.id.startswith("v1_")
    assert mlflow_experiment.get_active_dataset(experiment_id=...) == loaded.dataset.id
    assert experiment_artifacts.read_dataset_record(loaded.dataset.id)["recipe"]["target"]

def test_second_materialize_attaches_without_reading_the_source(...):
    materialize(session=active)
    csv_path.unlink()                      # source gone — fast path must not touch it
    again = materialize(session=active, include_rows=False)
    assert again.id.startswith("v1_")

def test_recipe_drift_warns_with_field_diff_and_attaches_pinned(caplog, ...):
    materialize(session=active)
    # change exclude_cols in the spec, rebuild session
    with caplog.at_level(logging.WARNING):
        result = materialize(session=drifted, include_rows=False)
    assert result.id.startswith("v1_")     # still pinned
    assert "recipe drift" in caplog.text and "exclude_cols" in caplog.text

def test_refresh_data_rederives_and_attaches_when_content_unchanged(...):
    first = materialize(session=active, include_rows=False)
    second = materialize(session=active, refresh_data=True, include_rows=False)
    assert second.id == first.id           # content identity dedups

def test_refresh_data_mints_new_version_when_content_changed(...):
    first = materialize(session=active, include_rows=False)
    # append a row to the CSV
    second = materialize(session=active, refresh_data=True, include_rows=False)
    assert second.id.startswith("v2_")
    # both records remain readable; pointer moved
    assert experiment_artifacts.read_dataset_record(first.id) is not None
    assert mlflow_experiment.get_active_dataset(...) == second.id

def test_attach_after_refresh_updates_recorded_recipe_last_wins(...):
    materialize(session=active)
    # Drift the recipe in a way that does NOT change content identity.
    # NOTE (review finding): exclude_cols is NOT safe for this — it flags
    # registry entries (feature=False), which changes the registry content
    # hash and therefore identity_hash. Use a threshold that changes no
    # actual column decision instead, e.g. null_drop_threshold 0.99 -> 0.98
    # on a frame with no column in (0.98, 0.99] null share — and assert the
    # premise explicitly before relying on it:
    assert drifted_loaded.dataset.identity_hash == first.identity_hash
    result = materialize(session=drifted, refresh_data=True, include_rows=False)
    record = experiment_artifacts.read_dataset_record(result.id)
    assert recipe_diff(record["recipe"], compute_recipe(drifted_spec, drifted)) == []
```

- [ ] **Step 3:** `uv run pytest tests/integration/data_pipeline tests/unit/data -v`
— PASS.

### Task 7: registry reads via records

**Files:**
- Modify: `automl/data/registry.py`
- Modify: `automl/data/dataset.py` (`DatasetIndex` slimmed)

- [ ] **Step 1: `list_datasets`** — assemble the in-memory view:

```python
def list_datasets(*, session: Session | None = None) -> DatasetIndex:
    active = _session(session)
    with mlflow_client.bound_for(active, experiment_id=active.active_experiment_id):
        records = experiment_artifacts.list_dataset_records()
        return DatasetIndex(
            datasets=tuple(Dataset.from_dict(record) for record in records),
            active_dataset_id=mlflow_experiment.get_active_dataset(
                experiment_id=active.active_experiment_id
            ),
        )
```

`DatasetIndex` in `dataset.py` loses `from_dict`/`to_dict`/`schema_version`
(it is never persisted again) and keeps `datasets`, `active_dataset_id`,
`active`, `to_dataframe`.

- [ ] **Step 2: `load_dataset_by_id`** — drop the index lookup + GCS manifest
read:

```python
    record = experiment_artifacts.read_dataset_record(dataset_id)
    if record is None:
        raise KeyError(f"dataset {dataset_id!r} not found")
    dataset = Dataset.from_dict(record)
```

(rest of the function — registry/frame reads from GCS, slice filtering —
unchanged; `_dataset_by_id` helper deleted.)

- [ ] **Step 3:** `automl/data/profile.py` imports nothing removed — verify
`uv run pytest tests/unit/data tests/integration/data_pipeline -v` — PASS.

### Task 8: CLI + agent flags

**Files:**
- Modify: `automl/cli/data.py`
- Modify: `automl/agent/run_options.py`
- Modify: `agent-skills/skills/automl/scripts/preflight.py`
- Modify: `agent-skills/skills/automl/scripts/render_context.py`
- Modify: `tests/unit/cli/test_cli_catalog.py`
- Modify: `tests/contracts/test_skill_commands.py`
- Modify: `agent-skills/references/setup/data-pipeline.md`

- [ ] **Step 1: `automl/cli/data.py`**

```python
materialize_parser = data_sub.add_parser("materialize")
materialize_parser.add_argument(
    "--refresh-data", action="store_true",
    help="re-derive the dataset from the source (default attaches to the pinned active dataset)",
)
materialize_parser.add_argument(
    "--refresh-source", action="store_true",
    help="rebuild the source's upstream (Snowflake base table) first; implies --refresh-data",
)

def _materialize(args: argparse.Namespace) -> int:
    dataset = materialize(
        refresh_data=args.refresh_data,
        refresh_source=args.refresh_source,
        include_rows=False,
        session=session_from_args(args),
    )
    print_json(dataset)
    return 0
```

- [ ] **Step 2: `automl/agent/run_options.py`** — `ExperimentRunOptions`
gains `refresh_data: bool = False` (next to `refresh_source`);
`add_experiment_run_options` adds
`parser.add_argument("--refresh-data", action="store_true")`;
`options_from_namespace` extracts it; `skill_command_args` serializes
`--refresh-data` when set. (The loop itself never sets either flag — they
exist so a *human* invoking `experiment run` can refresh once at start.)

- [ ] **Step 3: agent-skills scripts** — in `preflight.py`: there is **no
retired-flags list** (review-verified) — `--refresh-data` is rejected today
only because `parse_known_args` reports it unknown. The real work: add
`refresh_data` to `_base_payload` and the run-mode return, mirroring how
`refresh_source` threads through (~lines 28, 42, 122); it becomes accepted
automatically once `add_experiment_run_options` registers the flag. In
`render_context.py`, forward both to the `materialize_dataset` safe command:

```python
if invocation.get("refresh_data"):
    materialize_dataset_args.append("--refresh-data")
if invocation.get("refresh_source"):
    materialize_dataset_args.append("--refresh-source")
```

- [ ] **Step 4: Contract tests** — in `tests/contracts/test_skill_commands.py`:
**replace** `test_automl_render_context_rejects_retired_refresh_data_flag`
with `test_automl_render_context_forwards_refresh_data_to_materialize`
(mirror of the existing refresh-source forwarding test, asserting the
rendered command ends `" data materialize --refresh-data"`). In
`tests/unit/cli/test_cli_catalog.py`, extend
`test_experiment_run_forwards_refresh_and_confirmation_flags` with
`--refresh-data` in both the argv and the expected `automl_args`, and update
the `fake_materialize` kwargs assertions (`refresh_data`/`refresh_source`
both present).

- [ ] **Step 5: Reference doc** — update
`agent-skills/references/setup/data-pipeline.md`: describe the
attach-as-pinned default (file edits surface as nothing until
`--refresh-data` — uniform across sources, deliberate), both flags, the
drift warning, and that the active version is printed on every call.

- [ ] **Step 6:** `uv run pytest tests/unit/cli tests/contracts -v` — PASS.

### Task 9: full sweep of survivors, commit 2

- [ ] **Step 1:** Find every remaining reference to the deleted surface:

```bash
grep -rn "dataset_index\|read_dataset_manifest\|write_dataset_manifest\|log_dataset_catalog\|manifest_gcs_uri\|manifest_uri\|DatasetIndex.from_dict" automl tests agent-skills --include="*.py" | grep -v eval
```

Update each (known stragglers: `tests/unit/agent/test_proposer_context.py`
fixtures, `tests/unit/data/test_materialize_return_shape.py`,
`automl/agent/proposer_context.py` if it renders manifest URIs,
`tests/integration/data_pipeline/test_trial_replay.py` contract fixtures —
`manifest_uri` → `record_uri`).

- [ ] **Step 2:** `uv run pytest tests/unit tests/contracts tests/integration` — PASS.

- [ ] **Step 3: Commit**

```bash
git add -A automl tests agent-skills
git commit -m "Relocate dataset records to MLflow; attach-as-pinned materialize with explicit refresh flags

datasets/<id>/dataset.json on the overview run replaces the GCS manifest;
index/latest mirrors deleted (folders are the index; the experiment tag is
the pointer); recipe recorded on the record, drift warns with a field diff
and attaches pinned; --refresh-data re-derives, --refresh-source implies it
(design step 2, part 2)."
```

---

## PART 3 — eval records renamed for their nouns (commit 3)

### Task 10: `manifest.json` → `eval_dataset.json` / `augmentation.json`

**Files (expanded per review — the unlisted consumers would have left the
commit red):**
- Modify: `automl/eval/eval_dataset.py`
- Modify: `automl/eval/prepare.py`
- Modify: `automl/eval/_load.py`
- Modify: `automl/eval/registry.py` (builds `prefix + "/manifest.json"` at
  ~line 15-19 and calls `eval_datasets.read_manifest`)
- Modify: `automl/eval/evaluate.py` (reads `loaded.dataset.manifest_gcs_uri`
  at ~line 100; carries `eval_dataset_manifest_uri` at ~166, 175)
- Modify: `automl/eval/results.py` (`eval_dataset_manifest_uri` field at
  ~77, 90, 103 — **renames to `eval_dataset_record_uri`**, same noun rule)
- Modify: `automl/mlflow/experiment/eval_datasets.py`
- Tests: `tests/unit/eval/test_eval_dataset_identity.py`,
  `tests/unit/eval/test_augmentations.py`,
  `tests/unit/eval/test_eval_dataset_storage_seam.py` (patches the renamed seam),
  `tests/unit/eval/test_eval_thin_path.py` (~121-125),
  `tests/unit/eval/test_results_schemas.py` (`eval_dataset_manifest_uri`),
  `tests/unit/mlflow/test_eval_predictions_artifacts.py`,
  `tests/integration/eval/test_eval_dataset_persistence.py`,
  `tests/integration/eval/test_evaluate_persistence.py` (~82-86),
  `tests/integration/eval/test_augmentation_integration.py` (~84-86),
  `tests/e2e/test_eval_dataset_breadth.py`

- [ ] **Step 1: Rename map** (a persisted record is named for the noun it
serializes — design §6; locations stay GCS, only names change):

| Old | New |
|---|---|
| `.../eval/datasets/<id>/manifest.json` | `.../eval/datasets/<id>/eval_dataset.json` |
| `.../augmentations/<name>__<hash8>/manifest.json` | `.../augmentations/<name>__<hash8>/augmentation.json` |
| `EvalDataset.manifest_gcs_uri` property | `EvalDataset.record_gcs_uri` |
| `Augmentation.manifest_gcs_uri` property | `Augmentation.record_gcs_uri` |
| `manifest_uri_for(...)` | `record_uri_for(...)` |
| `eval_datasets.read_manifest` / `write_manifest` | `read_record` / `write_record` |
| `to_dict()` key `"manifest_gcs_uri"` | `"record_gcs_uri"` |

Apply mechanically across the five library files (the `_load.py` and
`prepare.py` augmentation listings build `prefix + "/manifest.json"` —
those literals become `"/augmentation.json"`). `Predictions.manifest_dict`
and trial-level artifacts are **out of scope** (trial contract unchanged,
design §6).

- [ ] **Step 2:** Update the listed tests with the same map (they pin URI
endings and payload keys — assertions move with the shape).

- [ ] **Step 3: Verify — BEFORE committing, not after** (review: this gate
catches the consumers the file list might still miss):

```bash
grep -rn "manifest" automl/eval automl/mlflow/experiment/eval_datasets.py
```

Expected: no hits for the renamed surface (predictions/trial manifests in
other modules are fine). Any hit = a missed consumer; fix it first.

- [ ] **Step 4:** `uv run pytest tests/unit/eval tests/integration/eval tests/unit/mlflow -v` — PASS.

### Task 11: green suite, handoff, commit 3

- [ ] **Step 1:** `uv run pytest tests/unit tests/contracts tests/integration` — PASS.

- [ ] **Step 2:** Update `docs/HANDOFF.md` (step 2 landed; next: step 3,
Snowflake).

- [ ] **Step 3: Commit**

```bash
git add -A automl tests docs/HANDOFF.md
git commit -m "Rename eval manifests for their nouns: eval_dataset.json + augmentation.json

(design step 2, part 3)"
```

---

## Self-review checklist

- [ ] Fast path provably reads no source and writes no GCS (the
  `csv_path.unlink()` test).
- [ ] Drift warning repeats on every call while drifted (call twice in the
  caplog test), never blocks, never auto-derives.
- [ ] `--refresh-source` without `--refresh-data` still re-derives (implies).
- [ ] Nothing anywhere deletes or migrates old MLflow/GCS state; the
  refuse-to-overwrite guard raises instead of clobbering.
- [ ] `runs:/` record URIs resolve through `client.download_artifact`
  (list-first — prod proxy 500-on-missing is already mitigated there).
- [ ] Pointer behavior: pinning v2 then calling materialize attaches v2 and
  *prints it*; minting v3 moves the pointer; old records stay readable.
- [ ] `grep -rn "refresh-data" agent-skills` shows it forwarded, not retired.
