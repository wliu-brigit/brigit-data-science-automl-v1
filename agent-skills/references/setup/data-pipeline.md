# Data pipeline — wiring projects/<project_name>/config.py DATA

AutoML reads the data configuration from the `DATA` constant in `config.py`.

```python
from automl.data import DataSpec
from automl.data.sources import SnowflakeSource

DATA = DataSpec(
    source=SnowflakeSource(
        base_table="TRAINING_TABLE",  # name only; lands at {database}.{schema}.{base_table}
        base_table_sql="data/queries/base_table.sql",
        training_data_sql="data/queries/training_data.sql",
        unique_key="TRANSACTION_ID",
        split_group_key="USER_ID",  # optional; defaults to unique_key
    ),
    exclude_cols=[],
    metadata_cols=["USER_ID"],
)
```

The framework reads `DATA` at startup to build a `DataPipeline`, which owns
the dataset materialization lifecycle:

- `automl.data.materialize(refresh_data=False, refresh_source=False)`
  **attaches to the pinned active dataset by default** — no source read, no
  GCS write. Editing a source file surfaces as nothing until you pass
  `--refresh-data`; this is uniform across sources (deliberate: warehouse
  pulls are minutes, and loop iterations must compare against one pinned
  snapshot). The version in use is printed on every call.
  - `--refresh-data` re-derives the dataset from the source; content identity
    decides whether a new version actually mints (unchanged bytes attach to
    the existing version).
  - `--refresh-source` rebuilds the source's upstream first (the Snowflake
    base table; a no-op for file sources) and implies `--refresh-data`.
  - If the config has drifted from the recipe recorded on the pinned dataset,
    every call emits a **loud warning naming the changed fields** and still
    attaches as pinned — the harness makes drift visible; only you resolve it.
- `automl.data.load_dataset(split_name=...)` loads the active materialized
  dataset, optionally sliced by the project run config split name.
- `automl.data.load_dataset_by_id(dataset_id, split_name=...)` and
  `automl.data.load_dataset_by_trial(trial_id, split_name=...)` replay a
  specific materialized dataset or trial dataset slice.

The AutoML loop validates the active dataset before launching trials. Trial
runs load named train/eval slices read-only through the runner.

## Target declaration

The target column is declared in `TASK`, not `DATA`:

```python
from automl.project import BinaryClassification

TASK = BinaryClassification(target="TARGET")
```

This is the name as it appears in the source data. The framework standardizes
column names internally, and runtime code reads the standardized form through
`ctx.target_column`. Code that needs the user-declared value reads
`ctx.raw_target_column`.

## Sources

Three source classes live under `automl/data/sources/`. Pick one based
on where the training data lives, then pass it into `DataSpec`.

| Source class | External data | Required constructor args |
|---|---|---|
| `SnowflakeSource` | Snowflake | `base_table`, `base_table_sql`, `training_data_sql`, `unique_key` |
| `LocalCSVSource` | Local CSV file | `csv_path`, `unique_key` |
| `GCSParquetSource` | Parquet file in GCS | `gcs_uri`, `unique_key` |

Every source declares two keys:

- `unique_key` (required) — the stable row identifier: a column name, or a
  tuple of column names for composite keys. Hard-validated at materialize:
  the columns must be present and duplicate-free, or materialization errors
  loudly. Eval joins, augmentations, and any row-addressed operation ride on
  this guarantee.
- `split_group_key` (optional, defaults to `unique_key`) — the key whose hash
  assigns split buckets. Declare it separately only when splits must group by
  a coarser key than row identity (e.g. rows are transactions identified by
  `transaction_id`, but splits must group by `user_id` so one user never
  straddles train/test).

MLflow and GCS are platform requirements for every source. Snowflake
credentials are needed only when the active project uses `SnowflakeSource`.

All sources use the same dry-run interface. The default cap is
`DataSpec.dry_run_rows == 10_001`. File sources apply it exactly (CSV while
reading, GCS parquet as a row-limited read). Snowflake instead pulls a
**deterministic bucket sample** — `WHERE SPLIT_PCT < k`, with `k` sized so
the sample approximates `dry_run_rows` — so the same dry-run sees the same
rows every run and dry-run identity is stable (`dry_run_rows` is an
approximate target there, not an exact cap).

## Common requirements

Whatever source you pick, the loaded dataframe must include:

- The raw target column named by `TASK.target` after framework
  standardization.
- Any columns listed as `exclude_cols` or `metadata_cols`.

Every materialized dataset carries `SPLIT_PCT`, a deterministic integer 0-99
bucket column hashed from the source's `split_group_key` (which defaults to
`unique_key`). Who computes it differs by source — file sources have the
pipeline assign it in Python; Snowflake delivers it frozen from the base
table (injected at DDL time) — and the record notes the provenance
(`split: python(...)` vs `split: sql`). Downstream validation is shared:
`SPLIT_PCT` present, integer, 0-99, no missing values. For file sources, a
source column already named `SPLIT_PCT` is a loud collision error — the
pipeline owns the column and never silently recomputes over an ambiguous one
(symmetric with the Snowflake injection collision error below).

**Recorded caveat:** Snowflake `HASH()` and the pandas hash assign
*different* buckets for the same key values. Irrelevant within a project; a
project migrating CSV → Snowflake reshuffles split membership, so old and
new datasets are not split-comparable. Inherent to two engines; documented,
not designed away.

## Snowflake

Provide two SQL files in `projects/<project_name>/data/queries/`. Both are
**SELECTs** — the harness owns all DDL:

- `base_table.sql` — the single SELECT (or WITH) defining the base data:
  joins, CTEs, filters, feature SQL. The harness wraps it in
  `CREATE OR REPLACE TABLE {database}.{schema}.{base_table}` and **injects
  `SPLIT_PCT` from `split_group_key`** (`MOD(ABS(HASH(<key>)), 100)`).
  Do not emit `SPLIT_PCT` yourself — a body that already does is a loud
  collision error (one declaration only, in `config.py`). The generated DDL
  runs only when the base table is missing (bootstrap) or on
  `--refresh-source`.
- `training_data.sql` — the SELECT that pulls training rows from the base
  table. `SPLIT_PCT` flows through; dropping it from the projection is caught
  by validation with a "carry SPLIT_PCT through from the base table" message.

Substitutions in both files: `{database}` / `{schema}` (from the
`SNOWFLAKE_DATABASE` / `SNOWFLAKE_SCHEMA` environment) and `{base_table}`
(from the recipe). Identifiers are substituted verbatim — no case-mangling.

On every real pull the source first checks the **split invariant**
empirically against the table (`SPLIT_PCT == MOD(ABS(HASH(key)), 100)` for
all rows); a mismatch means the stored buckets are stale (the key changed,
or the table was built out-of-band) and errors naming `--refresh-source` —
the harness never auto-rebuilds.

Example skeleton for `training_data.sql`:

```sql
SELECT
    *
FROM {database}.{schema}.{base_table}
```

## Local CSV

```python
from automl.data import DataSpec
from automl.data.sources import LocalCSVSource

DATA = DataSpec(
    source=LocalCSVSource(csv_path="raw_data/your_file.csv", unique_key="row_id"),
    exclude_cols=[],
    metadata_cols=[],
)
```

`unique_key` is a stable column name, or a list of column names, available in
the raw file. During materialization, the pipeline computes
`SPLIT_PCT = stable_hash(split_group_key) % 100` (defaulting to the unique
key), so rows with the same group-key values land in the same split across
runs.

## GCS parquet

```python
from automl.data import DataSpec
from automl.data.sources import GCSParquetSource

DATA = DataSpec(
    source=GCSParquetSource(
        gcs_uri="gs://your-bucket/path/to/data.parquet",
        unique_key="row_id",
    ),
    exclude_cols=[],
    metadata_cols=[],
)
```

## Custom DataPipeline (escape hatch)

For projects that need to customize materialization, define a
`DataPipeline` subclass and pass it explicitly through `DataSpec.pipeline_cls`
in `projects/<project_name>/config.py`:

```python
from automl.data import DataSpec
from automl.data.pipeline import DataPipeline


class MyPipeline(DataPipeline):
    def run(self):
        loaded = super().run()
        # Optionally transform or replace the LoadedDataset.
        return loaded


DATA = DataSpec(
    source=...,
    pipeline_cls=MyPipeline,
)
```

This keeps the project handoff native to `config.py`: merely adding a
`data/pipeline.py` file does not change runtime behavior.

## What the pipeline does

`DataPipeline.run()` materializes source data through this deterministic
sequence before publishing an immutable dataset:

- Load raw rows through `DATA.source.load(...)`.
- Standardize source column names.
- Adopt the source-provided `SPLIT_PCT` under its canonical name (sources
  that own bucket assignment, i.e. Snowflake), or check for a colliding
  source column (file sources).
- Normalize target, key (`unique_key`/`split_group_key`), and metadata
  declarations against the standardized columns.
- Apply quality filters while preserving target, key, metadata, and
  `SPLIT_PCT` columns.
- Add deterministic `SPLIT_PCT` buckets hashed from `split_group_key`
  (file sources only — Snowflake's arrived frozen from the base table).
- Validate the ingestion edge: `unique_key` present and duplicate-free,
  `SPLIT_PCT` present, integer, 0-99.
- Build the `FeatureRegistry` from the filtered dataframe and declared column
  roles.
- Compute the immutable `Dataset` identity and component hashes.

You do not need to override this for normal projects; the source owns external
data access, and the pipeline owns materialization.

## Loading named splits

After materialization, training and evaluation slices are loaded through the
registry facade. The runner uses the project run config:

```python
import automl.data as data

run_config = active.config.require_run_config()
train = data.load_dataset(split_name=run_config.train_split, session=active)
eval_rows = data.load_dataset(split_name=run_config.eval_split, session=active)
```

For a specific materialized dataset or a replayed trial dataset, use:

```python
import automl.data as data

train = data.load_dataset_by_id(dataset_id, split_name="train", session=active)
trial_train = data.load_dataset_by_trial(trial_id, split_name="train", session=active)
```
