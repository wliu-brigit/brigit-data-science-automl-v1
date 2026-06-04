# Data pipeline — wiring projects/<project_name>/config.py DATA

AutoML reads the data configuration from the `DATA` constant in `config.py`.

```python
from automl.data import DataSpec
from automl.data.sources import SnowflakeSource

DATA = DataSpec(
    source=SnowflakeSource(
        base_table="MY_DB.MY_SCHEMA.TRAINING_TABLE",
        base_data_sql="data/queries/base_data.sql",
        training_data_sql="data/queries/training_data.sql",
    ),
    exclude_cols=[],
    metadata_cols=["USER_ID"],
)
```

The framework reads `DATA` at startup to build a `DataPipeline`, which owns
the dataset materialization lifecycle:

- `automl.data.materialize(refresh_source=False)` prepares or attaches to the
  latest immutable dataset for the active session.
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
| `SnowflakeSource` | Snowflake | `base_table`, `base_data_sql`, `training_data_sql`, `unique_key` |
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
`DataSpec.dry_run_rows == 10_001`. Snowflake applies the cap in SQL,
Local CSV applies it while reading the CSV, and GCS parquet uses a
row-limited PyArrow read.

## Common requirements

Whatever source you pick, the loaded dataframe must include:

- The raw target column named by `TASK.target` after framework
  standardization.
- Any columns listed as `exclude_cols` or `metadata_cols`.

The pipeline adds `SPLIT_PCT` as a uniformly distributed integer 0-99 column
during materialization, hashed from the source's `split_group_key` (which
defaults to `unique_key`). A source column already named `SPLIT_PCT` is a
loud error — the pipeline owns the column and never silently recomputes over
an ambiguous one.

## Snowflake

Provide two SQL files in `projects/<project_name>/data/queries/`:

- `base_data.sql` — DDL that creates or refreshes the upstream table. Run only
  when `refresh_source=True` is passed to `automl.data.materialize`.
- `training_data.sql` — query that pulls training rows. Must reference
  `{database}.{schema}.{base_table}`.

Both SQL files use `{database}.{schema}.{base_table}` substitutions.

Example skeleton for `training_data.sql`:

```sql
SELECT
    *
FROM {database}.{schema}.{base_table};
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
- Normalize target, key (`unique_key`/`split_group_key`), and metadata
  declarations against the standardized columns.
- Apply quality filters while preserving target, key, and metadata
  columns.
- Add deterministic `SPLIT_PCT` buckets hashed from `split_group_key`.
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
