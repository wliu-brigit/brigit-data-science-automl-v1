# Project Instructions - Example Home Credit

## Goal

Optimize binary-classification AUC for the Kaggle Home Credit default-risk
target. Treat this as the committed example project for AutoML dry-run
validation, logging, reproducibility, and MLflow artifact organization.

## Constraints (hard)

- Keep models fast enough for dry-run iteration.
- Do not read test labels directly.
- Do not change the target column or evaluation recipe.
- Ensembles are out of scope unless explicitly requested.

## Domain notes

- `SK_ID_CURR` is an application identifier and should remain metadata, not a
  model feature.
- The default data source is the committed 1,000-row CSV sample at
  `data/application_train_sample.csv`.
- Dry runs use the first 100 rows via `DATA.dry_run_rows`.
- Full-data runs can use `GCSParquetSource` by setting
  `EXAMPLE_HOMECREDIT_GCS_URI` to a prepared parquet file in GCS.
- `SPLITID` is derived by stable hash from `SK_ID_CURR`.
- `config.py` owns the default `EVAL`; custom evaluation helpers should live
  under a project-local `eval/` package only when needed.
- Missingness is meaningful in Home Credit data, so tree models or explicit
  imputation strategies are reasonable.

## Approaches to try

- Start with a simple logistic regression or tree baseline.
- Try LightGBM or XGBoost only when dependencies are available.
- Use registry-selected feature columns and preserve feature registry comments.
- Home Credit includes categorical string features. Encode categorical columns
  explicitly or restrict numeric baselines to registry `num` / `bool` features
  and comment any model-side feature removals.

## Approaches to avoid

- Do not use external Home Credit tables.
- Do not do broad ensembling in dry-run validation.

## Notebook workflow

- `0_understand_project_sessions_and_routes.ipynb` explains `use_project`,
  `config.py` loading, contextvar-backed active sessions, explicit
  `session=active`, and route flags for CLI subprocesses.
- `1_define_and_materialize_dataset.ipynb` previews source and pipeline choices,
  then creates the logged immutable dataset.
- `2_profile_logged_dataset.ipynb` loads the logged dataset artifacts and
  profiles the full dataset.
- `2_run_agent_automl.ipynb` launches the normal agent-driven AutoML loop.
- `3_author_new_trial.ipynb` creates a fresh notebook-authored model draft and
  can run it through the standard runner.
- `4_fork_existing_trial.ipynb` starts from a logged run and edits its model
  source.
- `5_reevaluate_existing_model.ipynb` logs new evaluations for an existing
  model run.
- `6_inspect_logged_runs_and_artifacts.ipynb` explains prior runs, model
  artifacts, datasets, validation outputs, and evaluation prediction loading.

## Full-data setup

The project-local helper downloads the Home Credit Kaggle competition data,
converts `application_train.csv` to parquet, and uploads it to the GCS URI you
choose:

```bash
uv run --script projects/example_homecredit/prepare_full_homecredit_gcs.py \
  --gcs-uri gs://your-bucket/path/application_train.parquet
```

Before running it, set `KAGGLE_API_TOKEN=KGAT_...` in the repo `.env`. This is
the only supported Kaggle credential path for this helper.

Then run the project with:

```bash
export EXAMPLE_HOMECREDIT_GCS_URI=gs://your-bucket/path/application_train.parquet
```

## Open questions

- Do the new MLflow artifacts fully explain data lineage, validation, model
  code, and agent activity for each trial?
