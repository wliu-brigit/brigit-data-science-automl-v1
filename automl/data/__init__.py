"""Data domain public API."""

from automl.data.contract import (
    DatasetRef,
    SliceContract,
    TrialDataContract,
    TrialRef,
    validate_loaded_dataset,
    validate_trial_data_contract,
    verify_loaded_slice,
    verify_trial_tag_lineage,
)
from automl.data.dataset import ComponentHashes, Dataset, DatasetIndex, LoadedDataset, LoadedSlice
from automl.data.features import FeatureEntry, FeatureRegistry
from automl.data.pipeline import DataPipeline, build_dataset, materialize
from automl.data.profile import Profile, get_profile, profile
from automl.data.recipe import compute_recipe, recipe_diff
from automl.data.registry import list_datasets, load_dataset, load_dataset_by_id, load_dataset_by_trial
from automl.data.sources import DataSource, GCSParquetSource, LocalCSVSource, SnowflakeSource
from automl.data.spec import DataSpec
from automl.data.split import Key, add_split_pct, split_report, validate_split_pct, validate_unique_key

__all__ = [
    "ComponentHashes",
    "DataPipeline",
    "DataSource",
    "DataSpec",
    "Dataset",
    "DatasetIndex",
    "DatasetRef",
    "FeatureEntry",
    "FeatureRegistry",
    "GCSParquetSource",
    "Key",
    "LoadedDataset",
    "LoadedSlice",
    "LocalCSVSource",
    "Profile",
    "SnowflakeSource",
    "SliceContract",
    "TrialDataContract",
    "TrialRef",
    "add_split_pct",
    "build_dataset",
    "compute_recipe",
    "get_profile",
    "list_datasets",
    "load_dataset",
    "load_dataset_by_id",
    "load_dataset_by_trial",
    "materialize",
    "profile",
    "recipe_diff",
    "split_report",
    "validate_loaded_dataset",
    "validate_trial_data_contract",
    "verify_loaded_slice",
    "verify_trial_tag_lineage",
]
