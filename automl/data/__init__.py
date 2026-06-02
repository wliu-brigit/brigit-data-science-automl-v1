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
from automl.data.registry import list_datasets, load_dataset, load_dataset_by_id, load_dataset_by_trial
from automl.data.sources import DataSource, GCSParquetSource, LocalCSVSource, SnowflakeSource
from automl.data.spec import DataSpec
from automl.data.split import HashKey, add_split_id, hash_key_columns, split_report

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
    "HashKey",
    "LoadedDataset",
    "LoadedSlice",
    "LocalCSVSource",
    "Profile",
    "SnowflakeSource",
    "SliceContract",
    "TrialDataContract",
    "TrialRef",
    "add_split_id",
    "build_dataset",
    "get_profile",
    "hash_key_columns",
    "list_datasets",
    "load_dataset",
    "load_dataset_by_id",
    "load_dataset_by_trial",
    "materialize",
    "profile",
    "split_report",
    "validate_loaded_dataset",
    "validate_trial_data_contract",
    "verify_loaded_slice",
    "verify_trial_tag_lineage",
]
