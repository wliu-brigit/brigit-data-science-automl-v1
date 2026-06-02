"""Eval domain public API."""

from automl.eval._load import load_eval_dataset
from automl.eval.base import EvalSpec, Metric
from automl.eval.eval_dataset import Augmentation, EvalDataset
from automl.eval.evaluate import evaluate, evaluate_frame
from automl.eval.metrics import Auc, LogLoss, ThresholdSweep
from automl.eval.prepare import prepare_eval_augmentation, prepare_eval_dataset
from automl.eval.registry import list_eval_datasets
from automl.eval.results import EvalIndex, EvalIndexEntry, EvalResult, Predictions

__all__ = [
    "Auc",
    "Augmentation",
    "EvalDataset",
    "EvalIndex",
    "EvalIndexEntry",
    "EvalResult",
    "EvalSpec",
    "LogLoss",
    "Metric",
    "Predictions",
    "ThresholdSweep",
    "evaluate",
    "evaluate_frame",
    "load_eval_dataset",
    "list_eval_datasets",
    "prepare_eval_augmentation",
    "prepare_eval_dataset",
]
