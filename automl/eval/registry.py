"""Eval dataset registry reads."""

from __future__ import annotations

from automl.eval.eval_dataset import EvalDataset
from automl.mlflow import routing as mlflow_routing
from automl.mlflow.experiment import eval_datasets
from automl.project import Session, session as active_project_session


def list_eval_datasets(*, session: Session | None = None) -> list[EvalDataset]:
    active = session if session is not None else active_project_session()
    root = _eval_dataset_root(active)
    datasets: list[EvalDataset] = []
    for prefix in eval_datasets.list_prefixes(root):
        record_uri = prefix.rstrip("/") + "/eval_dataset.json"
        try:
            if eval_datasets.blob_exists(record_uri):
                datasets.append(EvalDataset.from_dict(eval_datasets.read_record(record_uri)))
        except Exception:
            continue
    return sorted(datasets, key=lambda item: item.created_at, reverse=True)


def _eval_dataset_root(active: Session) -> str:
    prefix = mlflow_routing.experiment_route_prefix_for(
        gcs_prefix=active.config.gcs_prefix,
        project_name=active.project_name,
        experiment_id=active.active_experiment_id,
        namespace=active.namespace,
        dry_run=active.dry_run,
    )
    return f"gs://{active.config.gcs_bucket}/{prefix}/eval/datasets/"


__all__ = ["list_eval_datasets"]
