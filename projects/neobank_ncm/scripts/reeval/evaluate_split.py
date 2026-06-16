"""Evaluate a logged trial model on any named split (default: oot).

Resolves the named split recorded with the dataset, scores the trial's
logged model on it, and attaches the result to that trial's MLflow run —
the same two library calls the runner uses for the in-loop eval, pointed
at the requested split. Two sanctioned uses:

- `--split oot` — legacy Phase 5, run ONCE on the winner. The `oot` split
  (Jan–Feb 2026, known-only) is never touched by the AutoML loop.
- `--split train_known` — the legacy train known-only diagnostic (the
  runner's automatic train eval skips on this project's NULL-target rows).

Usage (needs warehouse/MLflow access):

    uv run python projects/neobank_ncm/scripts/reeval/evaluate_split.py \
        --model-run-id <trial's MLflow run id> [--split oot] [--dataset-id <id>]

The headline comparison: oot AUC here vs the legacy v3 OOT known-only AUC
(data/legacy/preprocessor_meta.json -> performance.oot_known_only_auc).
"""

from __future__ import annotations

import argparse
import json

import automl.data as data
from automl.eval import evaluate, prepare_eval_dataset
from automl.project import use_project


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-run-id", required=True, help="MLflow run id of the winning trial")
    parser.add_argument(
        "--dataset-id",
        default=None,
        help="dataset to slice (defaults to the project's active dataset)",
    )
    parser.add_argument("--split", default="oot", help='named split to score (default "oot")')
    parser.add_argument(
        "--namespace",
        default="",
        help="MLflow namespace the trial lives under (e.g. qa/... for QA runs)",
    )
    args = parser.parse_args()

    session = use_project("neobank_ncm", namespace=args.namespace)

    dataset_id = args.dataset_id
    if dataset_id is None:
        dataset_id = data.load_dataset(split_name=args.split, session=session).id

    eval_dataset, _ = prepare_eval_dataset(
        session=session, dataset_id=dataset_id, split=args.split
    )
    result = evaluate(
        session=session,
        model_run_id=args.model_run_id,
        eval_dataset_id=eval_dataset.id,
        label=args.split,
    )

    print(
        json.dumps(
            {
                "model_run_id": args.model_run_id,
                "dataset_id": dataset_id,
                "split": args.split,
                "primary": result.primary,
                "metrics": list(result.metrics),
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
