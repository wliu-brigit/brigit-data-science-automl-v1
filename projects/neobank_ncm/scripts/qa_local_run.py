"""QA dry run of the full harness path with a CSV stand-in for Snowflake.

Generates the synthetic fixture, swaps the recipe's source to LocalCSVSource
at the session level (config.py is untouched), materializes the dataset
(GCS + MLflow registration), and runs the baseline replication trial through
the real runner — pre-fit contract, fit, pyfunc logging, evaluation on the
test split, artifacts, manifest. Snowflake is the only seam not exercised.

Logs under a transient qa/ namespace so `automl project delete --scope qa`
sweeps it. Usage:

    uv run python projects/neobank_ncm/scripts/qa_local_run.py [--namespace qa/...]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import tempfile
from datetime import date
from pathlib import Path

from automl.data import DataSpec, LocalCSVSource, materialize
from automl.project import update_session, use_project
from automl.runner import run_trial
from projects.neobank_ncm.tests.fixtures import write_fixture_csv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--namespace",
        default=f"qa/neobank-csv-dryrun-{date.today():%Y%m%d}",
        help="MLflow namespace; must start with qa/ (transient, sweepable)",
    )
    args = parser.parse_args()
    if not args.namespace.startswith("qa/"):
        raise SystemExit("namespace must start with qa/ — this is a transient QA run")

    session = use_project("neobank_ncm", namespace=args.namespace)

    csv_path = write_fixture_csv(
        Path(tempfile.mkdtemp(prefix="neobank_qa_")) / "base_table.csv"
    )
    print(f"fixture: {csv_path}")

    spec = DataSpec(
        source=LocalCSVSource(
            csv_path=csv_path, unique_key="entity_id", split_group_key="user_id"
        ),
        metadata_cols=session.config.data_spec.metadata_cols,
        exclude_cols=session.config.data_spec.exclude_cols,
        dry_run_rows=session.config.data_spec.dry_run_rows,
    )
    session = update_session(config=dataclasses.replace(session.config, data_spec=spec))

    dataset = materialize(refresh_data=True, include_rows=False, session=session)
    print(f"dataset materialized: {dataset.id}")

    result = run_trial("neobank_ncm", session=session)
    print(
        json.dumps(
            {
                "status": result.status,
                "run_id": result.run_id,
                "trial_id": result.trial_id,
                "namespace": args.namespace,
                "experiment_url": session.mlflow_experiment_url(),
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
