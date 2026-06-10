"""QA dry run of the full harness path with a CSV stand-in for Snowflake.

Generates the synthetic fixture, flips the project's source toggle
(NEOBANK_NCM_CSV — see config.py) so the recipe itself resolves to a
LocalCSVSource, materializes the dataset (GCS + MLflow registration), and
runs the baseline replication trial through the real runner — pre-fit
contract, fit, pyfunc logging, evaluation on the test split, artifacts,
manifest. Snowflake is the only seam not exercised.

Logs under a transient qa/ namespace so `automl project delete --scope qa`
sweeps it. Usage:

    uv run python projects/neobank_ncm/scripts/qa_local_run.py [--namespace qa/...]
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import date
from pathlib import Path

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

    csv_path = write_fixture_csv(
        Path(tempfile.mkdtemp(prefix="neobank_qa_")) / "base_table.csv"
    )
    print(f"fixture: {csv_path}")

    # the source toggle must be set before the project config loads
    os.environ["NEOBANK_NCM_CSV"] = str(csv_path)
    from automl.data import materialize
    from automl.project import use_project
    from automl.runner import run_trial

    session = use_project("neobank_ncm", namespace=args.namespace)

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
