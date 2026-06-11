"""Build the entity-graph store from a REGISTERED dataset (the v3 path).

Where graph_store_demo builds from the local sample parquet, THIS goes
through the harness: bind the project session, load the materialized dataset
from the MLflow registry + GCS, hand the frame to graph.build.build_store.
Loading a registered dataset needs MLflow + GCS only — Snowflake/VPN is NOT
required (that's only for materializing new datasets).

Preflight reports exactly what's missing instead of stack-tracing, because
this is the first script in the project that a fresh machine runs against
the real stack:

    uv run --group fraud python -m projects.fraud_anomaly_detection.analysis.graph_store_build
    uv run --group fraud python -m ... --dataset-id v2_2ac98b52 --out .../fraud_graph_v3.duckdb
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

DATASET_ID = "v2_2ac98b52"  # v3 base table (HANDOFF 2026-06-09): 2.41M rows x 115 cols
PROJECT = Path("projects/fraud_anomaly_detection")
DEFAULT_OUT = PROJECT / "data" / "graph" / "fraud_graph_v3.duckdb"
REQUIRED_ENV = ("MLFLOW_TRACKING_URI", "GCS_BUCKET", "GCP_PROJECT")


def _load_env() -> None:
    path = Path(".env")
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def _preflight() -> None:
    problems = []
    if not Path(".env").exists():
        problems.append(".env not found at the repo root (copy .env.example and fill"
                        " MLflow + GCS values; Snowflake fields not needed for this script)")
    missing = [key for key in REQUIRED_ENV if not os.environ.get(key)]
    if missing:
        problems.append(f"missing env: {', '.join(missing)}")
    adc = Path.home() / ".config/gcloud/application_default_credentials.json"
    if not adc.exists():
        problems.append("GCS ADC missing — run: gcloud auth application-default login")
    if problems:
        raise SystemExit("preflight failed:\n  - " + "\n  - ".join(problems))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset-id", default=DATASET_ID)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    _load_env()
    _preflight()

    # heavy imports after preflight so a misconfigured machine fails helpfully
    from automl.data.registry import load_dataset_by_id
    from automl.project.session import use_project

    from projects.fraud_anomaly_detection.graph.build import build_store

    print(f"binding project session + loading dataset {args.dataset_id}"
          " (MLflow registry -> GCS; minutes at v3 size)...")
    sess = use_project("fraud_anomaly_detection", dry_run=False)
    loaded = load_dataset_by_id(args.dataset_id, session=sess)
    df = loaded.df
    print(f"loaded {len(df):,} rows x {len(df.columns)} cols")

    summary = build_store(df, args.out, source_label=f"dataset:{args.dataset_id}")
    for key, val in summary.items():
        print(f"  {key:<22}{val:>12,}")
    print(f"\nstore built: {args.out}")
    print("next: uv run --group fraud python -m"
          " projects.fraud_anomaly_detection.analysis.graph_question_battery"
          f" --store {args.out}")


if __name__ == "__main__":
    main()
