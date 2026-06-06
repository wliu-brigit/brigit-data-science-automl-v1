"""Typed project config for fraud_anomaly_detection.

This file is the project recipe: it describes the prediction problem once, in
the contracts the automl library defines. Four constants are required — TASK,
DATA, EVAL, RUN_CONFIG — assembled into PROJECT_CONFIG at the end. Edit this
file directly; every value marked TBD is yours to fill in. Then validate:

    uv run automl --project fraud_anomaly_detection validate project

Deeper references (paths relative to the repo root):

    agent-skills/references/setup/run-config.md          RUN_CONFIG fields
    agent-skills/references/setup/data-pipeline.md       DATA / DataSpec
    agent-skills/references/setup/evaluation-metric.md   EVAL / EvalSpec

A complete, filled-in recipe lives at projects/example_homecredit/config.py.
"""

from __future__ import annotations

from pathlib import Path

from automl.data import DataSpec, GCSParquetSource, LocalCSVSource, SnowflakeSource
from automl.eval import AveragePrecision, EvalSpec

from projects.fraud_anomaly_detection.eval.metrics import (
    BandReport,
    EarlyDefaultCapture,
    PrecisionRecallAtDepth,
)
from automl.project import (
    BinaryClassification,
    ModelRoute,
    ModelsConfig,
    Multiclass,
    ProjectConfig,
    Regression,
    RunConfig,
    Splits,
    Where,
)


PROJECT_DIR = Path(__file__).resolve().parent


# ── TASK — what the model predicts ──────────────────────────────────────────
# Names the target column and the kind of problem. Pick exactly one.

# is_fraud is derived in training_data.sql: heuristic_fraud_band == 'EXTREMELY_LIKELY'.
# Proxy label — the band is computed upstream from this table's own features, so
# supervised eval against it measures agreement with the heuristic, not ground truth.
TASK = BinaryClassification(target="is_fraud")  # positive_label=1 by default
# TASK = Regression(target="loan_amount")
# TASK = Multiclass(target="risk_tier", classes=("low", "medium", "high"))


# ── DATA — where the rows come from and how columns are treated ─────────────
# The source is the single entry point for this project's data. Snowflake is
# the canonical source for internal projects; the alternatives below are
# project-owned escape hatches for local files or pre-exported parquet.
# Deeper reference: agent-skills/references/setup/data-pipeline.md

source = SnowflakeSource(
    # Harness-owned table, materialized once over the upstream snapshot
    # (fraud_advance_feature_base, built out-of-band) with SPLIT_PCT injected.
    # The upstream table itself is never written to by the harness.
    base_table="fraud_advance_feature_base_automl",
    base_table_sql="data/queries/base_table.sql",      # the SELECT defining the base data
    training_data_sql="data/queries/training_data.sql",  # the SELECT pulling training rows
    unique_key="advance_id",  # one row per advance (upstream final dedup guarantees it)
    split_group_key="user_id",  # a user must never straddle train/test
)
# source = LocalCSVSource(csv_path=PROJECT_DIR / "data" / "my_data.csv", unique_key="ROW_ID")
# source = GCSParquetSource(gcs_uri="gs://bucket/path/data.parquet", unique_key="ROW_ID")

DATA = DataSpec(
    source=source,
    # Identifiers and raw timestamps: kept for traceability, never features.
    # (advance_id / user_id are auto-registered via unique_key / split_group_key.)
    metadata_cols=[
        "routing_number",
        "account_number",
        "bank_account_key",
        "plaid_account_id",
        "persistent_account_id",
        "institution_id",
        "network_label",
        "socure_id",
        "device_id",
        "ip_address",
        "signup_ip",
        "feature_as_of_ts",
        "origination_date",
        "identity_created_time",
        "plaid_account_created_at",
        "first_identity_created_on_account_asof",
        "latest_identity_created_on_account_asof",
        "previous_advance_on_account_ts",
        "socure_created_at",
    ],
    # Dropped entirely: post-outcome repayment/DPD fields (leakage), and the
    # heuristic score/band the is_fraud label is derived from (label leakage —
    # a model reading them reproduces the label for free).
    exclude_cols=[
        "heuristic_fraud_score",
        "heuristic_fraud_band",
        "label_repaid_current_snapshot",
        "expected_dpd45_date",
        "expected_dpd45_month",
        "days_past_due",
        "charge_off_timestamp",
        "pre_charge_off_timestamp",
        "gross_dpd45_amount",
        "label_gross_dpd7",
        "label_gross_dpd14",
        "label_gross_dpd21",
        "label_gross_dpd28",
        "label_gross_dpd35",
        "label_gross_dpd45",
        "label_mature_d7",
        "label_mature_d14",
        "label_mature_d21",
        "label_mature_d28",
        "label_mature_d35",
        "label_mature_d45",
    ],
    # ~100k dry-run rows: at the 98/2 pull composition (~0.33% positive) a
    # smaller sample has too few frauds to exercise the depth/band metrics.
    dry_run_rows=100_000,
    # pipeline_cls=MyPipeline,       # escape hatch: a project-owned DataPipeline subclass
    # null_drop_threshold=0.99,      # drop columns with more nulls than this fraction
    # constant_drop_threshold=1.0,   # drop columns at/above this constant-value fraction
)


# ── EVAL — how trials are scored ─────────────────────────────────────────────
# The primary metric is what the AutoML loop optimizes and compares trials by.
# Built-ins: Auc (ROC AUC), LogLoss. A custom metric is a project-owned class
# implementing the automl.eval.Metric protocol.
# Deeper reference: agent-skills/references/setup/evaluation-metric.md

# PR-AUC, not ROC-AUC: at low fraud prevalence ROC-AUC is dominated by easy
# negatives and reads misleadingly high; average precision tracks the
# precision/recall tradeoff on the positive class and is what the loop ranks
# trials by. The secondaries are project-owned (metrics.py): review-depth
# precision/recall (top 0.5%/1%/5%), the per-band agreement/discovery report,
# and the non-circular early-default (gross DPD45) capture on mature rows.
EVAL = EvalSpec(
    primary=AveragePrecision(),
    metrics=[PrecisionRecallAtDepth(), BandReport(), EarlyDefaultCapture()],
)


# ── RUN_CONFIG — how the AutoML loop runs ────────────────────────────────────
# experiment_id      names the MLflow experiment trials are logged under; pick
#                    a short, stable slug.
# splits             named row-criteria over the materialized dataset, written
#                    as Where(...) predicates (ops: == != < <= > >= .isin
#                    .notin .is_null .not_null, composed with & | ~). SPLIT_PCT
#                    is an ordinary column: a deterministic 0-99 hash bucket of
#                    each row's split_group_key, so the default below is an
#                    80/20 train/test split. Any column works — time-based:
#                    Splits(
#                        train=Where("application_date") < "2026-03-01",
#                        test=(Where("application_date") >= "2026-03-01")
#                             & (Where("SPLIT_PCT") < 50),
#                    )
#                    Rolling/backtesting windows are just a family of named
#                    splits (train_q1/test_q2, ...). Overlap is allowed and
#                    recorded, never policed.
# models             the three agent roles in the loop: the manager
#                    orchestrates, the proposer designs the next trial, the
#                    coder implements it. Each is ModelRoute(model, effort);
#                    effort is one of "low" / "medium" / "high".
# per_trial_seconds  hard time budget for a single trial.
# Deeper reference: agent-skills/references/setup/run-config.md

RUN_CONFIG = RunConfig(
    experiment_id="fraud_anomaly_v1",
    # 80/20: fit is unsupervised, so train-side metrics are label-honest too
    # (mild in-sample bias aside) and the density-style models (GMM, k-means)
    # want the training rows; ~20% still leaves hundreds of positives.
    splits=Splits(train=Where("SPLIT_PCT") < 80, test=Where("SPLIT_PCT") >= 80),
    models=ModelsConfig(
        manager=ModelRoute("opus", "high"),
        proposer=ModelRoute("opus", "high"),
        coder=ModelRoute("opus", "high"),
    ),
    # One hour: at ~100k dry-run rows the GMM trial already used 418s
    # (~120s of that is constant pyfunc-logging overhead); 600s left no
    # headroom for heavier models. No reason to run that tight.
    per_trial_seconds=3600,
)


# ── PROJECT_CONFIG — the assembled recipe ────────────────────────────────────
# automl.use_project("fraud_anomaly_detection") loads this object into the session;
# everything downstream reads it from there. Optional: pass
# required_transformers=[...] for preprocessing every trial must apply — see
# projects/example_homecredit/config.py for a worked example.

PROJECT_CONFIG = ProjectConfig.partial(
    task=TASK,
    data_spec=DATA,
    eval_spec=EVAL,
    run_config=RUN_CONFIG,
)
