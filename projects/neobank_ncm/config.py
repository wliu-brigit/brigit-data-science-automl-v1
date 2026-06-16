"""Typed project config for neobank_ncm.

Replication of the legacy Neobank NCM underwriting model v3 inside the AutoML
harness (legacy home: data-science/models/underwriting/neobank/new_user/v3.0).
Same population, same snapshot tables, same split design, same primary metric.

The base table wraps the three legacy sandbox tables (spine, risk features,
final synthetic scores) into one frozen snapshot — see
data/queries/base_table.sql. The reject-inference model is NOT replicated:
its output is consumed as the already-materialized
neobank_ncm_v3_synthetic_scores_final snapshot.

Validate with:

    uv run automl --project neobank_ncm validate project
"""

from __future__ import annotations

import os
from pathlib import Path

from automl.data import DataSpec, LocalCSVSource, SnowflakeSource
from automl.eval import Auc, EvalSpec
from automl.model import RequiredTransformer
from automl.project import (
    BinaryClassification,
    ModelRoute,
    ModelsConfig,
    ProjectConfig,
    RunConfig,
    Splits,
    Where,
)
from projects.neobank_ncm.model.preprocessing import BankInstitutionWOEEncoder


PROJECT_DIR = Path(__file__).resolve().parent


# ── TASK — what the model predicts ──────────────────────────────────────────
# went_dpd45 is ground truth on known rows (booked loans with mature outcome)
# and NULL on unknown rows (theoretical loans); unknown rows carry a
# synthetic_score soft label instead — see PROJECT_INSTRUCTIONS.md.

TASK = BinaryClassification(target="went_dpd45")


# ── DATA — where the rows come from and how columns are treated ─────────────
# The legacy v3 snapshot tables live read-only in sandbox_hyong; per the
# existing-table convention (README.md) the harness materializes its own copy
# from the wrapping SELECT and never writes the originals. The copy lands in
# the session schema — run with:
#
#     SNOWFLAKE_DATABASE=brigit_data_science  SNOWFLAKE_SCHEMA=sandbox_wliu
#
# Upstream DDL for the legacy tables is kept as reference-only provenance in
# data/queries/upstream_*.sql.

# Source toggle (project-owned escape hatch, see package CLAUDE.md): the real
# recipe is the SnowflakeSource below. Until VPN day, set
#
#     NEOBANK_NCM_CSV=/path/to/base_table.csv
#
# to swap in a LocalCSVSource at config load — offline QA and loop dry-runs
# against the synthetic fixture (tests/fixtures.write_fixture_csv) run the
# whole harness without Snowflake. Unset the variable to point back at the
# warehouse; the per-experiment pinned dataset keeps results comparable
# (re-materializing under a new source is an explicit --refresh-data step).
_CSV_OVERRIDE = os.environ.get("NEOBANK_NCM_CSV", "").strip()

if _CSV_OVERRIDE:
    source: LocalCSVSource | SnowflakeSource = LocalCSVSource(
        csv_path=Path(_CSV_OVERRIDE),
        unique_key="entity_id",
        split_group_key="user_id",
    )
else:
    source = SnowflakeSource(
        base_table="neobank_ncm_v3_replicate_base",
        base_table_sql="data/queries/base_table.sql",
        training_data_sql="data/queries/training_data.sql",
        unique_key="entity_id",
        split_group_key="user_id",
    )

DATA = DataSpec(
    source=source,
    metadata_cols=[
        # identifiers
        "entity_id",
        "user_id",
        "sa_id",
        # split machinery (referenced by RUN_CONFIG.splits)
        "split",
        "is_known",
        "origination_date",
        # legacy server-side downsample replay (sampling machinery, never a feature)
        "unknown_train_hash_rank",
        # label surrogate for the unknown group — never a feature
        "synthetic_score",
        # loan/account context kept for analysis, not underwriting features
        # (legacy v3 feature list excludes them; amount feeds the RI model only)
        "original_due_date",
        "amount",
        "valid_from",
        "valid_to",
    ],
    # The legacy experiment's "Pass 0 — drop questionable features": removed
    # from the candidate set before any model fitting, with documented
    # reasons. Enforced here so no trial can rediscover them.
    exclude_cols=[
        "linkedcardstatus",          # funnel-stage artifact, 99% missing in unknown group
        "signupsourcetype",          # fair-lending concern
        "totalreturnedpayments30d",  # past-payment signal (unknown group lacks it)
        "mostrecentpaymentreturntype",
        "dayssincelastsuccessfulpayment",
        "dayssincelastpaymentsettled",
        "paymentlastresult",
        "previoussubscriptionresult",
        "dayssincelastplaidbalancecheck",  # 100% missing
        "plaidavailablebalancelag1",       # 100% missing
        "plaiddaydiffbetweenlasttwobalancechecks",  # 100% missing
        "highestincomepayerid",      # 65% missing in unknown; very high cardinality
    ],
    dry_run_rows=10_001,
)


# ── EVAL — how trials are scored ─────────────────────────────────────────────
# Known-only AUC is the sole decision metric (legacy decision: all metrics on
# synthetic-labeled rows are contaminated by RI label quality). The eval split
# below is already restricted to is_known rows, so the built-in Auc applies.

EVAL = EvalSpec(primary=Auc())


# ── RUN_CONFIG — how the AutoML loop runs ────────────────────────────────────
# Splits mirror the legacy design, with the spine's split column as the
# single source of truth (train = 2025, oot = Jan–Feb 2026) and the legacy
# time boundary within 2025 (origination_date < 2025-11-01):
#
#   train        Jan–Oct 2025, known + unknown — what trials fit on in the
#                loop; trials may sub-split it for their own tuning/CV
#   train_known  the known-only view of train. The runner's automatic
#                train-side eval dies (silently) on train's NULL-target
#                unknown rows, so the legacy "train known-only" diagnostic
#                is computed on demand against this split instead
#                (scripts/reeval/evaluate_split.py --split train_known).
#   test         Nov–Dec 2025, known-only — the in-loop leaderboard metric
#                (disjoint from train: the loop fits train_split, scores
#                eval_split)
#   oot    Jan–Feb 2026, known-only      — the final test in all cases; NOT
#          used by the loop. Defined here so its meaning is recorded with the
#          dataset; the post-AutoML re-evaluation resolves it by name,
#          touched once (legacy Phase 5). The final retrain before it
#          (legacy Phase 4) uses full 2025 = the train + test windows.
#
# SPLIT_PCT (hash of user_id) is injected by the harness but unused: the
# legacy split is temporal and pre-defined in the spine.

SPLITS = Splits(
    train=(Where("split") == "train") & (Where("origination_date") < "2025-11-01"),
    train_known=(
        (Where("split") == "train")
        & (Where("origination_date") < "2025-11-01")
        & (Where("is_known") == True)  # noqa: E712
    ),
    test=(
        (Where("split") == "train")
        & (Where("origination_date") >= "2025-11-01")
        & (Where("is_known") == True)  # noqa: E712 — Where overloads ==
    ),
    oot=(Where("split") == "oot") & (Where("is_known") == True),  # noqa: E712
)

RUN_CONFIG = RunConfig(
    experiment_id="neobank_ncm_v3_replicate",
    splits=SPLITS,
    models=ModelsConfig(
        manager=ModelRoute("opus", "high"),
        proposer=ModelRoute("opus", "high"),
        coder=ModelRoute("opus", "high"),
    ),
    per_trial_seconds=600,
    # Full-data serving validation loads the model from GCS + benchmarks it; the
    # 300s core default has been tight on this dataset. Give it headroom so a
    # slow-but-healthy validation isn't killed.
    serving_validation_seconds=600,
    # The dataset is already materialized to GCS, and the AutoML loop reads from
    # GCS (only `data materialize --refresh-source` touches Snowflake). Skip the
    # live SELECT 1 in the preflight so the loop validates and runs off-VPN —
    # GCS is throttled on the VPN anyway. Materialize still fails loudly if
    # Snowflake is unreachable; offline checks (env vars, SQL files) still run.
    skip_snowflake_live_check=True,
    train_split="train",
    eval_split="test",
)


# ── REQUIRED_TRANSFORMERS — preprocessing every trial must apply ─────────────
# The legacy model never feeds the raw bankinstitution string to the
# estimator: it enters only as its WoE encoding (fit on labeled/known train
# rows; sparse and unseen banks inherit CHIME's value via OTHER).

REQUIRED_TRANSFORMERS = [
    RequiredTransformer(
        name="neobank_bankinstitution_woe",
        transformer=BankInstitutionWOEEncoder(),
        input_cols=["bankinstitution"],
    )
]


# ── PROJECT_CONFIG — the assembled recipe ────────────────────────────────────

PROJECT_CONFIG = ProjectConfig.partial(
    task=TASK,
    data_spec=DATA,
    eval_spec=EVAL,
    run_config=RUN_CONFIG,
    required_transformers=REQUIRED_TRANSFORMERS,
)
