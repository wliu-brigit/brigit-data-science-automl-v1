"""Project scaffolding for the CLI setup flow."""

from __future__ import annotations

import re
import textwrap
from pathlib import Path


PROJECT_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")

# The generated config.py is deliberately self-teaching: each recipe constant
# gets a commented section listing the common alternatives inline, so a new
# user never has to guess what to import or what a slot means. Two rules keep
# it honest:
#
# - The literal string ``TBD_`` appears ONLY in the placeholder values the
#   user must fill. The placeholder check (automl/project/checks.py) flags any
#   file containing ``TBD_``, so a comment spelling it would fail validation
#   forever. Commented-out alternatives use illustrative values instead.
# - The unfilled file must import cleanly, and the active code must construct
#   real library objects — tests/unit/project/test_metadata_and_scaffold.py
#   executes the scaffold against the library to pin both.
_CONFIG_TEMPLATE = '''
"""Typed project config for {project_name}.

This file is the project recipe: it describes the prediction problem once, in
the contracts the automl library defines. Four constants are required — TASK,
DATA, EVAL, RUN_CONFIG — assembled into PROJECT_CONFIG at the end. Edit this
file directly; every value marked TBD is yours to fill in. Then validate:

    uv run automl --project {project_name} validate project

Deeper references (paths relative to the repo root):

    agent-skills/references/setup/run-config.md          RUN_CONFIG fields
    agent-skills/references/setup/data-pipeline.md       DATA / DataSpec
    agent-skills/references/setup/evaluation-metric.md   EVAL / EvalSpec

A complete, filled-in recipe lives at projects/example_homecredit/config.py.
"""

from __future__ import annotations

from pathlib import Path

from automl.data import DataSpec, GCSParquetSource, LocalCSVSource, SnowflakeSource
from automl.eval import Auc, EvalSpec, LogLoss
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

TASK = BinaryClassification(target="<TBD_target_column>")  # positive_label=1 by default
# TASK = Regression(target="loan_amount")
# TASK = Multiclass(target="risk_tier", classes=("low", "medium", "high"))


# ── DATA — where the rows come from and how columns are treated ─────────────
# The source is the single entry point for this project's data. Snowflake is
# the canonical source for internal projects; the alternatives below are
# project-owned escape hatches for local files or pre-exported parquet.
# Deeper reference: agent-skills/references/setup/data-pipeline.md

source = SnowflakeSource(
    base_table="<TBD_base_table>",  # name only; lands at {{database}}.{{schema}}.{{base_table}}
    base_table_sql="data/queries/base_table.sql",      # the SELECT defining the base data
    training_data_sql="data/queries/training_data.sql",  # the SELECT pulling training rows
    unique_key="<TBD_unique_key>",  # stable row identifier; tuple for composite keys
    # split_group_key="USER_ID",    # declare only when splits must group by a coarser key
)
# source = LocalCSVSource(csv_path=PROJECT_DIR / "data" / "my_data.csv", unique_key="ROW_ID")
# source = GCSParquetSource(gcs_uri="gs://bucket/path/data.parquet", unique_key="ROW_ID")

DATA = DataSpec(
    source=source,
    metadata_cols=[],  # identifiers kept but never used as features (e.g. the unique key)
    exclude_cols=[],  # columns dropped entirely (leakage, post-outcome fields)
    dry_run_rows=10_001,  # row cap when running with --dry-run
    # pipeline_cls=MyPipeline,       # escape hatch: a project-owned DataPipeline subclass
    # null_drop_threshold=0.99,      # drop columns with more nulls than this fraction
    # constant_drop_threshold=1.0,   # drop columns at/above this constant-value fraction
)


# ── EVAL — how trials are scored ─────────────────────────────────────────────
# The primary metric is what the AutoML loop optimizes and compares trials by.
# Built-ins: Auc (ROC AUC), LogLoss. A custom metric is a project-owned class
# implementing the automl.eval.Metric protocol.
# Deeper reference: agent-skills/references/setup/evaluation-metric.md

EVAL = EvalSpec(primary=Auc())
# EVAL = EvalSpec(primary=LogLoss())


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
    experiment_id="TBD_experiment_id",
    splits=Splits(train=Where("SPLIT_PCT") < 80, test=Where("SPLIT_PCT") >= 80),
    models=ModelsConfig(
        manager=ModelRoute("sonnet", "medium"),
        proposer=ModelRoute("sonnet", "medium"),
        coder=ModelRoute("sonnet", "medium"),
    ),
    per_trial_seconds=600,
)


# ── PROJECT_CONFIG — the assembled recipe ────────────────────────────────────
# automl.use_project("{project_name}") loads this object into the session;
# everything downstream reads it from there. Optional: pass
# required_transformers=[...] for preprocessing every trial must apply — see
# projects/example_homecredit/config.py for a worked example.

PROJECT_CONFIG = ProjectConfig.partial(
    task=TASK,
    data_spec=DATA,
    eval_spec=EVAL,
    run_config=RUN_CONFIG,
)
'''


def create_project(
    project_name: str,
    *,
    project_root: Path,
    template: str = "snowflake",
) -> dict[str, object]:
    if not PROJECT_NAME_RE.fullmatch(project_name):
        raise ValueError("project name must be lower snake_case and start with a letter")
    if template != "snowflake":
        raise ValueError(f"unsupported project template: {template}")

    root = Path(project_root).resolve()
    project_dir = root / "projects" / project_name
    if project_dir.exists():
        raise FileExistsError(f"project already exists: {project_dir}")

    (root / "projects").mkdir(parents=True, exist_ok=True)
    (root / "projects" / "__init__.py").touch(exist_ok=True)

    created: list[str] = []
    for relative, content in _snowflake_templates(project_name).items():
        path = project_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8")
        created.append(str(path.relative_to(root)))

    return {
        "project": project_name,
        "template": template,
        "project_dir": str(project_dir),
        "created": created,
        "next_command": f"uv run automl --project {project_name} validate project",
    }


def _snowflake_templates(project_name: str) -> dict[str, str]:
    return {
        "__init__.py": "",
        "config.py": _CONFIG_TEMPLATE.format(project_name=project_name),
        "PROJECT_INSTRUCTIONS.md": f"""
            # Project Instructions - {project_name}

            ## Goal

            Describe what "better" means for this project.

            ## Constraints

            - Do not read test data directly.

            ## Approaches to try

            - Start with a simple baseline.
        """,
        "data/queries/base_table.sql": """
            -- The SELECT that defines your base data: joins, CTEs, filters, feature SQL.
            -- The harness wraps it in CREATE OR REPLACE TABLE and injects SPLIT_PCT from
            -- split_group_key — do not emit SPLIT_PCT yourself.
            SELECT *
            FROM {database}.{schema}.<TBD_SOURCE_TABLE>
        """,
        "data/queries/training_data.sql": """
            -- The SELECT that pulls training rows from the base table.
            -- SPLIT_PCT flows through; keep it in the projection.
            SELECT *
            FROM {database}.{schema}.{base_table}
        """,
    }


__all__ = ["PROJECT_NAME_RE", "create_project"]
