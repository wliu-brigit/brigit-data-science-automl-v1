"""Live SnowflakeSource materialize, against a throwaway dev_ project.

Written in step 3; the live run is deferred to the tail-end pass after step 4
(plans ledger, tail-end activities). The dev_ source table and its columns are
wendao-designated at run time through the AUTOML_SNOWFLAKE_E2E_* variables —
never hardcoded to a production table. Each run bootstraps its own
uniquely-named DEV_AUTOML_E2E_* base table (never touching pre-existing
tables); drop them when done.
"""

from __future__ import annotations

import os
import textwrap
import uuid
from pathlib import Path

import pytest

from tests.e2e._gates import LIVE_E2E_ENV, SERVICE_ENV

SNOWFLAKE_ENV = (
    "SNOWFLAKE_ACCOUNT",
    "SNOWFLAKE_USER",
    "SNOWFLAKE_PASSWORD",
    "SNOWFLAKE_DATABASE",
    "SNOWFLAKE_SCHEMA",
)
TABLE_ENV = (
    "AUTOML_SNOWFLAKE_E2E_SOURCE_TABLE",  # tiny dev_ table the base SELECT reads
    "AUTOML_SNOWFLAKE_E2E_TARGET",  # its binary target column
    "AUTOML_SNOWFLAKE_E2E_UNIQUE_KEY",  # its unique-key column
)
_required = (LIVE_E2E_ENV, *SERVICE_ENV, *SNOWFLAKE_ENV, *TABLE_ENV)

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.qa,
    pytest.mark.skipif(
        any(not os.environ.get(name) for name in _required),
        reason=f"snowflake e2e requires {', '.join(_required)}",
    ),
]

_PROJECT_NAME = "dev_snowflake_e2e"

_CONFIG_TEMPLATE = """\
from automl.data import DataSpec, SnowflakeSource
from automl.eval import Auc, EvalSpec
from automl.project import (
    BinaryClassification,
    ModelRoute,
    ModelsConfig,
    ProjectConfig,
    RunConfig,
    Splits,
    Where,
)

TASK = BinaryClassification(target="{target}")
DATA = DataSpec(
    source=SnowflakeSource(
        base_table="{base_table}",
        base_table_sql="data/queries/base_table.sql",
        training_data_sql="data/queries/training_data.sql",
        unique_key="{unique_key}",
    ),
    metadata_cols=("{unique_key}",),
)
EVAL = EvalSpec(primary=Auc())
RUN_CONFIG = RunConfig(
    experiment_id="{experiment_id}",
    splits=Splits(train=Where("SPLIT_PCT") < 80, test=Where("SPLIT_PCT") >= 80),
    models=ModelsConfig(
        manager=ModelRoute("sonnet", "medium"),
        proposer=ModelRoute("sonnet", "medium"),
        coder=ModelRoute("sonnet", "medium"),
    ),
    per_trial_seconds=120,
)
PROJECT_CONFIG = ProjectConfig.partial(
    task=TASK,
    data_spec=DATA,
    eval_spec=EVAL,
    run_config=RUN_CONFIG,
)
"""


def _write_project(repo_root: Path, *, base_table: str, experiment_id: str) -> str:
    project_dir = repo_root / "projects" / _PROJECT_NAME
    queries = project_dir / "data" / "queries"
    queries.mkdir(parents=True)
    (repo_root / "projects" / "__init__.py").touch()
    (project_dir / "__init__.py").touch()
    (project_dir / "config.py").write_text(
        _CONFIG_TEMPLATE.format(
            target=os.environ["AUTOML_SNOWFLAKE_E2E_TARGET"],
            base_table=base_table,
            unique_key=os.environ["AUTOML_SNOWFLAKE_E2E_UNIQUE_KEY"],
            experiment_id=experiment_id,
        ),
        encoding="utf-8",
    )
    source_table = os.environ["AUTOML_SNOWFLAKE_E2E_SOURCE_TABLE"]
    (queries / "base_table.sql").write_text(
        textwrap.dedent(
            f"""\
            -- e2e: the designated dev_ table is the whole base data.
            SELECT *
            FROM {{database}}.{{schema}}.{source_table}
            """
        ),
        encoding="utf-8",
    )
    (queries / "training_data.sql").write_text(
        "SELECT *\nFROM {database}.{schema}.{base_table}\n", encoding="utf-8"
    )
    return _PROJECT_NAME


def test_materialize_bootstraps_pulls_and_attaches(tmp_path):
    from automl.data import materialize
    from automl.project import clear_session, use_project

    stamp = uuid.uuid4().hex[:8]
    base_table = f"DEV_AUTOML_E2E_BASE_{stamp.upper()}"  # fresh per run; drop after
    experiment_id = f"dev-snowflake-e2e-{stamp}"
    name = _write_project(tmp_path, base_table=base_table, experiment_id=experiment_id)

    active = use_project(name, repo_root=tmp_path)
    try:
        # 1. First materialize: bootstraps the base table, mints v1, SPLIT_PCT valid.
        first = materialize(session=active)
        assert first.dataset.id.startswith("v1_")
        assert first.df["SPLIT_PCT"].between(0, 99).all()
        assert first.dataset.source_identity["split"] == "sql"

        # 2. Default call attaches to the pinned v1 (no re-pull, same id).
        again = materialize(session=active)
        assert again.dataset.id == first.dataset.id

        # 3. Explicit refresh re-pulls; unchanged content dedups back to v1.
        refreshed = materialize(session=active, refresh_data=True)
        assert refreshed.dataset.id == first.dataset.id
    finally:
        clear_session()
