"""Project scaffolding for the CLI setup flow."""

from __future__ import annotations

import re
import textwrap
from pathlib import Path


PROJECT_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


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
        "config.py": f'''
            """Typed project config for {project_name}."""

            from __future__ import annotations

            from pathlib import Path

            from automl.data import DataSpec, GCSParquetSource, LocalCSVSource, SnowflakeSource
            from automl.eval import Auc, EvalSpec
            from automl.project import (
                BinaryClassification,
                ModelRoute,
                ModelsConfig,
                ProjectConfig,
                RunConfig,
                Splits,
            )


            PROJECT_DIR = Path(__file__).resolve().parent

            TASK = BinaryClassification(target="<TBD_target_column>")

            source = SnowflakeSource(
                base_table="<TBD_base_table>",
                base_data_sql="data/queries/base_data.sql",
                training_data_sql="data/queries/training_data.sql",
            )

            DATA = DataSpec(
                source=source,
                metadata_cols=[],
                exclude_cols=[],
                dry_run_rows=10_001,
            )

            EVAL = EvalSpec(primary=Auc())

            RUN_CONFIG = RunConfig(
                experiment_id="TBD_experiment_id",
                splits=Splits(train=[(0, 80)], test=[(80, 100)]),
                models=ModelsConfig(
                    manager=ModelRoute("sonnet", "medium"),
                    proposer=ModelRoute("sonnet", "medium"),
                    coder=ModelRoute("sonnet", "medium"),
                ),
                per_trial_seconds=600,
            )

            PROJECT_CONFIG = ProjectConfig.partial(
                task=TASK,
                data_spec=DATA,
                eval_spec=EVAL,
                run_config=RUN_CONFIG,
            )
        ''',
        "PROJECT_INSTRUCTIONS.md": f"""
            # Project Instructions - {project_name}

            ## Goal

            Describe what "better" means for this project.

            ## Constraints

            - Do not read test data directly.

            ## Approaches to try

            - Start with a simple baseline.
        """,
        "data/queries/base_data.sql": """
            -- Snowflake base-data starter.
            CREATE OR REPLACE TABLE {database}.{schema}.{base_table} AS
            SELECT *
            FROM {database}.{schema}.<TBD_SOURCE_TABLE>;
        """,
        "data/queries/training_data.sql": """
            -- Snowflake training-data starter.
            SELECT
                *,
                MOD(ABS(HASH(<TBD_HASH_KEY_COLUMN>)), 100) AS SPLITID
            FROM {database}.{schema}.{base_table};
        """,
    }


__all__ = ["PROJECT_NAME_RE", "create_project"]
