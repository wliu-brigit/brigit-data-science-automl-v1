from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tests.e2e._gates import requires_live_e2e

from automl.cli import main as cli_main
from automl.experiment import delete as delete_experiment
from automl.mlflow import client as mlflow_client
from automl.project import clear_session, use_project
from automl.utils.io import gcs

pytestmark = [pytest.mark.e2e, pytest.mark.qa]


@requires_live_e2e("CLI route isolation")
def test_cli_route_isolates_dry_run_and_namespace_universes(capsys):
    repo_root = Path(__file__).resolve().parents[2]
    stamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    namespace = f"qa-cli-route-isolation-{stamp}"
    experiment_id = f"cli-route-isolation-{stamp}"
    project = "example_homecredit"
    real_route = f"{project}/{experiment_id}"
    dry_route = f"dry_run/{project}/{experiment_id}"
    qa_route = f"{namespace}/{project}/{experiment_id}"
    qa_dry_route = f"{namespace}/dry_run/{project}/{experiment_id}"

    def run_cli(*args: str) -> dict:
        assert cli_main(["--project", project, "--project-root", str(repo_root), *args]) == 0
        output = capsys.readouterr().out
        return _json_from_cli_output(output) if output.strip() else {}

    def prefix_for(route: str) -> str:
        active = use_project(project, repo_root=repo_root, experiment_id=experiment_id)
        try:
            return f"gs://{active.config.gcs_bucket}/{active.config.gcs_prefix}/{route}/"
        finally:
            clear_session()

    real_prefix = prefix_for(real_route)
    dry_prefix = prefix_for(dry_route)
    qa_prefix = prefix_for(qa_route)

    try:
        run_cli("--experiment-id", experiment_id, "data", "materialize")
        real_run = run_cli("--experiment-id", experiment_id, "trial", "run", project)
        assert real_run["status"] == "FINISHED"
        assert gcs.list_blob_names(real_prefix)

        run_cli(
            "--dry-run",
            "--experiment-id",
            experiment_id,
            "data",
            "materialize",
        )
        dry_run = run_cli(
            "--dry-run",
            "--experiment-id",
            experiment_id,
            "trial",
            "run",
            project,
        )
        assert dry_run["status"] == "FINISHED"
        assert gcs.list_blob_names(dry_prefix)

        dry_delete = run_cli(
            "--dry-run",
            "--experiment-id",
            experiment_id,
            "experiment",
            "delete",
            experiment_id,
            "--apply",
        )
        assert dry_delete["applied"] is True
        assert gcs.list_blob_names(dry_prefix) == []
        assert gcs.list_blob_names(real_prefix)
        real_experiment = mlflow_client.raw().get_experiment_by_name(real_route)
        assert real_experiment is not None
        assert real_experiment.lifecycle_stage == "active"

        run_cli(
            "--namespace",
            namespace,
            "--experiment-id",
            experiment_id,
            "data",
            "materialize",
        )
        qa_run = run_cli(
            "--namespace",
            namespace,
            "--experiment-id",
            experiment_id,
            "trial",
            "run",
            project,
        )
        assert qa_run["status"] == "FINISHED"
        assert gcs.list_blob_names(qa_prefix)

        qa_dry_preview = run_cli(
            "--namespace",
            namespace,
            "--dry-run",
            "--experiment-id",
            experiment_id,
            "experiment",
            "delete",
            experiment_id,
        )
        assert qa_dry_preview["plan"]["gcs_prefix_patterns"] == [prefix_for(qa_dry_route)]

        qa_delete = run_cli(
            "--namespace",
            namespace,
            "--experiment-id",
            experiment_id,
            "experiment",
            "delete",
            experiment_id,
            "--apply",
        )
        assert qa_delete["applied"] is True
        assert gcs.list_blob_names(qa_prefix) == []
        assert gcs.list_blob_names(real_prefix)
        qa_experiment = mlflow_client.raw().get_experiment_by_name(qa_route)
        assert qa_experiment is not None
        assert qa_experiment.lifecycle_stage == "deleted"
        assert mlflow_client.raw().get_experiment_by_name(real_route).lifecycle_stage == "active"
    finally:
        for kwargs in (
            {"dry_run": True},
            {"namespace": namespace},
            {},
        ):
            active = use_project(
                project,
                repo_root=repo_root,
                experiment_id=experiment_id,
                **kwargs,
            )
            try:
                delete_experiment(experiment_id, apply=True, session=active)
            finally:
                clear_session()


def _json_from_cli_output(output: str) -> dict:
    decoder = json.JSONDecoder()
    for index, char in enumerate(output):
        if char != "{":
            continue
        try:
            payload, end = decoder.raw_decode(output[index:])
        except json.JSONDecodeError:
            continue
        if output[index + end :].strip():
            continue
        if not isinstance(payload, dict):
            raise AssertionError(
                f"expected JSON object from CLI output, got {type(payload).__name__}"
            )
        return payload
    raise AssertionError(f"no JSON object found in CLI output:\n{output}")
