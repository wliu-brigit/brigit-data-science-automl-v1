from pathlib import Path

import pytest

from automl.project import ProjectConfig, Session
from automl.trial import paths

pytestmark = pytest.mark.unit


def _session(tmp_path: Path, *, namespace: str = "", dry_run: bool = False) -> Session:
    project_dir = tmp_path / "projects" / "demo"
    return Session(
        config=ProjectConfig(
            project_name="demo",
            repo_root=tmp_path,
            project_dir=project_dir,
            gcs_bucket="automl-test-bucket",
            gcs_prefix="automl-root",
            mlflow_tracking_uri=(tmp_path / "mlruns").as_uri(),
        ),
        experiment_id="route-exp",
        namespace=namespace,
        dry_run=dry_run,
    )


@pytest.mark.parametrize(
    ("namespace", "dry_run", "relative"),
    [
        ("", False, ("experiments", "demo", "route-exp")),
        ("", True, ("experiments", "dry_run", "demo", "route-exp")),
        ("qa", False, ("experiments", "qa", "demo", "route-exp")),
        ("qa", True, ("experiments", "qa", "dry_run", "demo", "route-exp")),
    ],
)
def test_route_root_pins_current_local_route_shape(tmp_path, namespace, dry_run, relative):
    active = _session(tmp_path, namespace=namespace, dry_run=dry_run)

    assert paths.route_root(active) == active.config.project_dir.joinpath(*relative)
