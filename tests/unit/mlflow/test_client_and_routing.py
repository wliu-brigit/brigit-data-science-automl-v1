from __future__ import annotations

from datetime import UTC, datetime

import pytest

from automl.errors import StorageError
from automl.mlflow import _routing, client, project, routing

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def clear_mlflow_binding():
    client.clear()
    yield
    client.clear()


def test_bound_raises_storage_error_before_bind():
    with pytest.raises(StorageError, match="MLflow not bound"):
        client.bound()


def test_bind_stores_process_level_connection_state():
    client.bind(
        tracking_uri="file:///tmp/mlruns",
        bucket="automl-test-bucket",
        gcs_prefix="automl-root",
        project_name="home_credit",
        experiment_id="baseline",
        dry_run=True,
        namespace="qa",
    )

    bound = client.bound()

    assert bound.tracking_uri == "file:///tmp/mlruns"
    assert bound.bucket == "automl-test-bucket"
    assert bound.gcs_prefix == "automl-root"
    assert bound.project_name == "home_credit"
    assert bound.experiment_id == "baseline"
    assert bound.dry_run is True
    assert bound.namespace == "qa"


@pytest.mark.parametrize(
    ("namespace", "dry_run", "expected"),
    [
        ("", False, "home_credit"),
        ("", True, "dry_run/home_credit"),
        ("qa", False, "qa/home_credit"),
        ("qa", True, "qa/dry_run/home_credit"),
    ],
)
def test_project_route_uses_namespace_then_dry_run_then_project(
    namespace: str,
    dry_run: bool,
    expected: str,
):
    client.bind(
        tracking_uri="file:///tmp/mlruns",
        bucket="automl-test-bucket",
        gcs_prefix="automl-root",
        project_name="home_credit",
        experiment_id="baseline",
        dry_run=dry_run,
        namespace=namespace,
    )

    assert _routing.project_route() == expected


@pytest.mark.parametrize(
    ("namespace", "dry_run", "expected"),
    [
        ("", False, "home_credit/baseline"),
        ("", True, "dry_run/home_credit/baseline"),
        ("qa", False, "qa/home_credit/baseline"),
        ("qa", True, "qa/dry_run/home_credit/baseline"),
    ],
)
def test_experiment_route_uses_namespace_then_dry_run_then_project(
    namespace: str,
    dry_run: bool,
    expected: str,
):
    client.bind(
        tracking_uri="file:///tmp/mlruns",
        bucket="automl-test-bucket",
        gcs_prefix="automl-root",
        project_name="home_credit",
        experiment_id="baseline",
        dry_run=dry_run,
        namespace=namespace,
    )

    assert _routing.experiment_route() == expected


@pytest.mark.parametrize("field,value", [("project_name", "bad/name"), ("experiment_id", "")])
def test_route_components_must_be_non_empty_safe_segments(field: str, value: str):
    kwargs = {
        "tracking_uri": "file:///tmp/mlruns",
        "bucket": "automl-test-bucket",
        "gcs_prefix": "automl-root",
        "project_name": "home_credit",
        "experiment_id": "baseline",
    }
    kwargs[field] = value
    client.bind(**kwargs)

    with pytest.raises(ValueError):
        _routing.experiment_route()


def test_bucket_uri_for_run_bulk_uses_bound_bucket_prefix_and_trailing_slash():
    client.bind(
        tracking_uri="file:///tmp/mlruns",
        bucket="automl-test-bucket",
        gcs_prefix="automl-root/",
        project_name="home_credit",
        experiment_id="baseline",
        dry_run=True,
        namespace="qa",
    )

    uri = _routing.bucket_uri_for(kind="run_bulk", run_id="run-123")

    assert uri.startswith(
        "gs://automl-test-bucket/automl-root/qa/dry_run/home_credit/baseline/runs/"
    )
    assert uri.endswith("/run-123/")


@pytest.mark.parametrize(
    ("gcs_prefix", "route", "expected"),
    [
        (
            "automl-root/",
            "qa/dry_run/home_credit/baseline",
            "gs://automl-test-bucket/automl-root/qa/dry_run/home_credit/baseline/",
        ),
        (
            "",
            "qa/dry_run/home_credit/baseline",
            "gs://automl-test-bucket/qa/dry_run/home_credit/baseline/",
        ),
    ],
)
def test_gcs_uri_for_route_pins_bucket_prefix_route_layout(gcs_prefix, route, expected):
    assert (
        routing.gcs_uri_for_route(
            bucket="automl-test-bucket",
            gcs_prefix=gcs_prefix,
            route=route,
        )
        == expected
    )


def test_run_gcs_uri_for_route_owns_run_partition_layout():
    uri = routing.run_gcs_uri_for_route(
        bucket="automl-test-bucket",
        gcs_prefix="automl-root",
        route="qa/dry_run/home_credit/route-exp",
        run_id="run-1",
        now=datetime(2025, 12, 18, tzinfo=UTC),
    )

    assert (
        uri
        == "gs://automl-test-bucket/automl-root/qa/dry_run/home_credit/route-exp/runs/2025-12/run-1/"
    )


def test_project_route_prefix_uses_bound_bucket_prefix_without_experiment():
    client.bind(
        tracking_uri="file:///tmp/mlruns",
        bucket="automl-test-bucket",
        gcs_prefix="automl-root/",
        project_name="home_credit",
        experiment_id="baseline",
        dry_run=True,
        namespace="qa",
    )

    assert _routing.project_route_prefix() == "automl-root/qa/dry_run/home_credit"


@pytest.mark.parametrize(
    ("gcs_prefix", "namespace", "dry_run", "expected"),
    [
        ("automl-root/", "", False, "automl-root/demo/exp-1"),
        ("automl-root", "", True, "automl-root/dry_run/demo/exp-1"),
        ("automl-root", "qa", True, "automl-root/qa/dry_run/demo/exp-1"),
        ("", "qa", True, "qa/dry_run/demo/exp-1"),
    ],
)
def test_experiment_route_prefix_for_pins_gcs_prefix_layout(
    gcs_prefix: str,
    namespace: str,
    dry_run: bool,
    expected: str,
):
    assert (
        routing.experiment_route_prefix_for(
            gcs_prefix=gcs_prefix,
            project_name="demo",
            experiment_id="exp-1",
            namespace=namespace,
            dry_run=dry_run,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("gcs_prefix", "namespace", "dry_run", "expected"),
    [
        ("automl-root/", "", False, "automl-root"),
        ("automl-root", "", True, "automl-root/dry_run"),
        ("automl-root", "qa", True, "automl-root/qa/dry_run"),
        ("", "qa", True, "qa/dry_run"),
        ("", "", False, ""),
    ],
)
def test_namespace_route_prefix_for_preserves_data_dataset_prefix_layout(
    gcs_prefix: str,
    namespace: str,
    dry_run: bool,
    expected: str,
):
    assert (
        routing.namespace_route_prefix_for(
            gcs_prefix=gcs_prefix,
            namespace=namespace,
            dry_run=dry_run,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("namespace", "dry_run", "expected"),
    [
        ("qa", True, "qa/dry_run/demo/exp-1"),
        ("team/qa", True, "team/qa/dry_run/demo/exp-1"),
        ("team/qa", False, "team/qa/demo/exp-1"),
    ],
)
def test_route_build_parse_round_trip_with_namespace_and_dry_run(
    namespace: str,
    dry_run: bool,
    expected: str,
):
    route = routing.experiment_route_for(
        project_name="demo",
        experiment_id="exp-1",
        namespace=namespace,
        dry_run=dry_run,
    )

    assert route == expected
    assert routing.parse_experiment_route(route) == {
        "namespace": namespace,
        "dry_run": dry_run,
        "project_name": "demo",
        "experiment_id": "exp-1",
    }


def test_route_builder_rejects_namespace_dry_run_segment():
    with pytest.raises(ValueError, match="reserved dry_run"):
        routing.experiment_route_for(
            project_name="demo",
            experiment_id="exp-1",
            namespace="team/dry_run/qa",
        )


def test_dry_run_and_real_experiments_do_not_share_prefixes():
    real = routing.experiment_route_for(project_name="demo", experiment_id="exp")
    dry = routing.experiment_route_for(
        project_name="demo",
        experiment_id="exp",
        dry_run=True,
    )

    assert real == "demo/exp"
    assert dry == "dry_run/demo/exp"
    assert not real.startswith(dry)
    assert not dry.startswith(real + "/")


def test_experiment_local_path_preserves_experiments_route_shape(tmp_path):
    path = routing.experiment_local_path(
        tmp_path,
        project_name="demo",
        experiment_id="exp-1",
        namespace="qa",
        dry_run=True,
    )

    assert path == tmp_path / "experiments" / "qa" / "dry_run" / "demo" / "exp-1"


@pytest.mark.parametrize(
    "route",
    [
        "",
        "demo",
        "demo/",
        "/demo/exp",
        "demo//exp",
        "dry_run",
        "qa/dry_run/demo/exp/extra",
        "qa/bad name/exp",
    ],
)
def test_parse_experiment_route_rejects_malformed_routes(route: str):
    with pytest.raises(StorageError):
        routing.parse_experiment_route(route)


@pytest.mark.parametrize(
    ("namespace", "dry_run", "route_root"),
    [
        ("", False, "demo/"),
        ("", True, "dry_run/demo/"),
        ("qa", False, "qa/demo/"),
        ("qa", True, "qa/dry_run/demo/"),
    ],
)
def test_project_list_experiments_filters_under_current_route_root(
    tmp_path, namespace, dry_run, route_root
):
    client.bind(
        tracking_uri=(tmp_path / "mlruns").as_uri(),
        bucket="automl-test-bucket",
        gcs_prefix="automl-root",
        project_name="demo",
        experiment_id="route-exp",
        dry_run=dry_run,
        namespace=namespace,
    )
    mlflow_client = client.raw()
    mlflow_client.create_experiment(f"{route_root}route-exp")
    mlflow_client.create_experiment(f"{route_root}candidate")
    mlflow_client.create_experiment(f"{route_root}000_overview")
    mlflow_client.create_experiment(f"{route_root}route-exp/nested")
    mlflow_client.create_experiment(f"other/{route_root}route-exp")

    assert project.list_experiments() == ["candidate", "route-exp"]


@pytest.mark.parametrize(
    ("namespace", "dry_run", "expected"),
    [
        ("", False, "demo"),
        ("", True, "dry_run/demo"),
        ("qa", False, "qa/demo"),
        ("qa", True, "qa/dry_run/demo"),
    ],
)
def test_project_route_for_pins_current_demo_route_matrix(namespace, dry_run, expected):
    assert (
        _routing.project_route_for(
            project_name="demo",
            namespace=namespace,
            dry_run=dry_run,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("namespace", "dry_run", "expected"),
    [
        ("", False, "demo/route-exp"),
        ("", True, "dry_run/demo/route-exp"),
        ("qa", False, "qa/demo/route-exp"),
        ("qa", True, "qa/dry_run/demo/route-exp"),
    ],
)
def test_experiment_route_pins_current_demo_route_matrix(namespace, dry_run, expected):
    client.bind(
        tracking_uri="file:///tmp/mlruns",
        bucket="automl-test-bucket",
        gcs_prefix="automl-root",
        project_name="demo",
        experiment_id="route-exp",
        dry_run=dry_run,
        namespace=namespace,
    )

    assert _routing.experiment_route() == expected
