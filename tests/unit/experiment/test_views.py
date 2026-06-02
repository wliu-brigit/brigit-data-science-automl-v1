import pytest

from automl.eval import Auc, EvalSpec
from automl.experiment import compare, create, leaderboard
from automl.experiment.views.queries import recent_failures, strategies_attempted
from automl.experiment.views.summary import build_summary, experiments
from automl.mlflow import client, experiment, tags, trial
from automl.project import ProjectConfig, Session

pytestmark = pytest.mark.unit


@pytest.fixture
def active(tmp_path):
    client.clear()
    config = ProjectConfig(
        project_name="home_credit",
        repo_root=tmp_path,
        project_dir=tmp_path / "projects" / "home_credit",
        eval_spec=EvalSpec(primary=Auc()),
        gcs_prefix="automl-root",
        mlflow_tracking_uri=(tmp_path / "mlruns").as_uri(),
    )
    session = Session(config=config, experiment_id="baseline")
    client.bind(
        tracking_uri=config.mlflow_tracking_uri,
        bucket="",
        gcs_prefix=config.gcs_prefix,
        project_name=config.project_name,
        experiment_id="baseline",
    )
    create(session=session)
    yield session
    client.clear()


def _run(slug, metric=None, status="FINISHED", strategy="baseline"):
    experiment.ensure()
    if status == "FAILED":
        with pytest.raises(RuntimeError):
            with trial.active(slug=slug, strategy=strategy) as run_id:
                raise RuntimeError("boom")
        return run_id
    with trial.active(slug=slug, strategy=strategy) as run_id:
        if metric is not None:
            trial.log_metric(run_id, "eval.test.auc", metric)
        return run_id


def test_leaderboard_ranks_scored_trials_and_counts_unscored(active):
    low = _run("low", 0.61, strategy="linear")
    high = _run("high", 0.91, strategy="tree")
    _run("unscored", None, strategy="manual")

    data = leaderboard(n=5, session=active)

    assert [row.run_id for row in data.rows] == [high, low]
    assert data.n_unscored == 1
    assert data.metric == "eval.test.auc"
    assert data.rows[0].primary_metric_name == "eval.test.auc"
    assert data.rows[0].primary_metric_value == pytest.approx(0.91)

    explicit = leaderboard(metric="eval.test.auc", n=5, session=active)
    assert [row.run_id for row in explicit.rows] == [high, low]
    assert explicit.metric == "eval.test.auc"


def test_leaderboard_n_unscored_is_not_limited_by_display_count(active):
    _run("low", 0.61)
    _run("mid", 0.71)
    high = _run("high", 0.91)
    _run("unscored", None)

    data = leaderboard(n=2, session=active)

    assert [row.run_id for row in data.rows] == [high, _run_id_for_slug("mid")]
    assert data.n_unscored == 1


def test_compare_returns_metric_deltas(active):
    left = _run("left", 0.7)
    right = _run("right", 0.85)

    result = compare([left, right], session=active)

    deltas = {item.metric: item for item in result.metric_deltas}
    assert deltas["eval.test.auc"].delta == pytest.approx(0.15)
    assert [run.run_id for run in result.runs] == [left, right]


def test_queries_and_summary_compose_over_seam(active):
    _run("failed", None, status="FAILED", strategy="tree")
    _run("scored", 0.8, strategy="linear")

    assert [row.strategy for row in recent_failures(session=active)] == ["tree"]
    assert strategies_attempted(session=active) == {"tree": 1, "linear": 1}

    summary = build_summary(session=active)
    assert summary["summary_kind"] == "experiment_summary"
    assert summary["trial_count"] == 2
    assert "learning_counts" not in summary
    assert experiments(session=active)[0]["experiment_id"] == "baseline"


def _run_id_for_slug(slug):
    return experiment.search_trials(f"tags.{tags.TRIAL_SLUG} = '{slug}'")[0].run_id
