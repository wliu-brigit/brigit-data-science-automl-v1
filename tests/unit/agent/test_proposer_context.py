from pathlib import Path

import pytest
from sklearn.preprocessing import FunctionTransformer

from automl.data.dataset import ComponentHashes, Dataset, DatasetIndex
from automl.data.profile import Profile
from automl.eval import Auc, EvalSpec
from automl.experiment.store import ExperimentOverview
from automl.experiment.views.types import LeaderboardData
from automl.model import RequiredTransformer
from automl.project import (
    BinaryClassification,
    ModelRoute,
    ModelsConfig,
    ProjectConfig,
    RunConfig,
    Session,
    Splits,
)
from automl.trial.types import TrialStatus, TrialSummary

pytestmark = pytest.mark.unit


def _session(tmp_path: Path) -> Session:
    instructions_path = tmp_path / "projects" / "demo" / "PROJECT_INSTRUCTIONS.md"
    instructions_path.parent.mkdir(parents=True)
    instructions_path.write_text("Prefer fast single-model trials.\n", encoding="utf-8")
    route = ModelRoute("sonnet", "medium")
    return Session(
        config=ProjectConfig(
            project_name="demo",
            repo_root=tmp_path,
            project_dir=instructions_path.parent,
            config_path=instructions_path.parent / "config.py",
            instructions_path=instructions_path,
            task=BinaryClassification(target="TARGET"),
            eval_spec=EvalSpec(primary=Auc()),
            run_config=RunConfig(
                experiment_id="exp_b",
                splits=Splits({"train": ((0, 80),), "test": ((80, 100),)}),
                models=ModelsConfig(manager=route, proposer=route, coder=route),
                per_trial_seconds=120,
            ),
            required_transformers=[
                RequiredTransformer(
                    name="demo_required",
                    transformer=FunctionTransformer(),
                    input_cols=["category"],
                )
            ],
            gcs_bucket="bucket",
            gcs_prefix="root",
        ),
        dry_run=True,
        namespace="qa",
    )


def _dataset() -> Dataset:
    return Dataset(
        id="dataset-v1",
        identity_hash="abc",
        component_hashes=ComponentHashes(
            source_identity="s",
            feature_registry="f",
            data_content="d",
            schema="c",
        ),
        gcs_bucket="bucket",
        project_name="demo",
        created_at="2026-05-01T00:00:00Z",
        source_identity={"kind": "fixture"},
        n_rows=25,
        n_columns=5,
        target_column="target",
        split_pct_col="SPLIT_PCT",
        unique_key=("row_id",),
        split_group_key=("row_id",),
        gcs_prefix="root",
    )


def test_gather_proposer_context_returns_a5_packet(monkeypatch, tmp_path):
    from automl.agent.proposer_context import gather_proposer_context

    active = _session(tmp_path)
    dataset = _dataset()
    human = TrialSummary(
        run_id="human-run",
        slug="human_baseline",
        strategy="baseline",
        status=TrialStatus.FINISHED,
        training_origin="human",
    )
    failure = TrialSummary(
        run_id="failed-run",
        slug="bad_trial",
        strategy="new_model",
        status=TrialStatus.FAILED,
    )
    agent_row = TrialSummary(
        run_id="agent-run",
        slug="agent_trial",
        strategy="feature_engineering",
        status=TrialStatus.FINISHED,
        primary_metric_name="auc",
        primary_metric_value=0.77,
    )

    def fake_leaderboard(**kwargs):
        rows = (human,) if kwargs.get("training_origin") == "human" else (agent_row,)
        return LeaderboardData(
            metric=kwargs["metric"],
            experiment_id="exp_b",
            rows=rows,
            n_unscored=0 if kwargs.get("training_origin") == "human" else 2,
        )

    monkeypatch.setattr("automl.agent.proposer_context.leaderboard", fake_leaderboard)
    monkeypatch.setattr(
        "automl.agent.proposer_context.recent_failures",
        lambda **kwargs: [failure],
    )
    monkeypatch.setattr(
        "automl.agent.proposer_context.strategies_attempted",
        lambda **kwargs: {"baseline": 1, "feature_engineering": 1},
    )
    monkeypatch.setattr(
        "automl.agent.proposer_context.list_datasets",
        lambda **kwargs: DatasetIndex(datasets=(dataset,), active_dataset_id="dataset-v1"),
    )
    monkeypatch.setattr(
        "automl.agent.proposer_context.get_profile",
        lambda **kwargs: Profile(
            dataset_id="dataset-v1",
            target_column="target",
            data_card_uri="runs:/overview/data_card.json",
            data_observations_uri="runs:/overview/data_observations.json",
            profile_manifest_uri="runs:/overview/profile.json",
        ),
    )
    monkeypatch.setattr(
        "automl.agent.proposer_context.allowed_dependencies",
        lambda session=None: ["pandas", "numpy", "scikit-learn"],
    )
    monkeypatch.setattr(
        "automl.agent.proposer_context.mlflow_experiment.read_overview",
        lambda experiment_id=None: ExperimentOverview(experiment_id="exp_b", project_name="demo"),
    )
    monkeypatch.setattr(
        "automl.agent.proposer_context.mlflow_experiment.mlflow_experiment_id",
        lambda experiment_id=None: "42",
    )
    monkeypatch.setattr(
        "automl.agent.proposer_context.find_prior_experiment",
        lambda *, session=None: {"experiment_id": "exp_a", "created_at": "2026-04-01T00:00:00Z"},
    )
    monkeypatch.setattr(
        "automl.agent.proposer_context.load_error_report",
        lambda run_id: {
            "phase": "model_import",
            "error_class": "SyntaxError",
            "message": "invalid syntax",
            "traceback_artifact": "logs/errors/traceback.txt",
            "traceback_tail": ["SyntaxError: invalid syntax"],
        },
    )

    packet = gather_proposer_context(metric="auc", n_top=3, session=active)

    assert packet["schema_version"] == 1
    assert packet["project_name"] == "demo"
    assert packet["experiment_id"] == "exp_b"
    assert packet["mlflow_experiment_id"] == "42"
    assert packet["project_instructions"] == "Prefer fast single-model trials."
    assert packet["leaderboard"]["rows"][0]["run_id"] == "agent-run"
    assert packet["human_trials"][0]["run_id"] == "human-run"
    assert packet["recent_failures"][0]["run_id"] == "failed-run"
    assert packet["recent_failures"][0]["error"] == {
        "phase": "model_import",
        "error_class": "SyntaxError",
        "message": "invalid syntax",
        "traceback_artifact": "logs/errors/traceback.txt",
        "traceback_tail": ["SyntaxError: invalid syntax"],
    }
    assert packet["strategies_attempted"]["baseline"] == 1
    assert packet["trial_count"] == 3
    assert packet["environment"]["allowed_dependencies"] == [
        "pandas",
        "numpy",
        "scikit-learn",
    ]
    assert packet["project_contract"]["target_column"] == "target"
    assert packet["project_contract"]["raw_target_column"] == "TARGET"
    assert packet["project_contract"]["primary_metric"] == "auc"
    assert packet["project_contract"]["required_transformers"] == [
        {
            "name": "demo_required",
            "type": "FunctionTransformer",
            "import_path": "sklearn.preprocessing._function_transformer.FunctionTransformer",
            "columns": ["category"],
        }
    ]
    assert packet["data_context"]["active_dataset"]["id"] == "dataset-v1"
    assert (
        packet["data_context"]["profile"]["profile_manifest_uri"] == "runs:/overview/profile.json"
    )
    assert packet["prior_experiment"] is None
    assert "top_trials" not in packet
    assert "learnings" not in packet
    assert "artifact_uris" not in packet


def test_find_prior_experiment_uses_creation_time_not_lexicographic_order(monkeypatch, tmp_path):
    from automl.agent.proposer_context import find_prior_experiment

    active = _session(tmp_path)
    overviews = {
        "exp_a": ExperimentOverview(
            experiment_id="exp_a",
            project_name="demo",
            created_at="2026-05-20T00:00:00Z",
        ),
        "exp_c": ExperimentOverview(
            experiment_id="exp_c",
            project_name="demo",
            created_at="2026-05-10T00:00:00Z",
        ),
        "exp_b": ExperimentOverview(
            experiment_id="exp_b",
            project_name="demo",
            created_at="2026-05-28T00:00:00Z",
        ),
    }

    monkeypatch.setattr(
        "automl.agent.proposer_context.mlflow_project.list_experiments",
        lambda: ["exp_c", "exp_a", "exp_b"],
    )
    monkeypatch.setattr(
        "automl.agent.proposer_context.mlflow_experiment.read_overview",
        lambda experiment_id=None: overviews.get(experiment_id),
    )

    assert find_prior_experiment(session=active) == {
        "experiment_id": "exp_a",
        "project_name": "demo",
        "created_at": "2026-05-20T00:00:00Z",
    }
