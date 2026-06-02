from dataclasses import replace

import pytest

from automl.errors import ProjectError
from automl.project import BinaryClassification, ProjectConfig

pytestmark = pytest.mark.unit


def _write_project(tmp_path, name="demo", config_source=""):
    project_dir = tmp_path / "projects" / name
    project_dir.mkdir(parents=True)
    (tmp_path / "projects" / "__init__.py").write_text("")
    (project_dir / "__init__.py").write_text("")
    (project_dir / "config.py").write_text(config_source)
    return project_dir


def test_project_config_loads_project_config_symbol(tmp_path, monkeypatch):
    config_source = """
from automl.project import BinaryClassification, ProjectConfig, RunConfig, Splits, ModelRoute, ModelsConfig

class DummyEvalSpec:
    primary_name = "auc"

TASK = BinaryClassification(target="TARGET")
DATA = object()
EVAL = DummyEvalSpec()
RUN_CONFIG = RunConfig(
    experiment_id="demo-exp",
    splits=Splits(train=[(0, 80)], test=[(80, 100)]),
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
    project_dir = _write_project(tmp_path, config_source=config_source)
    monkeypatch.setenv("GCS_BUCKET", "bucket")
    monkeypatch.setenv("GCS_PREFIX", "prefix")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

    config = ProjectConfig.load("demo", repo_root=tmp_path)

    assert config.project_name == "demo"
    assert config.project_dir == project_dir
    assert config.config_path == project_dir / "config.py"
    assert config.task == BinaryClassification(target="TARGET")
    assert config.data_spec is not None
    assert config.eval_spec is not None
    assert config.run_config.experiment_id == "demo-exp"
    assert config.gcs_bucket == "bucket"
    assert config.gcs_prefix == "prefix"
    assert config.mlflow_tracking_uri == "sqlite:///mlflow.db"
    assert not hasattr(config, "mlflow_artifacts_destination")
    assert config.is_complete()
    assert config.missing_fields() == []
    assert config.raw_target_column == "TARGET"
    assert config.target_column == "target"
    assert config.primary_metric == "auc"
    assert config.per_trial_seconds == 120
    assert config.models.manager.model == "sonnet"


def test_project_config_accepts_project_config_symbol(tmp_path):
    config_source = """
from automl.project import BinaryClassification, ProjectConfig

PROJECT_CONFIG = ProjectConfig.partial(task=BinaryClassification(target="TARGET"))
"""
    _write_project(tmp_path, config_source=config_source)

    config = ProjectConfig.load("demo", repo_root=tmp_path)

    assert config.task == BinaryClassification(target="TARGET")
    assert config.data_spec is None
    assert config.missing_fields() == ["DATA_SPEC", "EVAL_SPEC", "RUN_CONFIG"]


def test_project_config_repeated_same_root_load_preserves_project_class_identity(tmp_path):
    project_dir = _write_project(
        tmp_path,
        config_source="""
from projects.demo.helper import Marker
from automl.project import ProjectConfig

DATA = Marker()
PROJECT_CONFIG = ProjectConfig.partial(data_spec=DATA)
""",
    )
    (project_dir / "helper.py").write_text(
        """
class Marker:
    pass
"""
    )

    first = ProjectConfig.load("demo", repo_root=tmp_path)
    second = ProjectConfig.load("demo", repo_root=tmp_path)

    assert type(first.data_spec) is type(second.data_spec)


def test_project_config_switching_roots_reimports_project_modules(tmp_path):
    root_a = tmp_path / "root_a"
    root_b = tmp_path / "root_b"
    config_source = """
from projects.demo.helper import Marker
from automl.project import ProjectConfig

DATA = Marker()
PROJECT_CONFIG = ProjectConfig.partial(data_spec=DATA)
"""
    project_a = _write_project(root_a, config_source=config_source)
    project_b = _write_project(root_b, config_source=config_source)
    (project_a / "helper.py").write_text(
        """
class Marker:
    origin = "a"
"""
    )
    (project_b / "helper.py").write_text(
        """
class Marker:
    origin = "b"
"""
    )

    first = ProjectConfig.load("demo", repo_root=root_a)
    second = ProjectConfig.load("demo", repo_root=root_b)

    assert type(first.data_spec) is not type(second.data_spec)
    assert second.data_spec.origin == "b"


def test_project_config_load_allows_missing_config_py(tmp_path):
    project_dir = tmp_path / "projects" / "empty"
    project_dir.mkdir(parents=True)
    (tmp_path / "projects" / "__init__.py").write_text("")
    (project_dir / "__init__.py").write_text("")

    config = ProjectConfig.load("empty", repo_root=tmp_path)

    assert config.project_name == "empty"
    assert not config.is_complete()
    assert config.missing_fields() == ["TASK", "DATA_SPEC", "EVAL_SPEC", "RUN_CONFIG"]


def test_project_config_rejects_config_without_project_config(tmp_path):
    _write_project(
        tmp_path,
        config_source="""
from automl.project import BinaryClassification

TASK = BinaryClassification(target="TARGET")
""",
    )

    with pytest.raises(ProjectError, match="PROJECT_CONFIG"):
        ProjectConfig.load("demo", repo_root=tmp_path)


def test_project_config_require_methods_raise_with_config_path(tmp_path):
    _write_project(
        tmp_path,
        config_source="""
from automl.project import ProjectConfig

PROJECT_CONFIG = ProjectConfig.partial()
""",
    )
    config = ProjectConfig.load("demo", repo_root=tmp_path)

    with pytest.raises(ProjectError, match="TASK missing"):
        config.require_task()
    with pytest.raises(ProjectError, match=str(config.config_path)):
        config.require_run_config()


def test_project_config_is_frozen(tmp_path):
    _write_project(
        tmp_path,
        config_source="""
from automl.project import ProjectConfig

PROJECT_CONFIG = ProjectConfig.partial()
""",
    )
    config = ProjectConfig.load("demo", repo_root=tmp_path)

    with pytest.raises(Exception):
        config.project_name = "other"

    assert replace(config, project_name="other").project_name == "other"
