"""Offline end-to-end dry run: pipeline → splits → baseline fit → known-only AUC.

Exercises the whole chain on a synthetic CSV standing in for the Snowflake
snapshot — no warehouse, no MLflow server. This is the pre-VPN proof that
the recipe, the split predicates, the metadata/exclusion boundary, and the
baseline replication model all work against the harness contracts.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
from sklearn.metrics import roc_auc_score

from automl.data import DataSpec, LocalCSVSource, build_dataset
from automl.model import validate_model
from automl.project import ProjectConfig, Session
from projects.neobank_ncm.tests.fixtures import write_fixture_csv

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def loaded_dataset(tmp_path_factory, monkeypatch_module):
    monkeypatch_module.setenv("MLFLOW_TRACKING_URI", "file:///tmp/mlruns-neobank-test")
    csv_path = write_fixture_csv(tmp_path_factory.mktemp("fixture") / "base_table.csv")
    config = ProjectConfig.load("neobank_ncm", repo_root=REPO_ROOT)
    spec = DataSpec(
        source=LocalCSVSource(
            csv_path=csv_path, unique_key="entity_id", split_group_key="user_id"
        ),
        metadata_cols=config.data_spec.metadata_cols,
        exclude_cols=config.data_spec.exclude_cols,
        dry_run_rows=config.data_spec.dry_run_rows,
    )
    config = dataclasses.replace(config, data_spec=spec)
    loaded = build_dataset(session=Session(config=config, dry_run=True))
    return config, loaded


@pytest.fixture(scope="module")
def monkeypatch_module():
    from _pytest.monkeypatch import MonkeyPatch

    mp = MonkeyPatch()
    yield mp
    mp.undo()


@pytest.fixture
def model_class(loaded_dataset):
    # Resolve AFTER ProjectConfig.load: the harness's project importer purges
    # projects.* modules that belong to another repo root (repo-level tests
    # load scaffolded tmp projects), so a module-level binding captured at
    # collection time can reference stale class identities.
    del loaded_dataset  # ordering dependency only
    from projects.neobank_ncm.model import MODEL_CLASS

    return MODEL_CLASS


def test_pipeline_drops_excluded_and_flags_metadata(loaded_dataset):
    _, loaded = loaded_dataset
    df, registry = loaded.df, loaded.registry

    # excluded and metadata columns stay in the frame but never become
    # features (fixture plants signupsourcetype, a Pass-0 exclusion)
    features = set(registry.get_by_flag("feature"))
    assert "signupsourcetype" not in features
    assert "signupsourcetype" not in set(registry.get_by_flag("model"))
    for column in (
        "synthetic_score",
        "is_known",
        "split",
        "entity_id",
        "user_id",
        "unknown_train_hash_rank",
    ):
        assert column not in features
    assert "went_dpd45" not in features
    # candidate features present, including derived/plaid/netflow names
    for column in (
        "bankinstitution",
        "highestpayfrequency",
        "balancesdtodailyincomemeanratio",
        "incomebuffertodaystopaydayratio",
        "plaidfeaturessummary_incomewages_lookbackwindow14d_inflow_sum",
        "total_incount_14",
    ):
        assert column in features
    # metadata still rides along in the frame for trial code
    assert "synthetic_score" in df.columns


def test_splits_carve_the_legacy_windows(loaded_dataset):
    config, loaded = loaded_dataset
    df = loaded.df
    splits = config.run_config.splits

    train = df[splits.resolve("train").mask(df)]
    test = df[splits.resolve("test").mask(df)]
    oot = df[splits.resolve("oot").mask(df)]

    # fixture: 1200 known + 900 unknown in Jan–Oct, 400 known Nov–Dec,
    # 400 known + 200 unknown oot
    assert len(train) == 2100
    assert len(test) == 400
    assert len(oot) == 400

    assert test["is_known"].all()
    assert oot["is_known"].all()
    assert test["went_dpd45"].notna().all()
    assert oot["went_dpd45"].notna().all()
    # train mixes known (labels) and unknown (synthetic scores)
    assert train["went_dpd45"].isna().sum() == 900
    assert train.loc[train["went_dpd45"].isna(), "synthetic_score"].notna().all()
    # no overlap between the loop splits and the held-out oot
    assert not set(train["entity_id"]) & set(oot["entity_id"])
    assert not set(test["entity_id"]) & set(oot["entity_id"])


def test_baseline_passes_runner_prefit_contract(loaded_dataset, model_class):
    config, loaded = loaded_dataset
    # mirrors the runner exactly: contract validation fits on a 200-row head
    # sample WITH the session bound, so the required-transformer check (the
    # WoE entry must sit inside the model's ColumnTransformer) is enforced
    report = validate_model(
        model_class,
        df=loaded.df.head(200),
        registry=loaded.registry,
        session=Session(config=config, dry_run=True),
    )
    assert report.passed, [issue.message for issue in report.issues]


def test_baseline_fit_and_known_only_auc(loaded_dataset, model_class):
    config, loaded = loaded_dataset
    df = loaded.df
    splits = config.run_config.splits
    train = df[splits.resolve("train").mask(df)]
    test = df[splits.resolve("test").mask(df)]
    oot = df[splits.resolve("oot").mask(df)]

    model = model_class().fit(train, loaded.registry, seed=0)

    # the locked feature list only partially exists in the fixture — the
    # model must record what is missing rather than fail
    assert model.missing_features_
    assert model.woe_encoder_.mapping_  # fit on known rows only

    # MLflow-compatible predict path (registry cast/select → transform → score)
    test_scores = model.predict(model_input=test)
    oot_scores = model.predict(model_input=oot)

    test_auc = roc_auc_score(test["went_dpd45"].astype(int), test_scores)
    oot_auc = roc_auc_score(oot["went_dpd45"].astype(int), oot_scores)
    assert test_auc > 0.60, f"test AUC {test_auc:.3f} — planted signal not recovered"
    assert oot_auc > 0.60, f"oot AUC {oot_auc:.3f} — planted signal not recovered"


def test_oot_unknowns_belong_to_no_split(loaded_dataset):
    # The 2026 unknown rows ride in the frame (synthetic_scores_final covers
    # the oot window, so they arrive scored) but must be invisible to every
    # named split: never trained on, never in the leaderboard eval, and not
    # part of the one sanctioned oot read (which is known-only by predicate).
    config, loaded = loaded_dataset
    df = loaded.df
    splits = config.run_config.splits

    oot_unknown = df[(df["split"] == "oot") & (~df["is_known"])]
    assert len(oot_unknown) == 200  # the fixture plants them
    assert oot_unknown["synthetic_score"].notna().all()
    for name in ("train", "train_known", "test", "oot"):
        split_idx = df.index[splits.resolve(name).mask(df)]
        assert oot_unknown.index.intersection(split_idx).empty, (
            f"oot-unknown rows leaked into the {name} split"
        )


def test_unknown_downsample_replay_is_deterministic(loaded_dataset, monkeypatch):
    config, loaded = loaded_dataset
    df = loaded.df
    train = df[config.run_config.splits.resolve("train").mask(df)]

    # import fresh (post-load) and use the SAME module object for both the
    # constant and the class — see the model_class fixture note
    from projects.neobank_ncm.model import baseline

    # tighten the legacy 200K cutoff so it binds on the 900-row fixture pool
    monkeypatch.setattr(baseline, "UNKNOWN_TRAIN_SAMPLE", 400)

    first = baseline.MODEL_CLASS().fit(train, loaded.registry)
    second = baseline.MODEL_CLASS().fit(train, loaded.registry)

    # 1200 known at the locked 80/20 ratio -> 299 unknowns: the legacy
    # formula int(n*(1-r)/r) truncates 299.999... in float, and the replica
    # keeps that exact behavior (282,642 known -> 70,660, as in the notebook)
    assert len(first.sampled_unknown_ids_) == 299
    assert first.sampled_unknown_ids_ == second.sampled_unknown_ids_
    ranks = train.set_index("entity_id").loc[
        first.sampled_unknown_ids_, "unknown_train_hash_rank"
    ]
    assert (ranks <= 400).all()


def test_baseline_handles_all_unknown_sample(loaded_dataset, model_class):
    _, loaded = loaded_dataset
    df = loaded.df
    unknown_only = df[df["went_dpd45"].isna()].head(120)

    model = model_class().fit(unknown_only, loaded.registry)
    scores = model.predict(model_input=unknown_only.head(10))
    assert len(scores) == 10
