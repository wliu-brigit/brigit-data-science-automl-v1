from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.contract

REPO_ROOT = Path(__file__).resolve().parents[2]
ACTIVE_DOCS = [
    "CLAUDE.md",
    "agent-skills/references/setup/data-pipeline.md",
]


def _read(relative: str) -> str:
    return (REPO_ROOT / relative).read_text(encoding="utf-8")


def test_active_data_docs_do_not_advertise_load_training_data_hook():
    offenders = [relative for relative in ACTIVE_DOCS if "load_training_data" in _read(relative)]

    assert offenders == []


def test_data_pipeline_docs_show_named_split_loading_surface():
    text = _read("agent-skills/references/setup/data-pipeline.md")

    assert "data.load_dataset(split_name=" in text
    assert 'data.load_dataset_by_id(dataset_id, split_name="train", session=active)' in text
    assert 'data.load_dataset_by_trial(trial_id, split_name="train", session=active)' in text
    assert "run_config.train_split" in text
    assert "run_config.eval_split" in text
