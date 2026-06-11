# Read-Once Contract Builder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `_trial_data_contract` loads the full frame **once** and derives every split's slice contract in memory, instead of one full `load_dataset_by_id` per non-fit split (3 extra full parses on neobank).

**Architecture:** A contained change to `automl/runner/trial.py::_trial_data_contract` (design §5). Slice-hash semantics stay identical — hash the *sliced* frame with the same `dataframe_content_hash` — so existing `SliceContract` records and tag-lineage verification are untouched. Independent of plan 2 (works with or without the cache; with the cache, the single remaining load is a local read).

**Tech Stack:** Python 3.13, pandas, pytest via `uv run`.

**Design:** `docs/execution/trial-reliability/design.md` §5; evidence `../finding-redundant-full-reads.md`.

---

### Task 1: Rewrite `_trial_data_contract` to slice in memory

**Files:**
- Modify: `automl/runner/trial.py::_trial_data_contract` (~line 568)
- Test: `tests/unit/runner/test_trial_data_contract.py` (create)

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/runner/test_trial_data_contract.py`:

```python
from __future__ import annotations

import pandas as pd
import pytest

from automl.data.dataset import Dataset, LoadedDataset, LoadedSlice
from automl.data.features import FeatureRegistry
from automl.project import Where
from automl.project.run_config import ModelRoute, ModelsConfig, RunConfig, Splits
from automl.runner import trial as trial_module
from automl.utils.hashing import dataframe_content_hash

pytestmark = pytest.mark.unit


_FRAME = pd.DataFrame(
    {
        "SPLIT_PCT": [10, 30, 70, 85, 95],
        "y": [0, 1, 0, 1, 0],
    }
)


def _dataset() -> Dataset:
    return Dataset.from_dict(
        {
            "id": "ds_001",
            "identity_hash": "sha256:identity",
            "component_hashes": {
                "source_identity": "sha256:src",
                "feature_registry": "sha256:reg",
                "data_content": "sha256:content",
                "schema": "sha256:schema",
            },
            "gcs_bucket": "bucket-x",
            "project_name": "proj",
            "created_at": "2026-06-10T00:00:00Z",
            "source_identity": {},
            "n_rows": 5,
            "n_columns": 2,
            "target_column": "y",
        }
    )


def _registry() -> FeatureRegistry:
    return FeatureRegistry.from_dataframe(
        pd.DataFrame({"name": ["y"], "dtype": ["int64"]})
    )


def _run_config() -> RunConfig:
    route = ModelRoute(model="claude-test", effort="low")
    return RunConfig(
        experiment_id="exp",
        splits=Splits(
            train=Where("SPLIT_PCT") < 80,
            test=Where("SPLIT_PCT") >= 80,
        ),
        models=ModelsConfig(manager=route, proposer=route, coder=route),
        per_trial_seconds=600,
    )


class _Config:
    def require_run_config(self):
        return _run_config()


class _Session:
    config = _Config()
    project_name = "proj"
    active_experiment_id = "1"


def _loaded_fit() -> LoadedSlice:
    predicate = _run_config().splits.resolve("train")
    df = _FRAME[predicate.mask(_FRAME)].reset_index(drop=True)
    return LoadedSlice(
        dataset=_dataset(),
        df=df,
        registry=_registry(),
        split_name="train",
        predicate=predicate,
    )


def test_contract_loads_full_frame_exactly_once(monkeypatch):
    calls = []

    def fake_load(dataset_id, *, split_name=None, predicate=None, session=None):
        calls.append({"dataset_id": dataset_id, "split_name": split_name})
        assert split_name is None and predicate is None, "must load the FULL frame"
        return LoadedDataset(dataset=_dataset(), df=_FRAME.copy(), registry=_registry())

    monkeypatch.setattr(trial_module.data, "load_dataset_by_id", fake_load)
    contract = trial_module._trial_data_contract(
        active=_Session(),
        run_id="run1",
        trial_id="1_test",
        loaded_fit=_loaded_fit(),
    )
    assert len(calls) == 1
    assert calls[0]["dataset_id"] == "ds_001"
    assert {s.name for s in contract.slices} == {"train", "test"}


def test_slice_hashes_match_direct_slicing(monkeypatch):
    monkeypatch.setattr(
        trial_module.data,
        "load_dataset_by_id",
        lambda *a, **k: LoadedDataset(
            dataset=_dataset(), df=_FRAME.copy(), registry=_registry()
        ),
    )
    contract = trial_module._trial_data_contract(
        active=_Session(),
        run_id="run1",
        trial_id="1_test",
        loaded_fit=_loaded_fit(),
    )
    run_config = _run_config()
    for slice_contract in contract.slices:
        predicate = run_config.splits.resolve(slice_contract.name)
        expected = _FRAME[predicate.mask(_FRAME)].reset_index(drop=True)
        assert slice_contract.n_rows == len(expected)
        assert slice_contract.content_hash == dataframe_content_hash(expected)
        assert slice_contract.predicate == predicate.to_dict()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/runner/test_trial_data_contract.py -q`
Expected: FAIL — the first test's `fake_load` assertion trips (`split_name` is not None: today's code requests one split per call), and call count is 1 per non-fit split rather than a single full load.

- [ ] **Step 3: Rewrite `_trial_data_contract`**

Replace the function body in `automl/runner/trial.py`:

```python
def _trial_data_contract(
    *,
    active: Session,
    run_id: str,
    trial_id: str,
    loaded_fit,
) -> TrialDataContract:
    run_config = active.config.require_run_config()
    # One full-frame load (a local cache hit once plan 2 lands); every split's
    # slice is derived in memory. Hash semantics are unchanged: hash the
    # *sliced* frame, exactly as the per-split loads did.
    full = data.load_dataset_by_id(loaded_fit.id, session=active)
    slices: list[SliceContract] = []
    for name, predicate in run_config.splits.predicates.items():
        sliced = full.df[predicate.mask(full.df)].reset_index(drop=True)
        slices.append(
            SliceContract(
                name=name,
                predicate=predicate.to_dict(),
                n_rows=len(sliced),
                content_hash=dataframe_content_hash(sliced),
            )
        )
        del sliced
    contract = TrialDataContract(
        trial=TrialRef(
            project_name=active.project_name,
            experiment_id=active.active_experiment_id,
            trial_id=trial_id,
            run_id=run_id,
        ),
        dataset=DatasetRef.from_dataset(full.dataset),
        splits={
            name: predicate.to_dict() for name, predicate in run_config.splits.predicates.items()
        },
        slices=tuple(slices),
    )
    del full
    return contract
```

Notes for the implementer:
- The fit split is no longer special-cased: its slice is recomputed from the full frame with the same predicate, which yields an identical hash to the previous `loaded_fit` reuse (the mask is deterministic). `loaded_fit` is still the source of `.id`.
- `del sliced` / `del full` keep peak memory at full-frame + one slice; do not hold the full frame past this function (design §5: load → hash → release).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/runner/test_trial_data_contract.py -q`
Expected: all PASS.

- [ ] **Step 5: Run the full runner suite + contracts**

Run: `uv run pytest tests/unit/runner tests/unit/data tests/contracts -q`
Expected: all PASS (`test_trial_folder_execution.py` exercises the runner end-to-end with fakes; if its fake `load_dataset_by_id` asserted per-split calls, update it to serve the full-frame call).

- [ ] **Step 6: Commit**

```bash
git add automl/runner/trial.py tests/unit/runner/test_trial_data_contract.py
git commit -m "perf(runner): build slice contracts from one in-memory frame, not one load per split"
```

---

### Task 2: Measure the remaining hash cost (evidence, no code)

**Files:** none — paste numbers into the PR description (design §9: "slice-hash CPU").

- [ ] **Step 1: Time `dataframe_content_hash` at realistic scale**

Run:

```bash
uv run python -c "
import time
import numpy as np, pandas as pd
from automl.utils.hashing import dataframe_content_hash
rng = np.random.default_rng(0)
df = pd.DataFrame(rng.standard_normal((200_000, 500)))
df.columns = [f'c{i}' for i in df.columns]
started = time.perf_counter()
dataframe_content_hash(df)
print(f'200k x 500: {time.perf_counter() - started:.2f}s')
"
```

Expected: a number (likely a few seconds). Scale linearly to estimate the 1.15M × 2,253 slice cost and note it in the PR — if hashing dominates the contract step once the network is gone, that's a *recorded observation* for a future optimization effort, not something to fix in this plan.

---

## Done criteria

- `uv run pytest tests/unit tests/contracts -q` green.
- `grep -n "load_dataset_by_id" automl/runner/trial.py` shows exactly two call sites: the fit load and the single full load in `_trial_data_contract`.
- The hash-cost measurement is recorded in the PR description.
