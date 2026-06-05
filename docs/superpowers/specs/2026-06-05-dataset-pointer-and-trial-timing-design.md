# Dataset Pointer and Trial Timing Design

## Status

Approved in conversation on 2026-06-05. This is a forward-only design; no
backward-compatibility behavior is required.

## Goals

- Make dataset selection explicit, inspectable, and shared by proposer, coder,
  runner, profile, and notebook workflows.
- Remove implicit latest-dataset fallback from default read paths.
- Keep one canonical trial timing artifact at `timing/summary.json`.
- Preserve the current runner timing readability while adding agent-loop timing
  in chronological order.

## Dataset Selection

The experiment has exactly one active dataset pointer.

- Dataset records remain MLflow experiment artifacts:
  `datasets/<dataset_id>/dataset.json`.
- The canonical read-optimized pointer remains the MLflow experiment tag
  `data.active_dataset_id`.
- A human-readable mirror is written as the experiment artifact
  `datasets/active_pointer.json`.
- All pointer writes go through one core function, `activate_dataset(...)`,
  which validates the dataset record exists and writes both the tag and artifact.
- All default dataset reads go through one core function,
  `resolve_active_dataset(...)`, which validates the tag and artifact agree.
- If the active pointer is missing, invalid, or inconsistent, default reads fail
  loudly. There is no latest fallback.

`active` means selected, not newest. A user can activate V2 while V3 exists.

### Dataset Write Rules

- First materialize creates the first dataset version and activates it.
- Non-refresh materialize attaches to the active dataset only.
- Refresh materialize derives data, attaches or mints the resulting dataset, and
  activates that resulting dataset.
- `automl data activate <dataset_id>` changes the experiment active dataset
  without materializing data.
- Library callers get the same behavior via `activate_dataset(dataset_id, ...)`.

### Dataset Read Rules

- `load_dataset(...)`, `profile(...)`, `get_profile(...)`, proposer context, and
  runner default data load use `resolve_active_dataset(...)`.
- `load_dataset_by_id(...)` remains the explicit bypass.
- A trial-level dataset override may be added as `automl trial run --dataset-id
  <dataset_id>`. It applies only to that trial and does not change the active
  pointer.
- Config defines the data recipe/source. It does not select a dataset version.

### Trial Lineage

Each trial records the dataset actually used through the existing trial data
contract and tags:

- `data.dataset_id`
- `data.identity_hash`
- `data.record_uri`
- slice content hashes and row counts

This keeps historical replay independent of later active-pointer changes.

## Timing

`timing/summary.json` is the only canonical timing artifact for a trial.
Agent reports may keep messages and tool events, but not separate timing truth.

Numbers are seconds rounded to five decimal places. Object insertion order is
the display order and should follow chronological execution.

### Timing Schema

```json
{
  "schema_version": 2,
  "unit": "seconds",
  "total_seconds": 123.45678,
  "phases": {
    "setup": 4.12345,
    "proposer": 12.34567,
    "proposal_handoff": 1.23456,
    "coder_implementation": 18.12345,
    "runner": 58.23456,
    "coder_report": 2.34567,
    "publish": 3.12345
  },
  "phase_details": {
    "setup": {
      "total_seconds": 4.12345,
      "phases": {
        "data_materialize": 2.10000,
        "experiment_proposer_context": 2.02345
      }
    },
    "proposer": {
      "total_seconds": 12.34567
    },
    "proposal_handoff": {
      "total_seconds": 1.23456,
      "phases": {
        "validate_proposal": 0.30000,
        "persist_proposal": 0.10000,
        "create_trial": 0.83456
      }
    },
    "coder_implementation": {
      "total_seconds": 18.12345
    },
    "runner": {
      "total_seconds": 58.23456,
      "phases": {
        "model_import": 0.00061,
        "data_load": 3.63819,
        "pre_fit_validation": 0.02141,
        "mlflow_setup": 0.28264,
        "fit": 0.00664,
        "contract_validation": 0.00899,
        "local_artifacts": 0.09603,
        "mlflow_pyfunc_log": 3.06452,
        "evaluation": 15.65664,
        "validation_fixture": 2.52407,
        "validation_fixture_publish": 0.00130,
        "validation": 8.22876,
        "validation_publish": 0.38654
      }
    },
    "coder_report": {
      "total_seconds": 2.34567
    },
    "publish": {
      "total_seconds": 3.12345
    }
  }
}
```

### Timing Semantics

- `setup`: observed loop setup before the proposer. Include whatever measured
  setup steps are present for that run, such as dataset materialization,
  experiment context rendering, or proposer-context rendering. The breakdown is
  data-driven: some runs may have several setup entries, and some may have only
  the setup total.
- `proposer`: wall-clock time of the proposer subagent.
- `proposal_handoff`: main-session/tooling work between proposer and coder,
  including proposal validation, proposal persistence, and trial creation. If
  validation and persistence are measured as one command, keep them as one
  detail entry rather than inventing a split.
- `coder_implementation`: coder subagent time before the runner starts, mainly
  reading context and editing `model.py`.
- `runner`: actual trial runner execution.
- `coder_report`: coder subagent time after the runner ends, mainly interpreting
  command output and writing the final response.
- `publish`: agent timeline publish/reconciliation time.

Current architecture runs the runner inside the coder subagent. The preferred
implementation records runner start/end timestamps so `coder_implementation`,
`runner`, and `coder_report` are exact chronological phases. If that proves too
heavy for the first pass, a derived split is acceptable: use the coder wall-clock
span and runner duration to estimate coder time outside the runner, keep the
math simple, and avoid adding extra schema fields just to explain the estimate.

`total_seconds` is wall-clock time: the chronological span covered by the timing
report. It is not required to equal the sum of `phases` if there is
uninstrumented time.

## Error Handling

- Missing active dataset pointer: error.
- Active pointer references a missing dataset record: error.
- Active tag and `datasets/active_pointer.json` disagree: error.
- Timing enrichment cannot find agent spans: preserve runner-only
  `timing/summary.json` and do not invent agent timings.
- Timing enrichment finds runner duration but no runner timestamps: prefer
  adding runner boundary instrumentation. If that is too invasive, derive the
  coder split from coder wall-clock minus runner duration without introducing a
  second timing artifact or extra timing schema.

## Testing

- Unit tests for `activate_dataset(...)` writing tag and artifact together.
- Unit tests for `resolve_active_dataset(...)` rejecting missing, invalid, and
  inconsistent pointers.
- CLI tests for `automl data activate <dataset_id>`.
- Integration tests showing materialize activates first and refreshed datasets.
- Runner/proposer tests proving both resolve the same active dataset.
- Timing unit tests for rounding, chronological phase order, and schema shape.
- Timeline publish tests showing `timing/summary.json` is enriched with setup,
  proposer, proposal handoff, coder implementation, runner, coder report, and
  publish phases when timestamps are available.
