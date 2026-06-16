# To-do (tiny): a half-written eval artifact hard-blocks the retry

**Status:** noted, not started. Small but a real robustness gap.

## What happens

`evaluate()` → `write_predictions()` writes two things for a split: the GCS
`predictions.parquet` (large-bytes payload) **then** the MLflow JSON manifest.
If the second write fails (e.g. a transient MLflow TLS/network error mid-run),
the parquet is already in GCS but nothing records it. The next attempt calls
the GCS write with `overwrite=False` → `ifGenerationMatch=0` (create-only), so
it gets **HTTP 412 PreconditionFailed** and the whole eval hard-fails — the
orphaned parquet from the dead attempt now permanently blocks every retry until
someone manually deletes it.

Hit on neobank_ncm VPN day (2026-06-09): a `train_known` eval died on an MLflow
TLS error after the parquet landed; re-running 412'd until the orphan was
cleared with `gcs.delete_prefix(...)`.

## Options

- Make the eval-artifact write **idempotent**: pass `overwrite=True` for
  prediction payloads (an eval is deterministic for a fixed model+split, so
  re-writing the same parquet is safe), or
- write parquet + manifest as a single all-or-nothing step (manifest first, or
  clean up the parquet if the manifest write fails), or
- at minimum, on 412 detect the orphan and overwrite rather than raising.

Relates to [[automl-agent-logging-decisions]] (retry/download fix design).

## Don't forget

- Lives in `automl/mlflow/trial/artifacts/` (predictions.py / data.py) — the
  seam, not project code.
- Add an integration test that simulates a mid-write failure then re-runs.
- Delete this file once landed.
