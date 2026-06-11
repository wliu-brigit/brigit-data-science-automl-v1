# To-do (tiny): torch pyfunc-reload SIGSEGVs serving validation (NN family)

**Status:** hit on neobank_ncm full-data loop 2026-06-10, trial
`tabular_mlp_softlabel_challenger` (PyTorch tabular MLP, `new_model`). Native
crash, **no MLflow run and no `AUTOML_ERROR` produced** — the fault bypasses the
runner's error tagging entirely. Distinct from the bytes-timeout and the
content-hash issues.

## What happened

The NN trial trained and logged artifacts fine, then the process died with
**SIGSEGV (exit 139)** during the post-fit **serving-validation subprocess**
(`automl/runner/serving_validation.py`), which `mlflow.pyfunc.load_model`s the
model and benchmarks it. Re-importing `torch` inside that reloaded pyfunc,
alongside multiprocessing on macOS (leaked-semaphore signature), is a known
native-crash class. The dry-run NN (11K rows) passed; full-data + the reload
path is where it bit.

## Two gaps, ranked

1. **Robustness (the real bug): a native crash in the validation subprocess
   leaves the trial with no result JSON, no MLflow run, no error tag** — it's
   invisible except in stdout. The runner should detect a signal-killed
   subprocess exit (`returncode < 0`) and record a
   failure report + tag, the same way the timeout path now does. Without this,
   any native crash in validation is silent.

2. **Mitigation (likely makes NN work):** set, in the trial/validation env,
   `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `KMP_DUPLICATE_LIB_OK=TRUE`, and
   call `torch.set_num_threads(1)` in NN `model.py`. The handoff already flagged
   these for the NN family (`docs/HANDOFF.md`, "native crash" note). Cheapest
   first step: export them before `experiment run` and retry the NN.

## Don't forget

- Relates to `docs/to-do/serving-validation-timeout-crashes-and-halts-loop.md`
  (same handler, same "don't crash the trial" intent) and
  [[automl-agent-logging-decisions]].
- A signal-exit guard in `_run_pyfunc_validation` is the durable fix; the env
  vars are the unblock-the-NN-now fix.
- Delete this file once a native validation crash records a clean failure.
