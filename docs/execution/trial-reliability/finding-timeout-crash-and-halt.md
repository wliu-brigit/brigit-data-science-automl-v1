# To-do: serving-validation timeout crashes the handler and halts the loop

**Status:** found on neobank_ncm first real loop run (2026-06-09→10), exp
`neobank_ncm/neobank_ncm_v3_replicate`, trial `2_xgb_unconstrained`
(run `0ccb7f64c1...`). **Core change** (`automl/runner/serving_validation.py`)
— this branch has been project-only, so flag before touching core.

## What happened

The challenger trained fine and logged `eval.test.auc=0.7014`, then the
runner's post-training **serving-validation subprocess hit the 120s timeout**
and the trial was marked `FAILED`. The loop then **stopped** (1 of 3 iterations
ran). Three distinct problems stacked up:

### 1. The timeout handler crashes on `bytes` (the actual bug)

`serving_validation.py:536`:

```python
stderr_tail = (exc.stderr or b"")[-1000:] if isinstance(exc.stderr, bytes) else (exc.stderr or "")[-1000:]
```

On `TimeoutExpired`, `exc.stderr` is **bytes** (even with `text=True`), and this
keeps it as bytes — never decodes. It goes into `report["stderr_tail"]`, then
`json.dumps(report)` → `TypeError: Object of type bytes is not JSON
serializable`. So the handler whose own comment says *"Don't crash the trial"*
crashes the trial. **Fix:** decode →
`raw[-1000:].decode("utf-8", "replace")` (same for any stdout tail).

### 2. The 120s timeout is too tight — and VPN makes it worse

`_VALIDATION_TIMEOUT_S = 120`. The **baseline took 120.07s** of validation —
literally at the boundary; the challenger tipped over. Serving validation loads
the model from GCS + benchmarks it, and **GCS is throttled on the Check Point
VPN** (see [[gcs-vpn-throughput]]). The loop doesn't need Snowflake (dataset is
already materialized in GCS), so **running it off-VPN makes validation finish in
seconds** — the cleanest immediate unblock, no code change. Longer term: raise
the timeout and/or make it configurable.

### 3. A validation timeout shouldn't necessarily HALT the whole loop

Even with #1 fixed, a `status="failed"` validation report marks the trial
`FAILED`, and one FAILED trial stopped the run. Decide whether a validation
*timeout* (vs a correctness failure) should fail-soft: record it, keep the
`eval.test.auc`, and let the loop continue.

## Don't forget

- Add a unit test: feed the handler a `TimeoutExpired(stderr=b"...")` and assert
  the report serializes.
- Relates to [[automl-agent-logging-decisions]] and
  `docs/to-do/runner-best-effort-visibility.md`.
- Delete this file once landed.
