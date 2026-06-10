# Serving-validation robustness — design

**Status:** findings captured (2026-06-10); the core design decision (fail-soft
vs halt) is **open** and is the main thing the next session must settle before
writing `plans/`. Recommendations below; the **code is the source of truth** —
treat this as best-effort framing.

---

## 1. Where `main` is today

`automl/runner/serving_validation.py` on `main`:

- Runs validation via `subprocess.run(..., capture_output=True, text=True,
  timeout=_VALIDATION_TIMEOUT_S)` with `_VALIDATION_TIMEOUT_S = 120` (a hardcoded
  module constant).
- On `TimeoutExpired`, builds a failure report — but the stderr-tail line keeps
  `exc.stderr` as **bytes**, so `json.dumps(report)` raises `TypeError` and the
  handler crashes (the bug).
- Has **no handling** for a subprocess that dies on a signal (`returncode < 0`):
  a native crash produces no report, no tag.

**Not on `main`** (sit on the `neobank_ncm_v3_replicate` branch, *not* ported —
re-implement or cherry-pick deliberately):
- `_decode_tail()` — decodes the bytes tail (fixes the crash).
- `serving_validation_seconds` as a `RUN_CONFIG` field (configurable timeout),
  read via `active.config.serving_validation_seconds`, default 120.
- The timeout error message references the configured value.

These two are known-good and low-risk; the next session can lift them as the
first step.

## 2. The three problems (ranked)

1. **Silent native crash (the real robustness gap).** A signal-killed validation
   subprocess (`returncode < 0`, e.g. -11 SIGSEGV) leaves the trial with no
   result JSON, no MLflow run, no error tag. The runner should detect
   `completed.returncode < 0` on the **completed** `subprocess.run` result (the
   non-timeout path) and synthesize a failure report + tag, the same way the
   timeout path does once fixed. Without this, *any* native crash in validation
   is invisible. See [`finding-nn-pyfunc-sigsegv.md`](finding-nn-pyfunc-sigsegv.md).
2. **Timeout handler crash + tight cap.** Fixed by `_decode_tail` + configurable
   `serving_validation_seconds` (above). See
   [`finding-timeout-crash-and-halt.md`](finding-timeout-crash-and-halt.md).
3. **One FAILED validation halts the loop.** The open policy decision (§3).

## 3. The open decision: fail-soft vs halt

A validation *timeout* or *native crash* is not the same as a *correctness*
failure (predictions diverging from training). Options:

- **A — fail-soft for non-correctness failures:** record the failure report +
  tag, **keep** the trained model and its already-logged `eval.test.auc`, mark
  the trial e.g. `FINISHED_WITH_VALIDATION_WARNING` (or similar), and let the
  loop continue. Correctness mismatches still hard-fail.
- **B — keep halting**, but make sure the failure is always *visible* (problems
  1–2 fixed) so it's at least diagnosable.

Recommendation: **A**, but it's a real product call — fail-soft means a model
that passed training but failed serving validation can still enter the
leaderboard, which has implications for what "a trial succeeded" means.

**Cross-layer note / caveat:** "halt the loop" is decided **above** the runner —
the agent/loop manager stops on a FAILED trial. So fail-soft likely touches the
**agent/loop layer**, not just `serving_validation.py`. Scope this before
planning; don't assume it's a one-file change.

## 4. Caveats / what'll bite you

- **`returncode` sign convention.** On POSIX a signal-killed child gives a
  negative `returncode` (e.g. -11). Confirm what `subprocess.run(check=False)`
  reports here and that `capture_output` didn't already consume the crash — the
  child may die *mid-write* of its own report.
- **The NN env mitigation is a separate axis.** `OMP_NUM_THREADS=1`,
  `MKL_NUM_THREADS=1`, `KMP_DUPLICATE_LIB_OK=TRUE`, `torch.set_num_threads(1)`
  may *prevent* the SIGSEGV; the `returncode < 0` guard makes it *visible*. Want
  both — they solve different halves. The env mitigation belongs in the
  trial/validation env, and partly in NN project `model.py`.
- **Don't conflate `per_trial_seconds` with `serving_validation_seconds`.**
  `per_trial_seconds` is **advisory, not enforced** (no SIGALRM anywhere — see
  `docs/to-do/tiny_per-trial-seconds-not-enforced.md`); only
  `serving_validation_seconds` is a real `subprocess.run(timeout=)`.
- **The bytes-decode bug needs a regression test** — feed the handler a
  `TimeoutExpired(stderr=b"...")` and assert the report serializes.

## 5. References

- Migrated findings (this folder): `finding-timeout-crash-and-halt.md`,
  `finding-nn-pyfunc-sigsegv.md`.
- Sibling effort: [`../dataset-read-reliability/`](../dataset-read-reliability/).
- Left in `to-do/`: `runner-best-effort-visibility.md` (related visibility theme).
- Memory: `automl-agent-logging-decisions`, `opus-headless-loop-vpn-gcs`.
