# Timeout Policy

Two time knobs exist on `RUN_CONFIG` (`projects/<project_name>/config.py`),
and they are different kinds of thing:

- **`per_trial_seconds` — an advisory budget, not a hard kill.** Nothing in
  the runner enforces it (no `SIGALRM`, no subprocess timeout). It is read and
  surfaced to the proposer/coder as a planning constraint ("design a trial
  that fits this budget"). The only hard cap on a whole trial today is
  whatever tool timeout the invoking agent applies to `run.py`. This is
  deliberate: a hard kill could murder a legitimately-long full-data fit
  (decided 2026-06-10).
- **`serving_validation_seconds` — a real, enforced timeout** (default 300).
  The post-fit serving-validation subprocess runs under
  `subprocess.run(timeout=...)`; on expiry the trial records a serializable
  failure report and a `validation.status=failed` tag, keeps the trained
  model and its eval metric, and the loop continues (fail-soft).

If per-trial enforcement is ever wanted, see
`docs/execution/trial-reliability/design.md` and mirror the validation
timeout's fail-soft shape — record `status="timeout"` cleanly, never crash
the handler.
