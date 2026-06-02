# Per-Trial Timeout Policy

The runner enforces `RUN_CONFIG.per_trial_seconds` from
`projects/<project_name>/config.py` using
`signal.SIGALRM` raised inside the runner process. On timeout, the alarm fires
and `result["status"]` is set to `"timeout"`.

## Linux-only

`signal.SIGALRM` is a POSIX signal not available on Windows. brigit-automl v0.1.0
targets Linux (Saturn Cloud, Ubuntu, macOS). If Windows support becomes a
requirement, replace this with a process-group timeout wrapper.

## Why not also a subprocess timeout?

Earlier drafts also wrapped the evaluation subprocess in a `subprocess.run(..., timeout=120)`.
This was removed because the inner timeout (120s) was inconsistent with the
SIGALRM-based per-trial budget (default 600s). With one canonical guard, there's
no inconsistency.
