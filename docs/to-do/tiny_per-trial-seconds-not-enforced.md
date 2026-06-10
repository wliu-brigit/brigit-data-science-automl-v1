# To-do (tiny): `per_trial_seconds` is declared but not enforced

**Status:** noted, not started. Decided to leave as-is for now (wendao,
2026-06-10) — a hard kill could murder a legitimately-long full-data fit. This
file just records the truth so the docs stop lying about it.

## What's wrong

`RUN_CONFIG.per_trial_seconds` (config.py) reads like a hard per-trial timeout,
and two docs claim the runner enforces it:

- `agent-skills/references/loop/timeouts.md` — *"The runner enforces
  `RUN_CONFIG.per_trial_seconds` ... using `signal.SIGALRM` raised inside the
  runner process. On timeout, `result["status"]` is set to `"timeout"`."*
- `agent-skills/agents/automl-coder.md` — *"Per-trial timeout is enforced
  inside the runner from `RUN_CONFIG.per_trial_seconds`."*

Neither is true. Grep `automl/` for `SIGALRM` / `signal.alarm` / `setitimer` —
nothing. `per_trial_seconds` is only *read* and passed to the proposer as an
advisory constraint string (`agent-skills/skills/automl/SKILL.md`). The only
hard cap on a trial today is the **Bash-tool timeout** of whatever agent runs
`run.py` in the loop (Claude Code's Bash tool maxes at 600s) — which happens to
match the default, hiding the gap.

Contrast: the **serving-validation** timeout *is* a real
`subprocess.run(timeout=...)` and is now configurable via
`RUN_CONFIG.serving_validation_seconds` (added 2026-06-10).

## Two ways to make it honest (pick later)

1. **Cheapest:** correct the two docs to say `per_trial_seconds` is an advisory
   budget surfaced to the proposer, not a hard kill. No code change.
2. **Actually enforce:** wrap the fit/run in a `signal.SIGALRM` (POSIX-only,
   matches the original design) or a process-group timeout, and set
   `result["status"] = "timeout"` when it fires. More invasive; risks killing a
   healthy long fit on full data — which is exactly why it's deferred.

## Don't forget

- If you enforce it, add a unit test that a trivial over-budget run reports
  `status="timeout"` cleanly (no crash), and mirror the validation timeout's
  fail-soft behavior.
- Relates to `agent-skills/references/loop/timeouts.md` and
  [[automl-agent-logging-decisions]].
- Delete this file once the docs are fixed (option 1) or enforcement lands
  (option 2).
