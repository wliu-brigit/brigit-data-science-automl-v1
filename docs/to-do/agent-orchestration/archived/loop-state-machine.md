# Loop State Machine — Make `--max-iter` Deterministic

## Status

Priority: P1. This is the highest-priority remaining architecture item because
`--max-iter` and failure-stop behavior are cost-control promises, but they are
not yet enforced by deterministic code.

This note is a future-work assessment. It captures the problem, the
investigation done so far, and pointers to the prior implementation that
was inadvertently lost — but it deliberately does **not** prescribe a
solution. Whoever picks this up should re-read the referenced code,
re-verify the call chain, and design the fix.

The investigation findings below were correct at the time of writing
(post-Pass C cleanup, 2026-05-11). Verify they still hold before acting.

## Problem

`automl_config.yaml: budgets.max_iterations` and the `--max-iter` flag
are **not enforced by any deterministic code path**. They are passed
into the rendered context as data, and the LLM is then trusted (via
SKILL.md prose) to count iterations against the budget and stop.

There is no Python-side iteration counter, no hook-side hard stop, and
no programmatic guard before the proposer agent runs. If the LLM
miscounts, gets distracted, or misinterprets a failed trial, the loop
can run too few or too many iterations.

This matters because each trial is real compute (GCS reads, model
training, MLflow logging). An engineer setting `--max-iter 5` is making
a cost-control claim that the system does not actually keep.

## Investigation so far

### Current call chain (verified)

1. `uv run automl run --project X --max-iter 5` is parsed in
   `automl/cli/run_loop.py`. The Python code **spawns one Claude
   subprocess** (`run_loop_main` → `subprocess.run(launch.command, ...)`,
   `automl/cli/run_loop.py:237-242`) and exits when that subprocess
   exits. There is no Python-side loop.
2. The spawned Claude session invokes `/brigit-automl:automl run …`,
   which loads `skills/automl/SKILL.md`.
3. `skills/automl/scripts/preflight.py` parses `--max-iter` and emits
   it as `max_iterations` in the rendered context JSON
   (`preflight.py:61`, `preflight.py:124`).
4. `skills/automl/scripts/render_context.py` includes that value in the
   packet the agent sees.
5. `skills/automl/SKILL.md` step 8 instructs the LLM, in prose, *"For
   each iteration, use the `automl-proposer` agent to produce either a
   stop JSON or a complete TrialProposal"*.
6. `references/loop/protocol.md` documents the stop conditions
   (including `iterations >= budgets.max_iterations`). This is
   **documentation the LLM reads**, not code that runs.

A targeted grep across `automl/`, `skills/`, and `agents/` finds **zero**
call sites of any `iterations >= max_iter` style check. The only true
hard stops in the system today are:

- `--max-budget-usd` enforced by the `claude` binary (dollar cost)
- Wall clock (also LLM-checked, no programmatic enforcement)
- User Ctrl-C
- The OS killing the process

### History — there was a state machine, and it was removed

A working iteration-state machine **existed** in the past. The relevant
git history (all on the active worktree):

- **`4bb20dd`** — original commit *"feat(automl): loop_state.py for
  resilient resume"*. Introduced `scripts/loop_state.py` with
  subcommands `init`, `tick --status`, `read`, and `should-continue`.
  The `should-continue` subcommand returned exit code 0 to continue and
  exit code 1 to stop — designed for use in a shell loop. It checked
  iteration count, consecutive-failure count, and wall-clock budget
  deterministically against
  `<project_root>/.cache/automl/tmp/loop_state.json`.
- **`b65f7e6`** (2026-05-01, *"refactor(data): remove local learning
  state"*) — relocated `loop_state.json` from project root to
  `.cache/automl/tmp/` and softened the docstring from "persistent" to
  "scratch", clarifying that **MLflow is the durable state**. The
  script was retained and still wired in.
- **`1b4933f`** (recent, *"refactor: replace commands with skills"*) —
  the `commands/ → skills/` reorganisation. The new SKILL.md prose did
  not carry forward the calls to `scripts/loop_state.py`. The script
  was orphaned from this point on but not deleted.
- **`a8fc946`** (*"Organize pytest tiers and add live QA E2E fixture"*)
  — formalised the retirement by adding `<project_root>/loop_state.json`
  to the forbidden-tokens list in
  `tests/contracts/test_phase_b_retired_scripts.py`.
- **2026-05-11 cleanup pass** — `scripts/loop_state.py` was deleted as
  an "orphan" during a scripts-directory cleanup, because no current
  caller referenced it. In retrospect this was deleting the
  *implementation* of a *missing feature*. The contract test was
  updated to add `scripts/loop_state.py` itself (not just its JSON
  output) to `RETIRED_PATHS`. **That entry should likely be reverted**
  when this work is picked up.

To recover the prior implementation:

```bash
git show 4bb20dd:scripts/loop_state.py        # original
git show b65f7e6:scripts/loop_state.py        # post-scratch-relocation
```

Both versions are short (~80 lines). They are useful as a reference for
shape and naming, but the right home today is almost certainly a
package module (`automl/session/loop_state.py` or `automl/runner/loop.py`),
not a top-level script — per the three-tier convention in
`automl_dev/CLAUDE.md`.

### What the right design should preserve

These constraints are load-bearing and should not be relitigated as
part of this work:

- **MLflow remains the durable state.** Iteration count, trial status,
  and learnings live in MLflow runs under the routed experiment, not in
  local JSON. The state machine reads from MLflow (or from a thin
  scratch file kept in sync), it does not replace it.
- **Resume semantics must work.** `automl run --project X` invoked a
  second time should pick up where it left off. The previous design
  read the existing `loop_state.json` on `init` and continued counting.
  An MLflow-first design would count successful trials via the MLflow
  client and use that as the iteration counter.
- **Local scratch is coordination-only.** Per
  `automl_dev/CLAUDE.md`, anything under `.cache/automl/tmp/` is
  ephemeral and not authoritative. If a `loop_state.json` returns, it
  should only be a cache of MLflow-derivable values.

## Open questions for the implementer

1. **Where does the loop actually live?** Three plausible shapes — each
   has tradeoffs the implementer should weigh against the user-facing
   feel of `/brigit-automl:automl`:
   - Python-side loop in `automl/cli/run_loop.py` that spawns a
     fresh single-iteration Claude session per iteration. Hard
     guarantee; pays Claude startup cost per iteration; changes
     SKILL.md from "loop until done" to "do one iteration."
   - Hook-side hard stop: a `SubagentStop` hook on `automl-coder`
     reads the MLflow count and writes a `STOP_NOW` marker. SKILL.md
     checks the marker at the top of each turn. LLM still drives, but
     the hook is the deterministic backstop.
   - Pre-proposer stop-check CLI verb (`automl loop-context check-stop
     --max-iter 5`) that returns `{"should_stop": true, "reason": …}`
     based on MLflow counts. SKILL.md calls it before each proposer
     turn. Still LLM-mediated in that the LLM must invoke the check,
     but the check itself is deterministic.

2. **What counts as "an iteration"?** The historical `loop_state.py`
   counted every `tick` regardless of status. The `protocol.md` stop
   condition reads `iterations >= max_iterations` without saying
   whether failed/timeout trials count. Decide and document.

3. **How does this interact with `consecutive_failures`?** The original
   `should-continue` also stopped on 3 consecutive failures. That
   behavior is currently also pure SKILL.md prose. Should the new state
   machine own it too?

4. **What does the MLflow-derived counter actually query?** Likely
   `len([r for r in get_trial_summaries(...) if r.status == "success"])`
   for the iteration count, plus a tail check for consecutive failures.
   Verify the existing `automl.mlflow.store` helpers can do this
   without an additional MLflow round-trip per iteration.

5. **What hooks does the `claude` binary already provide that could
   short-circuit the agent?** Worth investigating before introducing a
   custom marker-file mechanism.

## Files to read before designing

- `automl/cli/run_loop.py` — current entrypoint, the one-subprocess
  spawn pattern.
- `skills/automl/SKILL.md` — current LLM-driven protocol, steps 1-16.
- `skills/automl/scripts/preflight.py` and `render_context.py` — how
  `--max-iter` reaches the agent today.
- `references/loop/protocol.md` — the documented stop conditions
  (currently aspirational, not enforced).
- `automl/mlflow/store.py` — has `get_trial_summaries`,
  `ensure_project_overview`, etc. that a deterministic counter would
  reuse.
- `tests/contracts/test_phase_b_retired_scripts.py` — currently
  forbids the prior state-machine artifacts. Will need to relax.
- Git history for `scripts/loop_state.py` — referenced above.
