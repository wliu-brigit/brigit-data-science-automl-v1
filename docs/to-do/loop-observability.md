# Loop observability: the run should narrate itself

**Status:** to-do (split out of the fraud-pilot feedback dossier, 2026-06-05)

Sibling docs: [`logging-and-observability.md`](logging-and-observability.md)
holds the deferred general logging *design* questions — if that pass happens,
design these concrete wants inside it rather than separately.
[`agent-observability-follow-ups.md`](agent-observability-follow-ups.md) is a
validation checklist for the 2026-06-02 change set, different scope.

**Symptom (lived through):** `experiment run` prints nothing until the
session ends. During the fraud pilot we diagnosed liveness from process
tables, MLflow polling, and `__pycache__` timestamps. Operationally absurd
for runs that take 20-30+ minutes.

**Wants:**

- Stream phase markers to stdout as they happen (data refresh → proposing →
  coding → trial running → eval → logging), so `tail -f` answers "where is
  it?".
- Per-phase timestamps in the timeline artifact (partially exists).
- Optional: a `status` CLI verb reading the live lock/timeline.
- Budget headroom warning: the runner logs `timing.*` per phase and knows
  `per_trial_seconds` — warn when one phase consumed most of the budget
  (e.g. eval at scale), so the timeout is predicted, not discovered
  mid-full-run.
