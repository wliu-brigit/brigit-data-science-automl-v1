# AutoML Loop Protocol

The main `automl` skill alternates proposer and coder agent turns until a stop condition triggers.

## One iteration

```
Main skill turn
    └─ Claude hooks record proposer start/end in agent_timeline.py
    └─ automl-proposer agent → Proposal JSON OR {"action": "stop"}

Coder turn
    └─ Claude hooks record coder start/end in agent_timeline.py
    └─ automl-coder agent receives the validated Proposal, project_root,
       allowed_dependencies, and data_context
    └─ agent writes model.py, runs the trial, writes durable trial state to
       MLflow, and returns a compact summary
    └─ coder SubagentStop hook publishes canonical per-trial timeline artifacts
       to the active MLflow trial run under `agent/`

Main session: read MLflow context; check stop conditions; iterate.
End of session: safe_commands.timeline_publish reconciles the session summary,
raw hook events/transcripts, and any missing per-trial timeline artifacts. Trial
observability should not depend exclusively on this final reconciliation step.
```

Ensembles are opt-in for latency-sensitive runs. The proposer must not return
`strategy: "ensemble"` unless the user explicitly asks, project instructions
explicitly permit latency-heavy models, or constraints state online latency is
not a concern.

## Stop conditions (any one ends the loop)

- `iterations >= budgets.max_iterations`
- `wall_clock >= budgets.total_session_hours`
- 3 consecutive failed trials (likely contract bug)
- User interrupt
- `missing_dependency` failure -> halt + notify (user installs, resumes by
  re-running `/brigit-automl:automl experiment run --project <project_name>`)
- `automl-proposer` returns `{"action": "stop"}`

## Resumability

MLflow is the durable state. Re-running `automl --project <project_name> experiment run`
reads compact MLflow context, increments the trial counter, and picks up where
it left off. Trial scratch stays local; loop decisions come from MLflow context.
GCS stores immutable large/debug artifacts such as dataset bytes,
validation fixtures, raw agent hook events, and compressed Claude transcripts.

## Steering

`projects/<project_name>/PROJECT_INSTRUCTIONS.md` is reloaded fresh each Manager
turn. The user can edit it mid-run; the next iteration picks up the change.
