# Publishing an experiment-level learning

When a run, study, or analysis produces a durable takeaway about *this
experiment* — what worked, what didn't, what surprised us — record it as an
**experiment learning**. This guide is the format and the flow; it is generic
and ships with every project.

## Where learnings live

**MLflow is the durable home.** A learning belongs to the experiment's record —
or, when it generalizes across experiments, the project's. That is the source
of truth the AutoML loop reads back on later runs. Anything you write locally
while producing a learning is **scratch** (working notes, scripts, intermediate
tables): where you keep it doesn't matter and isn't prescribed — only the
promoted learning in MLflow is durable.

## Lifecycle — draft, confirm, approve

1. **Draft** while you investigate (scratch, anywhere).
2. **Confirm** against noise — ideally reproduced (a seed re-fit, a second time
   window) before you trust it.
3. **Approve** the confirmed learning into the experiment's MLflow record
   (`status: approved`); an approved learning that holds across experiments is
   reused at the project level.

## Study format

A learning is one self-contained study answering one question. Begin with a
small header, then the sections in order:

    ---
    schema_version: 1
    status: draft          # draft | approved
    date: YYYY-MM-DD
    ---

    # <the question, as a title>

    **Question.** What the study set out to answer.

    **Setup / methodology.** How it was tested: data, splits, model, what varied.

    **Result.** The measured numbers (a table where it helps).

    **Finding.** What the result means, judged against the metric's noise.

    **Caveat.** Boundaries — what was not tested, confounds, the obvious next step.

    **Evidence & reproducibility.** *(recommended, optional)* How to re-run or
    audit this — the trials in the experiment (each run carries its own model
    code, params, metrics) or a script that regenerates the numbers.

`schema_version` is per-study so the shape can evolve without breaking older
entries; this guide documents version 1.

## Judge against noise

State the noise basis for the experiment's metric and judge every effect against
it: a difference smaller than the metric's noise is reported as noise, not
improvement. If a study uses a fast proxy model for directional comparison, say
so — absolute values may sit below a full model while relative comparisons hold.
