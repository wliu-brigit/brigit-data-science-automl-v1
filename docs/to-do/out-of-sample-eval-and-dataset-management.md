# Out-of-sample evaluation & dataset management

**Status: placeholder (2026-06-03).** Not scoped — needs a deep-dive session
before any design. This note captures the ask and where the current code
stands so that session can start oriented.

## The ask

Two threads that probably want one holistic answer:

1. **Out-of-sample evaluation, integrated.** After training finishes, users
   want to validate the model on a sample that sat *outside* the training
   data. The re-evaluate flow can do this today, but it isn't
   well-integrated — it works, it just isn't easy to understand or obvious
   how to manage.
2. **Dataset management is split-brained.** The pipeline-produced dataset is
   managed at the **experiment** level (one data artifact per experiment).
   Eval datasets are *not* tied to that level — they hang off individual
   trial runs, which is interesting but doesn't feel like the right home.
   And eval datasets work differently by nature: you point at the split view
   with arguments (a `Where(...)` predicate, augmentation) and create a new one, rather than
   materializing through the pipeline.

The current functionality *supports* what users want; the problem is
legibility and management, not capability. The opportunity is to step back
and make out-of-sample evaluation native instead of assembled.

## Current state

- **Pipeline dataset** — materialized once per experiment, snapshot logged
  at the experiment level (`automl/data/`, experiment store in
  `automl/experiment/`).
- **Eval datasets** — `automl/eval/eval_dataset.py`: created either from
  the split view (`EvalDataset.split_view(...)` — a `Where(...)` predicate over the frame)
  or from external data (`EvalDataset.external(...)`), with optional
  `Augmentation`s; identity is content-derived
  (`compute_eval_dataset_identity`). Their records ride on trial runs
  rather than a home of their own.
- **Re-evaluation** — existing flow for scoring an existing model on a
  different eval dataset (see `automl/eval/` and notebook
  `4_reevaluate_existing_model`).

## Questions for the deep dive

- What is the right home for eval datasets — experiment-level alongside the
  pipeline dataset, project-level (shareable across experiments), or their
  own noun with its own lifecycle?
- Should "dataset" become one concept with two creation paths (pipeline vs
  selection/external) instead of two differently-managed things?
- What does a native out-of-sample flow look like end to end: declare the
  holdout up front (at materialization) vs construct it after training (at
  re-eval time)?
- How does this interact with the three-level MLflow hierarchy — where do
  out-of-sample results land so they're comparable across trials?

Related: [time-based splitting](../archive/2026-06-04-time-based-splitting.md) — a time-based
holdout is one kind of out-of-sample evaluation, and both touch the split
view.
