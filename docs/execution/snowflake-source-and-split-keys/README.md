# Snowflake source & split keys — design dossier

**Front door.** This folder holds the design for making `SnowflakeSource` real —
the warehouse entry point that the repo's principles already name canonical —
together with the key/split contract cleanup the work surfaced.
**[`design.md`](design.md) is the single active doc — start there.**

This README is intentionally **thin and decision-free**: it states the
*problem* and *what we're looking for*, nothing about *how* we solve it. The
"how" (and every settled decision with its rationale) lives in
[`design.md`](design.md).

**Status:** APPROVED 2026-06-04 after four review rounds + final alignment
pass. Implementation not started — next session begins step 1 of the plan in
[`design.md`](design.md) §14.

## The problem (high level)

- `SnowflakeSource` is a stub: it has the constructor shape and a pending
  `NotImplementedError`, while `projects/fraud_anomaly_detection/` (a ~100K-row
  internal dataset) is waiting on it. A past implementation existed and was
  lost in a repo migration; its conventions survive only as fragments
  (scaffold SQL templates, doc phrases, a dead `--refresh-source` flag).
- Designing it exposed contract debt in the data layer itself: one `hash_key`
  field doing two unrelated jobs (row identity vs. split bucketing), a
  row-fallback split path that silently breaks the eval layer, a
  row-order-sensitive dataset identity that a warehouse source would trip
  over, and a split column (`SPLITID`) whose name says nothing.

## What we're looking for (goals)

Not solutions — the bar the design should clear:

- Snowflake becomes the **main path** for internal projects, not an escape
  hatch: project owns SQL, harness owns boilerplate, GCS stays the home of the
  materialized bytes.
- **One pipeline, dumb sources**: no per-source forks in split, validation, or
  identity logic.
- Unchanged warehouse data **re-attaches** to its existing snapshot instead of
  minting duplicates.
- Refresh is **explicit**: nothing re-queries the warehouse or rebuilds the
  base table unless asked; an experiment keeps its pinned snapshot.
- Keys are **named for their jobs** and validated at the ingestion edge, so
  the eval layer downstream is safe by construction.
- Flexible/time-based splitting
  ([`../../to-do/time-based-splitting.md`](../../to-do/time-based-splitting.md))
  is **in scope** as the final step: a split should be a named, serializable
  row-criterion over the immutable dataset — declarable on time, buckets, or
  any column — recorded with every trial.
