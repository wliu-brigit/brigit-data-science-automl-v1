# Logging & Observability

## Status

Priority: P3. Deferred, needs deliberate design. Not blocking anything. **Decision
(2026-05-29): keep `automl/utils/logging.py::configure_logging` — do NOT delete it,
and do NOT wire it up in the current structural-cleanup pass.** Logging deserves a
proper observability design of its own; bolting it on mechanically now would
pre-empt that thinking.

This is a future-work assessment captured on **2026-05-29** (branch
`refactor/four-layer`). It is **not authoritative** and does not prescribe a
solution — it records why logging is being left alone for now and the questions a
future pass should answer.

## Problem

`utils/logging.py::configure_logging(name, *, level=INFO)` exists and is a clean,
AutoML-agnostic helper, but it has **no callers** — nothing in the package
configures logging, and domains do not consistently use `logging.getLogger`. A
structural review flagged it as "dead," but it is better understood as
**legitimate-but-unwired infrastructure**: a real capability we may want, simply
not connected yet. Per the keep-don't-delete principle, it stays.

Today the system's actual observability lives elsewhere:
- the agent timeline reconciled into MLflow (`agent/timeline.py`),
- MLflow metrics/tags/artifacts as the durable record,
- runner stdout markers (`AUTOML_STATUS=`, `AUTOML_PRIMARY=`, `AUTOML_RUN_ID=`),
- the compact main-conversation ledger.

So "should we have process logging, and how does it relate to all of the above?"
is a genuine design question, not a wiring chore.

## Questions a dedicated pass should answer (not decided)

- **Strategy:** levels, format, structured vs plain text. Who owns the format?
- **Where `configure_logging` is called:** CLI dispatcher entry? agent launch? the
  per-trial runner subprocess? (The loop spans process boundaries — manager
  subprocess, runner subprocess — so "configure once" is not obvious.)
- **Relationship to existing signals:** does logging duplicate or complement the
  stdout markers, the agent timeline, and MLflow artifacts? Avoid a fourth,
  redundant observability channel.
- **Durable vs ephemeral:** do we want structured logs persisted (GCS?) per
  session/trial, or is logging purely local/dev-time while MLflow stays the
  durable record?
- **`dry_run` / namespace:** how (if at all) does logging reflect the universe?
- **Noise control:** library-as-import (notebooks) must not get spammed by
  `basicConfig` side effects.

## What NOT to do

- Do not delete `configure_logging`.
- Do not wire it into the cleanup waves as an afterthought.
- Do not introduce ad-hoc `print`/`logging` calls across domains before the
  strategy is set.
