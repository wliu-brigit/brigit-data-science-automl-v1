# Feature-pool override without re-materializing

**Status:** parked (idea, not scheduled). Raised 2026-06-11 during the
neobank_ncm business-metric work; we chose a model-side workaround for now and
want to revisit the clean fix later.

## The ask

Let a project narrow the **candidate feature pool** (drop columns from what
trials may train on) by editing config alone, and have it take effect on the
**next loop read** — without re-materializing the dataset and without touching
the system-of-record snapshot on GCS.

## Why it doesn't work today

`DataSpec.exclude_cols` is consumed at **materialize time**, not read time:

- `DataPipeline.run()` builds the registry with `exclude_cols` and flags the
  excluded columns `model=False` (`automl/data/pipeline.py:82`), then **writes
  that registry to GCS** (`pipeline.py:339`).
- The loop **reads the stored registry back** — `load_dataset_by_id` →
  `_read_dataset_files` downloads `feature_registry.csv` and rebuilds via
  `FeatureRegistry.from_dataframe(...)` (`automl/data/registry.py:69-70`,
  `156-199`). It never re-derives flags from `config.py`.

So changing `exclude_cols` only emits a "recipe drift" warning
(`pipeline.py:360-373`); the pinned registry is unchanged. Applying it requires
a re-materialize, which re-runs the pipeline (re-reads the Snowflake source —
VPN, expensive) and **mints a new dataset version** because the registry hash
feeds the dataset identity (`pipeline.py:221-238`). That's the cost we want to
avoid for a pure pool-narrowing change that doesn't alter the stored bytes.

## Workaround in use now (neobank_ncm)

Enforce the deployable pool **model-side**: a project-owned deployable-feature
list, and each trial subsets its `feature_cols` to it. The runner contract
allows any subset of the dataset's columns (`automl/runner/contract.py:113-116`)
and validates the model's own annotated registry against its `feature_cols`
(`contract.py:118-119`). No GCS write, no Snowflake, existing dataset untouched.

## Possible clean fixes (to evaluate later)

- **Read-time pool overlay.** Apply `exclude_cols` (or a new
  `candidate_cols`/pool predicate) as a flag-flip on the loaded registry at
  read time, leaving the stored GCS registry as the immutable record of *what
  exists*. The config expresses *what's eligible this run*; the snapshot stays
  the source of truth for the bytes. Keep dataset identity tied to content, not
  to the eligibility view, so narrowing the pool does not mint a new version.
- **Separate "what exists" from "what's eligible."** Today the registry
  conflates the column inventory with model-eligibility. Splitting them lets
  eligibility move with config while inventory stays pinned.

Either way: decide whether pool changes should be comparable-by-default (same
dataset id) or tracked as a new view id. Note the interaction with the recipe-
drift check and with trial lineage/provenance before implementing.
