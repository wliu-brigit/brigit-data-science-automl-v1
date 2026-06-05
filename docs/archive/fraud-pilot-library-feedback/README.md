# Library feedback from the fraud_anomaly_detection pilot

**Status:** archived 2026-06-05 — every item resolved the same day, except
two spin-offs promoted to their own to-do entries
([`leaderboard-dataset-pinning.md`](../to-do/leaderboard-dataset-pinning.md),
[`loop-observability.md`](../to-do/loop-observability.md)) and one optional
follow-on ([`split-pct-lowercase.md`](../to-do/split-pct-lowercase.md)).

The fraud project is the library's first real internal use case, run partly to
shake out the harness before wider rollout. Kept as history: each item records
the live symptom from the pilot and where the fix landed.

## 1. Dry-run sampler: sample on a hash orthogonal to SPLIT_PCT

**Symptom (hit live):** first trial died with "eval dataset is empty". The
dry-run sampler filters `WHERE SPLIT_PCT < buckets` — a *prefix of the very
column splits cut on* — so any SPLIT_PCT-based `Splits` gets an empty test
partition in dry-run.

**✅ Fixed (2026-06-05):** `_dry_run_sql` samples on
`MOD(ABS(HASH(<unique_key>)), 100) < buckets`. Same determinism; independent
of SPLIT_PCT, so every split definition keeps its proportions in the sample.
The fraud project's `eval_pct` workaround was deleted in the same change and
its splits returned to plain `SPLIT_PCT` — verified against the warehouse:
dry-run slice 96.4k rows → 80.1/19.9 train/test with positives on both sides.

CSV/parquet sources are unaffected by the structural bug: they truncate to
the *first N rows* and SPLIT_PCT is computed post-load by the pipeline, so
the sample is not correlated with the split column. Their milder issue —
first-N bias when the file is sorted — is noted here for completeness, not
urgent.

✅ The companion bug is already fixed with a regression test: the bucket count
now uses `COUNT(*)` of the *training query*, not the base table (a 39k pull
was sized as 10.7M rows → clamped to one bucket → 386-row "10k" dry-run).

## 2. Coerce Snowflake Decimal columns at the fetch seam

**Symptom (hit live):** a trial whose fit *and* eval succeeded failed at the
final step — MLflow could not JSON-encode the model input example:
`Object of type Decimal is not JSON serializable`. Snowflake types computed
expressions (`HASH`, aggregates, `ROUND`) as high-precision `NUMBER`, which the
connector returns as `decimal.Decimal` objects. Stored columns downcast fine;
computed ones arrive poisoned.

**✅ Fixed (2026-06-05):** `coerce_decimal_columns` in
`automl/utils/io/snowflake.py`, applied inside `fetch_df` — the only place
warehouse data enters the system. Tests in
`tests/unit/utils/test_snowflake_io.py`. (The pilot's interim SQL-side cast
became unnecessary and was removed with the eval_pct cleanup.)

## 3. ✅ Case-insensitive split-predicate resolution

**Symptom (hit live):** `Where("EVAL_PCT")` vs the pipeline's lowercased
`eval_pct` → `KeyError` minutes into a trial. The pipeline normalizes column
names; predicates are hand-written.

**Fixed during the pilot:** `Predicate._resolve_column` now resolves pure case
mismatches (ambiguity still errors, with a clear message). Tests in
`tests/unit/project/test_predicates.py`. A follow-on worth considering: unify
the canonical pandas-side name `SPLIT_PCT` to lowercase so "all columns are
lowercase after the pipeline" holds with no exception (forward-only posture
makes the rename cheap now, expensive later).

## 4. Surface root causes in AUTOML_ERROR; pin leaderboard entries to datasets

Two agent-misdiagnosis incidents:

- ✅ **Fixed (2026-06-05):** the runner wraps exceptions and only the wrapper
  reached the `AUTOML_ERROR=` marker — the agent saw "storage" and diagnosed
  GCS permissions while the Decimal root cause sat in the traceback artifact.
  `automl.errors.format_error_chain` now renders the full `__cause__`/
  `__context__` chain into `TrialResult.error` (tests in
  `tests/unit/test_errors.py`).
- The session ledger compared AP across trials run on *different data
  snapshots* and recorded the comparison as a takeaway. Promoted to its own
  entry: [`../leaderboard-dataset-pinning.md`](../leaderboard-dataset-pinning.md).

## 5. Observability umbrella: the loop is silent while it runs

Promoted to its own entry: [`../loop-observability.md`](../loop-observability.md).

## 6. ✅ Scaffold: project packages mirror the library's domain layout

Project-owned code currently lands wherever the author puts it (the pilot's
metrics ended up at `projects/fraud_anomaly_detection/metrics.py`). Convention
agreed with wendao: a project mirrors the library's domains —

```
projects/<name>/
├── README.md              # front door: layout convention, metadata_cols note,
│                          #   pointer to the pre-built-table recipe (item 7)
├── config.py
├── PROJECT_INSTRUCTIONS.md
├── data/queries/...
├── eval/metrics.py        # custom Metric classes
├── model/preprocessing.py # custom transformers / pipeline overrides
└── tests/                 # project-owned tests, NOT in the core tests/ tree
```

so `from projects.<name>.eval.metrics import ...` reads like the library it
extends. **Mechanism (in order):** `project init` creates the packages with
docstring stubs; the setup reference documents the rule; a contracts test pins
the scaffold shape; one line in CLAUDE.md. Move the fraud project's
`metrics.py` to conform once the scaffold defines the shape.

✅ **Done (2026-06-05), all of it:** `project init` creates `eval/`,
`model/`, `data/`, and `tests/` package stubs plus the README front door;
`tests/contracts/test_project_scaffold.py` pins the shape; `pyproject.toml`
testpaths gained `projects/*/tests`; the fraud project conforms
(`eval/metrics.py`, stubs, tests inside the project).

Also fold in: scaffold comment teaching "columns referenced by splits belong
in `metadata_cols`" (declared by the author — the data pipeline deliberately
does not depend on RUN_CONFIG; rejected auto-registration to keep the
dependency direction clean).

## 7. ✅ Document the pre-built-table recipe (scaffolded README)

The common internal case — "my feature table already exists in Snowflake" —
was undocumented; the pilot assembled the pattern from code-reading.
**Done (2026-06-05), placement per wendao:** not in agent-skills references —
the scaffold now generates a per-project `README.md` (template in
`automl/project/scaffold.py`) carrying the layout convention, the
conventions-that-save-a-debugging-session list, and the full pre-built-table
recipe; the scaffolded `base_table.sql` points at it. The fraud project got
the rendered README. The scaffold also creates `tests/__init__.py` now.
The template is a real file (`automl/project/templates/README.md`), copied
with explicit `{project_name}` replacement, and includes a "Writing
PROJECT_INSTRUCTIONS.md" guide (audience, the don't-restate-config rule,
section-by-section contents, style).

Follow-up: `agent-skills/references/setup/project-instructions.md` overlaps
the README's instructions guide — align it (or have it point at the
scaffolded README) so one canonical teaching text exists.

## 8. ✅ AveragePrecision in the library

PR-AUC is the right primary for any rare-event problem; added to
`automl/eval/metrics.py` with tests during the pilot. Ready to commit.
