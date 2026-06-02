# Phase 0-7 Spec Coverage Review

Review date: 2026-05-28. Baseline reviewed: final audit commit `18f8bbe` plus this follow-up
documentation correction.

This file is the handoff ledger for "what is covered" versus "what is still left out" across
approved specs `00`-`11`. It does not replace the specs; it records the current implementation
coverage and the explicit gaps a future session should triage before merge.

## Executive Status

- Phase 0-7 implementation is committed and the final audit gates passed at commit `18f8bbe`.
- Migration checklist implementation rows are dispositioned as `[x]` or `[-]`.
- Architecture/import ratchets were green at final audit: no `automl_legacy` tree/imports, and
  PyPI `mlflow` imports are isolated to the MLflow seam.
- The branch is **not merge-ready yet**. See `final-review-open-items.md`.

## Spec Matrix

| Spec | Covered By | Current Coverage | Known Exceptions / Deferred Scope |
|---|---|---|---|
| `00-structural-design.md` | Phase 0 scaffold, Phase 1 skeleton, Phase 6 CLI, Phase 7 cutover, contracts | Four-layer package exists; six noun domains are active; old legacy trees are deleted; CLI is noun-first; architecture ratchets pass. Follow-up fixes route project Dataset artifacts through the namespace/dry-run project route and consume `DataSource.artifact_files()` for source trace logging. | Standalone `experiment create/archive` remains deferred. Stage-runner north-star remains deferred. |
| `01-project-context.md` | Phase 1 session/config, Phase 6 root flags/isolation | Session binding, project root, dry-run, namespace, and experiment selection flow through the CLI/library instead of route-string parsing. | Process-global convenience remains intentionally small; external gates still depend on loading the worktree `.env`. |
| `02-mlflow-seam.md` | Phase 1-7 seam work, contracts, import ratchets | Domain code routes MLflow through `automl/mlflow`; experiment/trial/data/eval helpers cover accepted gates; search pagination is implemented in seam queries. Follow-up fixes implement generic project-scoped `log_json()` and project-level source trace artifact logging. | Project learning artifacts are dropped; no-caller analytics remain deferred; multi-process overview coordination remains deferred. |
| `03-cleanup.md` | Phase 4 cleanup, Phase 6 namespace/dry-run, Phase 7 final harness | Namespace-scoped cleanup and dry-run isolation were verified by external gates; cleanup tests cover deletion behavior. | Broader operational policies beyond the accepted namespace/dry-run gates are not implemented here. |
| `04-validate.md` | Phase 1 project/model validation, Phase 5 proposal validation, Phase 6 CLI | Canonical validation report/issues live in `validate/base.py`; CLI exposes `validate project`, `validate model`, and `validate proposal`. | Project-specific validator discovery was intentionally kept narrow; no broad plugin registry was added. |
| `05-data.md` | Phase 2 data breadth, Phase 3/7 preservation, data contracts | Local CSV/GCS materialization, dataset identity, registry, profile, partial snapshot guard, idempotent rematerialize, source trace hook consumption, namespace/dry-run Dataset routing, and load-by-id/trial paths are covered by tests/gates. | Live Snowflake loading is stubbed; project learning flags were dropped by design; parquet pushdown/advanced partitioning and richer drift diagnostics remain outside current gates. |
| `06-model.md` | Phase 2 model breadth, Phase 7 source packaging | Model contracts, pyfunc wrapper behavior, and trial source packaging are implemented for accepted paths. | Path-based source recovery only works for new-format runs with `source/model.py`; old Phase 1-6 runs fail clearly if used as seeds. |
| `07-eval.md` | Phase 3 eval breadth, Phase 7 final loop | Eval compute/listing, split-view delegation, model contract integration, leaderboard preservation, and external eval gate are covered. | Full Dataset/EvalDataset substrate unification remains a documented north-star, not part of this refactor. |
| `08-runner.md` | Phase 1 runner, Phase 6 session lock, Phase 7 folder execution | Runner executes the project-model fallback and verified trial folders; metadata/source artifacts are captured for new trial folders. | Stage abstraction/pluggable runner remains deferred; live LLM subprocess behavior is not part of deterministic e2e gates. |
| `09-experiment.md` | Phase 4 reads/cleanup, Phase 5 proposer context, Phase 6 CLI | Experiment listing, run/delete, leaderboard, compare, summary, recent-failure/strategy context, and proposer-context surfaces are implemented. | `runs_using_strategy`, `runs_in_metric_band`, standalone create/archive, and learning counts remain deferred/dropped. |
| `10-trial.md` | Phase 4 trial reads, Phase 7 authoring/cutover | Trial list/show/delete/lock/create/fork/promote/run surfaces exist; source artifacts are written for new-format folder runs. | Older runs without source artifacts are not backfilled; source recovery intentionally fails clearly. |
| `11-agent.md` | Phase 5 agent/hook loop, Phase 7 final harness | Proposer context, proposal validation, loop launch spec, timeline reconciliation, and hook routing are implemented and externally gated. | Learning reads/artifact-error aggregation are dropped; multi-agent driver abstractions and timeline file splitting remain future work. |

## Potential Pre-Merge Blockers

1. **Snowflake support.** Not a spec surprise, but it is project-dependent. If any cutover project
   uses Snowflake, live loading/source trace work is required.

## Deferred But Spec-Approved

- Project learning subsystem, golden/weak flags, learning cache, and `learning_counts`.
- No-caller analytics `runs_using_strategy` and `runs_in_metric_band`.
- Standalone `experiment create/archive` verbs.
- Stage-runner/pluggable-runner north-star.
- Dataset/EvalDataset substrate unification north-star.
- Multi-process overview write coordination beyond current serial gate needs.

## Verification Evidence From Final Audit

- `uv run pytest -v` -> `283 passed, 7 skipped, 2 warnings`.
- `uv run pytest tests/contracts tests/integration/cleanup/test_experiment_delete.py -v` ->
  `16 passed, 2 warnings`.
- `uv run ruff check automl hooks skills agents references tests projects/payment_routing/config.py projects/example_homecredit/config.py projects/example_homecredit/model` ->
  `All checks passed`.
- `git diff --check` -> clean.
- External gates with `.env` loaded and `MLFLOW_TRACKING_URI=http://127.0.0.1:54321`:
  Phase 7/6/5/4/3 each -> `1 passed`.
