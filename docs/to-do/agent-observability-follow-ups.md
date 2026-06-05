# Agent observability follow-ups

## 0. Pending: end-to-end validation of the 2026-06-02 change set

The change set below is unit/integration/contract-tested (478 green) and the
hang fix is verified against production in isolation (`automl experiment
proposer-context`: >120s hang → ~28s). The **full-loop validation via
`3.1_run_agent_automl.ipynb` could not be completed on 2026-06-02**: the Claude
API was degraded that evening (manager-session requests hung ~16 min/attempt
with `api_error: Request timed out`, retried — ~112 min of pure API wait in
one run). Re-run the notebook on a healthy API day and check, in MLflow:

| change | what to verify on the trial / session artifacts |
|---|---|
| Retry cap + list-first downloads | session report `step_durations_s["experiment proposer-context"]` is seconds, not minutes; the final loop-context render no longer needs to be killed; no "Downloading artifacts" stalls in the manager transcript |
| Agent narratives | trial run has `agent/proposer/message.md` + `agent/coder/message.md`, full untruncated markdown |
| Unified timing | trial `timing/summary.json` is the canonical timing artifact: high-level phases are `setup`, `proposer`, `proposal_handoff`, `coder_implementation`, `runner`, `coder_report`, `publish`; detailed runner phases live under `phase_details.runner.phases`; session report still has `steps`, `step_durations_s`, `publish_s`, `unattributed_s` |
| Step naming | step names read `"<noun> <verb>"` (e.g. `trial run`) — events recorded before the mid-run fix on 2026-06-02 show flag values instead; ignore those |
| Stall self-diagnosis | `unattributed_s` ≈ session time spent waiting on the model/API; on a healthy run it should be small relative to `total_s` |

Note: the interrupted 2026-06-02 evening run (session `60ce6b31…`) publishes
with the new code at coder-stop, so its artifacts are a first (partial,
API-stalled) sample — the table above should still be checked on a clean run.

Open behavior question spotted during the investigation: materializing a new
dataset version on the dry-run route does not publish a profile
(`datasets/v2_ca9864c0/` had only `source_trace/`), which is what the
proposer-context download tripped on. Decide whether dry-run materialization
should profile, or whether profile-less datasets are an accepted state (the
reader now handles absence gracefully either way).

Two follow-ups from the 2026-06-02 investigation of the `proposer-context`
hang and the missing agent narrative (see `archive/` once landed; the fixes
themselves — seam-wide MLflow retry cap, list-first `download_artifact`
helper, per-trial `agent/<phase>/message.md`, unified step/runner timing in
the session report — shipped with that change and are documented by the code).

## 1. Manager end-of-run summary placement

The proposer and coder closing messages are now logged per trial
(`agent/proposer/message.md`, `agent/coder/message.md`). The **manager's**
end-of-run summary (stop reason, cross-iteration ledger) is still only in the
main-session transcript archived to GCS — it has no per-trial home because one
manager session spans N iterations/trials (the loop is LLM-driven inside a
single Claude session; see `agent-skills/references/loop/protocol.md`).

**Decision (wendao, 2026-06-02):** don't bolt this on now. The
[agent-orchestration redesign](agent-orchestration/) moves the loop out of the
LLM session; once a session maps 1:1 to an iteration/trial, the manager
summary gets the same per-trial treatment as the other phases. Revisit then.

## 2. MLflow `<3.12` missing-artifact 500 — verify after server upgrade

**Known limitation.** Tracking servers below MLflow 3.12 answer a download of
a **missing** artifact on the `mlflow-artifacts:` proxy with a retryable
**HTTP 500** (not 404), and the client's hard-coded retry set treats 500 as
transient. Measured against production (`blue.hellobrigit.com`, 2026-06-02):
default budget ≈ 254s per missing artifact; budget is per-request, so misses
compound. Upstream fixed exactly this in **MLflow 3.12.0**
([mlflow/mlflow#22310](https://github.com/mlflow/mlflow/pull/22310)): missing
artifacts now return 404, which the client does not retry.

Mitigations shipped on our side (stay in place regardless): seam-wide
`MLFLOW_HTTP_REQUEST_MAX_RETRIES=1` and the list-first
`client.download_artifact` helper that treats absence as a normal `None`.

**To do when the platform team upgrades the server to ≥ 3.12:**

- Re-run the read-only probe (download a deliberately-nonexistent artifact
  path on any run) and confirm the failure is now an immediate 404-backed
  error, **on our actual backing store** — the upstream fix touched the
  local-filesystem repo; verify it covers whatever `--artifacts-destination`
  blue uses.
- If confirmed, absence fails fast natively; the list-first helper stays (it
  is still cheaper than a failed download) and no further "status-based"
  handling is needed.

**Platform ask:** filed separately —
[`upgrade-mlflow-server.md`](upgrade-mlflow-server.md).
