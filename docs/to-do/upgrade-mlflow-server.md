# Upgrade shared MLflow server to ≥ 3.12

**Ask for the platform team:** upgrade the shared MLflow tracking server
(`blue.hellobrigit.com`) to **≥ 3.12.0**. Below 3.12, downloading a *missing*
artifact returns a retryable HTTP 500 instead of 404, stalling clients for
minutes per miss (fixed upstream in
[mlflow/mlflow#22310](https://github.com/mlflow/mlflow/pull/22310), released
2026-05-04 in 3.12.0).

Client-side mitigations already shipped (seam-wide retry cap = 1, list-first
`download_artifact`) and stay in place regardless.

**After the upgrade:** run the verification in
[`agent-observability-follow-ups.md`](agent-observability-follow-ups.md) §2
(read-only probe: a missing-artifact download should now fail fast with 404 on
our actual backing store).
