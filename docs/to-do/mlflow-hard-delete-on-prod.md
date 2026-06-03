# Hard-deleting MLflow experiments on the production server

**Problem:** from a client machine, `automl.project.cleanup.delete()` can only
**soft-delete** experiments/runs on the production tracking server
(`blue.hellobrigit.com`). Soft delete works over the HTTPS API with the normal
`.env` basic-auth credentials, but it leaves a tombstone:

- the metadata rows stay in the backend DB (lifecycle stage `deleted`);
- the experiment **name stays reserved** — `mlflow.experiment.lifecycle.ensure()`
  refuses to recreate an experiment with a tombstoned name
  ("deleted; hard-delete it or choose another id").

The library's `delete(..., hard_delete=True)` path shells out to
`mlflow gc --backend-store-uri <db>`, which needs **direct database access**.
The production proxy doesn't expose that (there is no REST endpoint for gc), so
hard purge is a **server-side operation** we currently cannot run.

**Ask for the platform team:** how should hard deletes / `mlflow gc` run against
the shared server? Options to discuss:

- a periodic server-side `mlflow gc` job (cron) that purges soft-deleted
  experiments/runs, or
- a documented manual procedure (backend store URI + artifacts destination)
  someone with DB access can run on request.

**Known tombstones waiting for a purge** (soft-deleted 2026-06-02; GCS bytes
already wiped, only DB metadata remains):

- `qa-expmove/dry_run/example_homecredit/example-homecredit`
- `qa/dry_run/example_homecredit/qa-context-tags-check`
- `qa/dry_run/example_homecredit/qa-user-tag-check`

Until resolved: treat soft delete as the ceiling from client machines, and never
reuse a deleted experiment name — pick a fresh id (QA namespaces already do this
by convention). Related platform ask: [`upgrade-mlflow-server.md`](upgrade-mlflow-server.md).
