# Purging archived MLflow experiments on the production server

**Problem:** from a client machine, `automl.project.cleanup.delete()` archives
and **soft-deletes** experiments/runs on the production tracking server
(`blue.hellobrigit.com`). Soft delete works over the HTTPS API with the normal
`.env` basic-auth credentials, but the archived MLflow record still needs a
backend purge when we want it gone permanently:

- the metadata rows stay in the backend DB (lifecycle stage `deleted`);
- the experiment **name stays reserved** — `mlflow.experiment.lifecycle.ensure()`
  refuses to recreate an experiment with a tombstoned name
  ("deleted; purge it or choose another id").

The library's `automl.mlflow.cleanup.purge(...)` path shells out to `mlflow gc
--backend-store-uri <db>`, which needs **direct database access**. The
production proxy doesn't expose that (there is no REST endpoint for gc), so
purge is a **server-side operation** we currently cannot run from a normal
client machine.

**Ask for the platform team:** how should purge / `mlflow gc` run against
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

Until resolved: treat archive + soft delete as the ceiling from client machines,
and never reuse a deleted experiment name unless it has been moved to a
`deleted/...` archive route or purged. QA namespaces already use unique names by
convention. Related platform ask:
[`upgrade-mlflow-server.md`](upgrade-mlflow-server.md).
