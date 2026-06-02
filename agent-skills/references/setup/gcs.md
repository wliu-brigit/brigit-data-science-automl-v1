# GCS setup

## What is this

A Google Cloud Storage bucket where AutoML keeps the datasets, validation
fixtures, and agent timeline files that go with each run. Same bucket across
dry-runs and real runs — just different prefixes inside it.

## Why we need it

MLflow holds the *durable state* (run records, metrics, model artifacts), but
the heavy bytes — parquet datasets, validation prediction tables, raw hook
events — live in GCS. Keeping them out of MLflow keeps the tracking server
fast and keeps artifact retrieval cheap.

## How to set it up

1. **Pick a bucket** you have read+write access to. The bucket and top-level
   prefix go in `.env` as `GCS_BUCKET` and `GCS_PREFIX`.

2. **Authenticate once per machine** using Application Default Credentials:

   ```bash
   gcloud auth application-default login
   ```

   This opens a browser, logs you in, and writes the credential to
   `~/.config/gcloud/application_default_credentials.json`. AutoML reads it
   from there automatically.

   If you don't have `gcloud`:
   - macOS: `brew install --cask google-cloud-sdk`
   - Linux: <https://cloud.google.com/sdk/docs/install>

3. **Set the GCS env vars** in `.env`:

   ```
   GCP_PROJECT=<the GCP project that owns the bucket>
   GCS_BUCKET=<the bucket name>
   GCS_PREFIX=automl
   ```

AutoML writes to
`gs://${GCS_BUCKET}/${GCS_PREFIX}/<project_name>/<experiment_id>/...`.

## Common gotchas

- **Permissions.** The user you log in as needs read + write on the bucket.
  The simplest correct role is `roles/storage.objectUser` — it bundles
  create/read/delete/list. If you hit a `403 storage.objects.get` error after
  a successful write, you probably have the older `legacyBucketWriter` role
  (writes work, reads don't). Ask your GCP admin to grant `objectUser`
  instead.
- **Wrong project.** ADC picks up the project from `gcloud config`. If you
  work across projects, set `GCP_PROJECT` in `.env` explicitly — it overrides
  the gcloud default.
