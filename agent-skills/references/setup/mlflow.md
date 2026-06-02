# MLflow setup

## What is this

An MLflow tracking server where every trial gets logged: parameters, metrics,
artifacts, model code. The single source of truth for trial history, the
leaderboard, and resume context.

## Why we need it

The AutoML loop reads MLflow each turn to decide what to try next, what to
seed from, and when to stop. Without it the loop has no memory between
trials.

## How to set it up

You have two options depending on whether you're developing locally or
running against a shared/team MLflow.

### Option A — Local MLflow (fastest for development)

If a local MLflow server is bundled with your workspace, start it:

```bash
mlflow_local start
```

Then point the workspace at it. In `.env`:

```env
MLFLOW_TRACKING_URI=http://127.0.0.1:54321
```

Also set credentials. Local servers have no auth, but the client requires some
value:

```
MLFLOW_TRACKING_USERNAME=local
MLFLOW_TRACKING_PASSWORD=local
```

### Option B — Shared / team MLflow

Get the tracking URI and credentials from your platform team (these are
usually stored in a team password manager and rotated periodically).

In `.env`:

```
MLFLOW_TRACKING_URI=<your team's MLflow URL, no trailing slash>
MLFLOW_TRACKING_USERNAME=<from platform team>
MLFLOW_TRACKING_PASSWORD=<from platform team>
```

## Common gotchas

- **`401 Unauthorized` on a URL that works in the browser.** Usually a stale
  password. Pull the latest from your team's password manager.
- **Trailing slash on the URI.** Some MLflow clients reject
  `https://example.com/`. Use the version without the trailing slash.
- **`.env` not loaded.** AutoML reads `.env` at the project root. If you put
  it somewhere else, the client falls back to whatever's in the shell, which
  is usually nothing.
