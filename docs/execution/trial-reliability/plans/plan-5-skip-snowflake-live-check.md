# `skip_snowflake_live_check` on RunConfig Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The off-VPN bypass for the Snowflake live probe becomes a `RunConfig` run-knob (`skip_snowflake_live_check`, default `False`) read by `validate_project`, with an explicit `probe_snowflake` override that wins over config — instead of operational state bolted onto `SnowflakeSource` (which `main` never had; the source flag exists only on the neobank branch and is retired at rebase).

**Architecture:** Design §8. Two files: the `RunConfig` field, and the gate inside `automl/project/checks.py::snowflake_connection` — env-var and SQL-file checks always run; only the live `SELECT 1` probe is skipped, with a visible warning issue. The flag can never affect dataset identity (identity comes from `compute_recipe`, which reads only `DATA`/source — verified).

**Tech Stack:** Python 3.13, pytest via `uv run`.

**Design:** `docs/execution/trial-reliability/design.md` §8.

---

### Task 1: `RunConfig.skip_snowflake_live_check`

**Files:**
- Modify: `automl/project/run_config.py` (the `RunConfig` dataclass)
- Test: `tests/unit/project/test_run_config.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/project/test_run_config.py` (reuse the `_models()` helper from plan 1 if already present):

```python
def test_skip_snowflake_live_check_defaults_false():
    config = RunConfig(experiment_id="exp", models=_models(), per_trial_seconds=600)
    assert config.skip_snowflake_live_check is False


def test_skip_snowflake_live_check_accepts_true():
    config = RunConfig(
        experiment_id="exp",
        models=_models(),
        per_trial_seconds=600,
        skip_snowflake_live_check=True,
    )
    assert config.skip_snowflake_live_check is True


@pytest.mark.parametrize("bad", [1, "true", None])
def test_skip_snowflake_live_check_rejects_non_bool(bad):
    with pytest.raises(TypeError):
        RunConfig(
            experiment_id="exp",
            models=_models(),
            per_trial_seconds=600,
            skip_snowflake_live_check=bad,
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/project/test_run_config.py -q -k skip_snowflake`
Expected: FAIL — unexpected keyword argument.

- [ ] **Step 3: Implement the field**

In `automl/project/run_config.py`, `RunConfig`: add the field declaration:

```python
    skip_snowflake_live_check: bool
```

the keyword parameter:

```python
        skip_snowflake_live_check: bool = False,
```

the validation:

```python
        if not isinstance(skip_snowflake_live_check, bool):
            raise TypeError(
                "skip_snowflake_live_check must be a bool, "
                f"got {type(skip_snowflake_live_check).__name__}"
            )
```

and the assignment:

```python
        object.__setattr__(self, "skip_snowflake_live_check", skip_snowflake_live_check)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/project/test_run_config.py -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add automl/project/run_config.py tests/unit/project/test_run_config.py
git commit -m "feat(run-config): skip_snowflake_live_check knob (default False)"
```

---

### Task 2: Gate the live probe in `snowflake_connection`

**Files:**
- Modify: `automl/project/checks.py` (`validate_project` signature + the `snowflake_connection` check)
- Test: `tests/unit/project/test_project_validation.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/project/test_project_validation.py` (follow the file's existing fake-config pattern if one exists; this shape is self-contained):

```python
from pathlib import Path

from automl.project.checks import snowflake_connection


class _Source:
    kind = "snowflake"
    base_table_sql = "sql/base.sql"
    training_data_sql = "sql/train.sql"


class _DataSpec:
    source = _Source()


class _RunConfig:
    skip_snowflake_live_check = True


class _SkippingConfig:
    data_spec = _DataSpec()
    run_config = _RunConfig()
    project_dir = Path(".")


def _issues(config, monkeypatch, *, probe=None, sql_exists=True):
    from automl.utils.io import snowflake as sf

    monkeypatch.setattr(sf, "missing_env", lambda: [])
    monkeypatch.setattr(Path, "exists", lambda self: sql_exists)
    calls = []
    monkeypatch.setattr(sf, "check_connection", lambda: calls.append(1))
    found = list(snowflake_connection(config=config, probe=probe))
    return found, calls


def test_config_flag_skips_probe_with_warning(monkeypatch):
    issues, probe_calls = _issues(_SkippingConfig(), monkeypatch)
    assert probe_calls == []
    assert any(
        issue.level == "warning" and "skipped" in issue.message for issue in issues
    )


def test_probe_true_overrides_config_flag(monkeypatch):
    issues, probe_calls = _issues(_SkippingConfig(), monkeypatch, probe=True)
    assert probe_calls == [1]
    assert not any("skipped" in issue.message for issue in issues)


def test_probe_false_skips_even_without_config_flag(monkeypatch):
    class _NoFlagConfig(_SkippingConfig):
        run_config = None

    issues, probe_calls = _issues(_NoFlagConfig(), monkeypatch, probe=False)
    assert probe_calls == []
    assert any("skipped" in issue.message for issue in issues)


def test_env_and_sql_checks_still_run_when_skipping(monkeypatch):
    issues, _ = _issues(_SkippingConfig(), monkeypatch, sql_exists=False)
    assert any(
        issue.level == "error" and "file not found" in issue.message for issue in issues
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/project/test_project_validation.py -q -k probe`
Expected: FAIL — `snowflake_connection() got an unexpected keyword argument 'probe'` (and the flag test finds the probe was called).

- [ ] **Step 3: Implement the gate**

In `automl/project/checks.py`:

1. `validate_project` gains the override and forwards it:

```python
def validate_project(
    *, session=None, live: bool = False, probe_snowflake: bool | None = None
) -> ValidationReport:
```

and the snowflake `run_check` call becomes:

```python
            run_check(
                "project.connections.snowflake",
                snowflake_connection,
                config=config,
                probe=probe_snowflake,
            )
```

2. `snowflake_connection` gains the parameter and gates **only** the live probe (env + SQL-file checks above it stay untouched). Replace the trailing `try: sf.check_connection()` block with:

```python
    run_config = getattr(config, "run_config", None)
    config_skips = bool(getattr(run_config, "skip_snowflake_live_check", False))
    skip_probe = (not probe) if probe is not None else config_skips
    if skip_probe:
        yield Issue(
            level="warning",
            check="project.connections.snowflake",
            message=(
                "Snowflake live probe skipped "
                "(RUN_CONFIG.skip_snowflake_live_check / probe override); "
                "env and SQL-file checks still ran"
            ),
        )
        return
    try:
        sf.check_connection()
    except Exception as exc:  # noqa: BLE001 - driver errors surface verbatim
        yield Issue(
            level="error",
            check="project.connections.snowflake",
            message=f"Snowflake connection failed: {exc}",
        )
```

and the signature:

```python
def snowflake_connection(*, config: Any, probe: bool | None = None) -> Iterable[Issue]:
```

(If `run_check` doesn't forward arbitrary kwargs to the check function, extend it — it lives in `automl/validate/base.py`; forwarding `**kwargs` is the expected shape.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/project -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add automl/project/checks.py tests/unit/project/test_project_validation.py
git commit -m "feat(validate): gate Snowflake live probe on RUN_CONFIG.skip_snowflake_live_check"
```

---

### Task 3: CLI override flag on `automl validate`

**Files:**
- Modify: `automl/cli/validate.py` and/or `automl/cli/_validate_actions.py` (wherever the project-validate verb parses args and calls `validate_project` — read both first; the action lives in `_validate_actions.py`)
- Test: `tests/unit/cli/test_cli_catalog.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/cli/test_cli_catalog.py`:

```python
def test_validate_project_has_probe_snowflake_flag():
    from automl.cli import build_parser

    parser = build_parser()
    validate_parser = _subparser(parser, "validate", "project")
    options = _option_strings(validate_parser)
    assert "--probe-snowflake" in options
    assert "--no-probe-snowflake" in options
```

(If the validate verb path is not `validate project`, adjust the `_subparser` path to the actual one found in `automl/cli/validate.py` — and use the same path in Step 3.)

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/cli/test_cli_catalog.py -q -k probe_snowflake`
Expected: FAIL — flag not in options.

- [ ] **Step 3: Implement**

In the validate verb's parser setup add:

```python
    parser.add_argument(
        "--probe-snowflake",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="force the Snowflake live probe on/off, overriding RUN_CONFIG.skip_snowflake_live_check",
    )
```

and thread `probe_snowflake=args.probe_snowflake` through to the `validate_project(...)` call in the action function.

- [ ] **Step 4: Run the CLI + contract suites**

Run: `uv run pytest tests/unit/cli tests/unit/project tests/contracts -q`
Expected: all PASS (update any pinned CLI catalog in the contract tests in the same commit if one fails).

- [ ] **Step 5: Commit**

```bash
git add automl/cli/ tests/unit/cli/test_cli_catalog.py
git commit -m "feat(cli): --probe-snowflake override for validate"
```

---

## Done criteria

- `uv run pytest tests/unit tests/contracts -q` green.
- `grep -rn "skip_live_check" automl/ projects/` returns nothing (the source-level flag is never created on `main`).
- With `RUN_CONFIG(skip_snowflake_live_check=True)`, `uv run automl validate project --live` (exact verb per CLI) reports the env/SQL checks plus one warning about the skipped probe — and no connection attempt.
- Follow-through note for the `neobank_ncm_v3_replicate` rebase recorded in the PR description: `projects/neobank_ncm/config.py` moves to the new field.
