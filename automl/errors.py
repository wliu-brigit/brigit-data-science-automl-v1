"""Exception hierarchy for brigit-automl.

Lives at the package top (not under ``utils/``) because the exception hierarchy
is part of the public surface. ``StorageError`` wraps backend errors at the
MLflow/GCS persistence seam; the per-domain leaves describe where a failure
originated.
"""


class AutoMLError(Exception):
    """Base class for all brigit-automl errors."""


class ProjectError(AutoMLError):
    """Project context / config problems."""


class ValidationError(AutoMLError):
    """Validation framework failures."""


class ContractError(ValidationError):
    """Contract-level validation failures."""


class ConfigError(ProjectError):
    """Project configuration failures."""


class ProposalError(ValidationError):
    """Proposal validation or handoff failures."""


class TrialError(AutoMLError):
    """Trial authoring or inspection failures."""


class TrialFitError(TrialError):
    """Trial model fitting failures."""


class TrialEvalError(TrialError):
    """Trial evaluation failures."""


class DataError(AutoMLError):
    """Data domain problems."""


class ModelError(AutoMLError):
    """Model domain problems."""


class EvalError(AutoMLError):
    """Eval domain problems."""


class RunnerError(AutoMLError):
    """Runner / trial-execution problems."""


class StorageError(AutoMLError):
    """MLflow/GCS persistence-seam failures; wraps the backend error via __cause__."""


def format_error_chain(exc: BaseException) -> str:
    """``Wrapper: msg (caused by Root: msg)`` — the full ``__cause__`` chain.

    Wrapper exceptions (``raise X from err``) otherwise hide the root cause
    from anything that only sees the message: the AUTOML_ERROR stdout marker,
    agent failure summaries, log lines. A StorageError that was really a
    JSON-encoding TypeError must say so where the agent reads it.
    """
    parts: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        parts.append(f"{type(current).__name__}: {current}")
        if current.__cause__ is not None:
            current = current.__cause__
        elif not current.__suppress_context__:
            current = current.__context__
        else:
            current = None
    if len(parts) == 1:
        return parts[0]
    return f"{parts[0]} (caused by {' <- '.join(parts[1:])})"
