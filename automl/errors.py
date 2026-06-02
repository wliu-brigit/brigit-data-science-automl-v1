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
