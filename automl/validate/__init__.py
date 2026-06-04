"""Validate vocabulary public API.

A true leaf: value objects plus the crash-safe check runner. Validation
recipes live with their domains — ``automl.model.validate_model``,
``automl.project.validate_project``, ``automl.agent.validate_proposal``.
"""

from automl.validate.base import Issue, Severity, ValidationReport, run_check

__all__ = [
    "Issue",
    "Severity",
    "ValidationReport",
    "run_check",
]
