import pytest

from automl import errors

pytestmark = pytest.mark.unit


def test_error_hierarchy_is_public_and_domain_specific():
    assert issubclass(errors.ProjectError, errors.AutoMLError)
    assert issubclass(errors.DataError, errors.AutoMLError)
    assert issubclass(errors.ModelError, errors.AutoMLError)
    assert issubclass(errors.EvalError, errors.AutoMLError)
    assert issubclass(errors.RunnerError, errors.AutoMLError)
    assert issubclass(errors.StorageError, errors.AutoMLError)


def test_storage_error_supports_backend_cause_chaining():
    backend_error = RuntimeError("backend failed")

    try:
        raise errors.StorageError("storage failed") from backend_error
    except errors.StorageError as exc:
        assert exc.__cause__ is backend_error


def test_format_error_chain_surfaces_the_root_cause():
    # A wrapper must not hide the root cause from message-only consumers
    # (AUTOML_ERROR marker, agent failure summaries).
    try:
        try:
            raise TypeError("Object of type Decimal is not JSON serializable")
        except TypeError as inner:
            raise errors.StorageError("Failed to log MLflow pyfunc model") from inner
    except errors.StorageError as exc:
        chain = errors.format_error_chain(exc)
    assert chain == (
        "StorageError: Failed to log MLflow pyfunc model "
        "(caused by TypeError: Object of type Decimal is not JSON serializable)"
    )


def test_format_error_chain_without_cause_is_just_the_message():
    assert errors.format_error_chain(ValueError("plain")) == "ValueError: plain"


def test_format_error_chain_follows_implicit_context_unless_suppressed():
    try:
        try:
            raise KeyError("inner")
        except KeyError:
            raise RuntimeError("outer")  # implicit __context__, no `from`
    except RuntimeError as exc:
        assert "KeyError" in errors.format_error_chain(exc)

    try:
        try:
            raise KeyError("inner")
        except KeyError:
            raise RuntimeError("outer") from None  # suppressed
    except RuntimeError as exc:
        assert errors.format_error_chain(exc) == "RuntimeError: outer"
