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
