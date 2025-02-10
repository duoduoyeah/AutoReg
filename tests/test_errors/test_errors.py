import pytest
from auto_reg.errors.exceptions import (
    MissingEnvironmentVariables,
)
from auto_reg.static import Messages

def test_missing_environment_variables():
    # Test basic exception
    with pytest.raises(MissingEnvironmentVariables) as exc_info:
        raise MissingEnvironmentVariables()
    assert str(exc_info.value) == Messages.MISSING_ENVIRONMENT_VARIABLES

    # Test with extra info
    extra_info = {"missing_key": "API_KEY"}
    with pytest.raises(MissingEnvironmentVariables) as exc_info:
        raise MissingEnvironmentVariables(extra_info=extra_info)
    assert str(exc_info.value) == f"{Messages.MISSING_ENVIRONMENT_VARIABLES} (Extra Info: {extra_info})"
