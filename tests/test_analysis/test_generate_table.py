# this test hasn't been finished

import pytest
from unittest.mock import AsyncMock
from auto_reg.analysis.models import (
    RegressionAnalysis,
)

from auto_reg.regression.regression_config import (
    RegressionConfig,
)

from auto_reg.analysis.generate_table import (
    analyze_regression_result,
)


# Sample RegressionConfig (replace with your actual class and data)
@pytest.fixture
def sample_regression_config():
    return RegressionConfig(target_variable="Y", features=["X1", "X2"])


@pytest.fixture
def sample_regression_table():
    return """
    | Feature | Coefficient | P-value |
    |---|---|---|
    | X1 | 0.5 | 0.01 |
    | X2 | -0.2 | 0.05 |
    | Intercept | 1.0 | 0.001 |
    """


@pytest.fixture
def mock_chat_openai():
    mock = AsyncMock()
    mock.return_value = "The regression shows X1 is positively correlated and significant."
    return mock


@pytest.mark.asyncio  # Mark the test as asynchronous
async def test_analyze_regression_result_success(
    sample_regression_config, sample_regression_table, mock_chat_openai
):
    """Test a successful analysis scenario."""

    regression_description = "A simple linear regression."
    language_used = "English"  #Or "Chinese"

    # Call the function with the mocked ChatOpenAI model
    analysis_result = await analyze_regression_result(
        sample_regression_config,
        regression_description,
        sample_regression_table,
        mock_chat_openai,  # Pass the mock
        language_used,
    )

    # Assertions:  Check the results.  Crucial part.
    assert isinstance(analysis_result, RegressionAnalysis)