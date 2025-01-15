# Import main components
from .reg_model import panel_data
from .auto_reg_setup.regression_config import (
    RegressionConfig,
    ResearchConfig,
    BaseRegressionConfig
)
from .auto_reg_setup.varable_config import (
    RegressionModel,
    ResearchTopic,
    ControlVariables,
    NewVariable
)
from .auto_reg_analysis.generate_table import (
    draw_tables,
    analyze_regression_results,
    design_regression_tables
)
from .auto_reg_analysis.models import (
    RegressionEquation,
    RegressionAnalysis,
    RegressionResultTable,
    ResultTables
)

# Define package version
__version__ = "0.1.0"

# Define what should be available when using `from auto_reg import *`
__all__ = [
    # Regression models
    'panel_data',
    
    # Configuration classes
    'RegressionConfig',
    'ResearchConfig',
    'BaseRegressionConfig',
    'RegressionModel',
    'ResearchTopic',
    'ControlVariables',
    'NewVariable',
    
    # Analysis and table generation
    'draw_tables',
    'analyze_regression_results',
    'design_regression_tables',
    'RegressionEquation',
    'RegressionAnalysis',
    'RegressionResultTable',
    'ResultTables'
]
