class Prompts:
    """Class for storing prompts for auto_regression."""

    DEFAULT_MODEL_TYPE: str = "panel data regression model"

    regression_models: dict[str, str] = {
        "basic_regression": "panel data regression model",
        "stability_test": "panel data regression model for stability test",
        "mediating_effect": "panel data regression model for examining mediating effect",
        "moderating_effect": "panel data regression model for examining moderating effect", 
        "heterogeneity": "panel data regression model for examining heterogeneity",
        "endogeneity": "panel data regression model for examining endogeneity"
    }
    
    DEFAULT_BASIC_TASK_DESCRIPTION: str = "I aim to investigate the relationship between a company's investment in cloud computing and its revenue over time using panel data regression analysis."

    DEFAULT_PREIVOUS_TASK_DESCRIPTION: str = "Previously we have completed a basic regression model. "

    DEFAULT_PREIVOUS_RESULT_CONFIGURATION: str = "The configuration of the basic regression model is {'type': 'basic_regression', 'dependent_vars': ['revenue'], 'independent_vars': ['cloud_investment'], 'effects': ['entity_id', 'time'], 'control_vars': ['total_assets', 'market_share', 'rd_spend', 'op_costs', 'gdp_growth', 'employee_count', 'regional_econ'], 'constant': True}\n"
