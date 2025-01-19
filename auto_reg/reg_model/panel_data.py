from linearmodels.panel import PanelOLS
import pandas as pd
from linearmodels.panel.results import PanelEffectsResults
from ..auto_reg_setup.regression_config import RegressionConfig
from pydantic import BaseModel, ConfigDict

class RegressionResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    description: str  # A textual description of the regression result
    results: list[PanelEffectsResults]  # A list of regression results from the panel data model
    regression_type: str  # The type of regression performed
    regression_config: RegressionConfig  # The configuration settings used for the regression


def fixed_effects(effects: list[str], df: pd.DataFrame) -> tuple[bool, bool, bool]:
    """
    Return the fixed effects for the regression

    Use "entity" to indicate the entity effect, not the variable name
    Use "time" to indicate the time effect, not the variable name
    For other effects, directly use the column name

    """
    if "entity" in effects:
        entity_effects = True
    else:
        entity_effects = False
    
    if "time" in effects:
        time_effects = True
    else:
        time_effects = False
        
    other_effects = None
    other_cols = [col for col in effects if col not in ["entity", "time"]]
    if len(other_cols) == 1:
        other_effects = df[other_cols[0]]
    elif len(other_cols) > 1:
        other_effects = df[other_cols]
    
    return entity_effects, time_effects, other_effects

def panel_regression(df: pd.DataFrame,
                    regression_config: RegressionConfig,
                    ) -> list[PanelEffectsResults]:
    """
    Basic panel data model.

    When run another regression without controls, the first regression is the one with controls.
    """
    regression_results = []

    
    dep_var = df[regression_config.dependent_vars]
    
    
    exog_vars = df[regression_config.independent_vars + regression_config.control_vars]

    
    if regression_config.constant:
        exog_vars = exog_vars.assign(constant=1)

    entity_effects, time_effects, other_effects = fixed_effects(regression_config.effects, df)


    model = PanelOLS(
        dependent=dep_var,
        exog=exog_vars,
        entity_effects=entity_effects,
        time_effects=time_effects,
        other_effects=other_effects,
    )

    # run regression
    result = model.fit(cov_type="clustered", cluster_entity=True)
    regression_results.append(result)

    # run another regression without controls
    if regression_config.run_another_regression_without_controls:

        exog_vars = df[regression_config.independent_vars]
        if regression_config.constant:
            exog_vars = exog_vars.assign(constant=1)
            
        model = PanelOLS(
            dependent=dep_var,
            exog=exog_vars,
            entity_effects=entity_effects,
            time_effects=time_effects,
            other_effects=other_effects,
        )
        result = model.fit(cov_type="clustered", cluster_entity=True)
        regression_results = [result] + regression_results
    
    

    return regression_results   


def two_stage_regression(df: pd.DataFrame, regression_config: RegressionConfig) -> list[PanelEffectsResults]:
    """
    Two stage regression using instrumental variables (2SLS)
    
    First stage: Regress endogenous variable on instruments and controls
    Second stage: Use predicted values from first stage
    """
    # Get the endogenous variable (first independent variable)
    endogenous_var = regression_config.independent_vars[0]
    
    # First stage: regress endogenous variable on instrument and controls
    dep_var = df[endogenous_var]  # endogenous variable is now dependent variable
    exog_vars = df[[regression_config.instrument_var] + regression_config.control_vars]

    if regression_config.constant:
        exog_vars = exog_vars.assign(constant=1)

    entity_effects, time_effects, other_effects = fixed_effects(regression_config.effects, df)
    
    model = PanelOLS(
        dependent=dep_var,
        exog=exog_vars,
        entity_effects=entity_effects,
        time_effects=time_effects,
        other_effects=other_effects,
    )   
    first_stage = model.fit(cov_type="clustered", cluster_entity=True)
    
    # Second stage: use predicted values
    df_with_predicted = df.copy()
    df_with_predicted[f'{endogenous_var}_predicted'] = first_stage.fitted_values
    
    # Run second stage with predicted values
    dep_var = df_with_predicted[regression_config.dependent_vars]
    exog_vars = df_with_predicted[[f'{endogenous_var}_predicted'] + regression_config.control_vars]
    
    model = PanelOLS(
        dependent=dep_var,
        exog=exog_vars,
        entity_effects=entity_effects,
        time_effects=time_effects,
        other_effects=other_effects,
    )
    second_stage = model.fit(cov_type="clustered", cluster_entity=True)

    return [first_stage, second_stage]

def group_regression(df: pd.DataFrame, regression_config: RegressionConfig) -> list[PanelEffectsResults]:
    """
    Group regression.

    The first regression is the one with dummy variable == 0
    The second regression is the one with dummy variable == 1
    """
    group_var = regression_config.group_var
    
    # Split sample based on group variable
    df_group_0 = df[df[group_var] == 0]
    df_group_1 = df[df[group_var] == 1]
    
    # Run regression for each group
    results = []
    for group_df in [df_group_0, df_group_1]:
        dep_var = group_df[regression_config.dependent_vars]

        exog_vars = group_df[regression_config.independent_vars + regression_config.control_vars]
        if regression_config.constant:
            exog_vars = exog_vars.assign(constant=1)

        entity_effects, time_effects, other_effects = fixed_effects(regression_config.effects, group_df)
        
        model = PanelOLS(
            dependent=dep_var,
            exog=exog_vars,
            entity_effects=entity_effects,
            time_effects=time_effects,
            other_effects=other_effects
        )
        results.append(model.fit(cov_type="clustered", cluster_entity=True))

    return results


def get_function_name(func) -> str:
    """
    Returns the name of the function as a string.
    
    :param func: The function whose name is to be returned.
    :return: The name of the function.
    """
    return func.__name__

def run_regressions(df: pd.DataFrame, 
                    regression_configs: dict[str, RegressionConfig]) -> list[RegressionResult]:
    """
    Run regressions based on the regression config
    
    Requirement: Double Indexed DataFrame
        With the first index being the entity and the second index being the time.

    Return: 
    A list of RegressionResult, each contains:
    1. the regression description
    2. the regression result
    3. the regression type
    4. the regression config
    """

    # check if the df is double indexed
    if not isinstance(df.index, pd.MultiIndex):
        raise ValueError("DataFrame must be double indexed")
    
    regression_results: list[RegressionResult] = []

    for reg_type, reg_config in regression_configs.items():        
        if reg_config.instrument_var is not None:
            modify_description = f"{reg_type}\n The first regression result is the one with instrumental variable, i.e. stage 1 of 2SLS\n The second regression result is the one use predicted values from the first stage, i.e. stage 2 of 2SLS\n"
            regression_results.append(
                RegressionResult(
                    description=modify_description, 
                    results=two_stage_regression(df, reg_config),
                    regression_type=get_function_name(two_stage_regression), 
                    regression_config=reg_config))

        elif reg_config.group_var is not None:
            modify_description = f"{reg_type}\n The first regression result is the one with dummy variable == 0\n The second regression result is the one with dummy variable == 1"
            regression_results.append(
                RegressionResult(
                    description=modify_description, 
                    results=group_regression(df, reg_config),
                    regression_type=get_function_name(group_regression), 
                    regression_config=reg_config))
        else:
            if reg_config.run_another_regression_without_controls:
                reg_type = f"{reg_type}\n The first regression result is the one without controls\n The second regression result is the one with controls"

            regression_results.append(
                RegressionResult(
                    description=reg_type, 
                    results=panel_regression(df, reg_config),
                    regression_type=get_function_name(panel_regression), 
                    regression_config=reg_config))

    return regression_results

def add_reg_descriptions(regression_results: list[RegressionResult]) -> None:
    """
    Add the regression description to the regression result
    """
    for i, reg_result in enumerate(regression_results):
        reg_result.description = f"Index: {i}\n Under Index {i}, the number of regressions is: {len(reg_result.results)}\n{reg_result.description}\n "

def remove_reg_descriptions(regression_results: list[RegressionResult]) -> None:
    """
    Remove the index and regression count information from the regression description,
    keeping only the original description part.
    """
    for reg_result in regression_results:
        lines = reg_result.description.split("\n")
        # Find the first line that doesn't start with "Index:" or "Under Index"
        start_idx = 0
        for i, line in enumerate(lines):
            if not line.strip().startswith(("Index:", "Under Index")):
                start_idx = i
                break
        reg_result.description = "\n".join(lines[start_idx:]).strip() + "\n"