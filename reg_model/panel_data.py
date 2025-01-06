from linearmodels.panel import PanelOLS
import pandas as pd
from linearmodels.panel.results import PanelEffectsResults
from ..auto_reg_setup.regression_config import RegressionConfig
from typing import Optional

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
        regression_results.append(result)
    
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


def run_regressions(df: pd.DataFrame, 
                    regression_configs: dict[str, RegressionConfig]) -> list[tuple[str, list[PanelEffectsResults]]]:
    """
    Run regressions based on the regression config
    
    Requirement: Double Indexed DataFrame
        With the first index being the entity and the second index being the time.
    """
    # check if the df is double indexed
    if not isinstance(df.index, pd.MultiIndex):
        raise ValueError("DataFrame must be double indexed")
    
    # This is a list of tuples, first item in the tuple is regression description, second item is the regression result.
    regression_results: list[tuple[str, list[PanelEffectsResults]]] = []

    for reg_type, reg_config in regression_configs.items():        
        if reg_config.instrument_var is not None:
            regression_results.append((reg_type, two_stage_regression(df, reg_config)))
        elif reg_config.group_var is not None:
            modify_type = f"{reg_type}\n The first regression result list below is the one with dummy variable == 0\n The second regression result is the one with dummy variable == 1"
            regression_results.append((modify_type, group_regression(df, reg_config)))
        else:
            if reg_config.run_another_regression_without_controls:
                reg_type = f"{reg_type}\n The first regression result list below is the one with controls\n The second regression result is the one without controls"

            regression_results.append((reg_type, panel_regression(df, reg_config)))

    #  In the tuple, add the regression nums in the first item

    for i, (reg_description, results) in enumerate(regression_results):
        current_regression_num = len(results)
        reg_description = f"Index: {i}\n Under Index {i}, the number of regressions is: {current_regression_num}\n{reg_description}\n "
        regression_results[i] = (reg_description, results)



    return regression_results
