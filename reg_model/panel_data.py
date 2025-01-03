from linearmodels.panel import PanelOLS
import pandas as pd
from linearmodels.panel.results import PanelEffectsResults

def basic_panel_data(df: pd.DataFrame,
                    dependent_var: list[str],
                    independent_vars: list[str], 
                    effects: list[str] = [],
                    control_vars: list[str] = [],
                    constant: bool = True) -> PanelEffectsResults:
    """
    Basic panel data model
    """
    # below will be built as a function
    dep_var = df[dependent_var]
    exog_vars = df[independent_vars + control_vars]
    if constant:
        exog_vars = exog_vars.assign(constant=1)
    entity_effects, time_effects, other_effects = fixed_effects(effects, df)

    
    model = PanelOLS(
        dependent=dep_var,
        exog=exog_vars,
        entity_effects=entity_effects,
        time_effects=time_effects,
        other_effects=other_effects,
    )

    result = model.fit(cov_type="clustered", cluster_entity=True)
    return result

def fixed_effects(effects: list[str], df: pd.DataFrame) -> tuple[bool, bool, bool]:
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
