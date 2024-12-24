
import pandas as pd
from linearmodels import PanelOLS
import os
from pathlib import Path

CURRENT_FILE = Path(__file__).resolve()
PANEL_DATA_FILE = os.path.join(
    os.path.dirname(os.path.dirname(CURRENT_FILE)),
    "test_data",
    "panel_data_5000_20_with_city.csv"
)

def out_put_file():
    pass

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

def test_panel_data(df):
    
    # Below are the future input
    dependent_var = ["y"]
    independent_vars = ["x"]
    control_vars = ["control_1", "control_2"]
    effects = ['time', 'city'] # at most two effects. Sugeestion: [time + other]
    constant = True
    
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
    print(result)
    
def main(df: pd.DataFrame):
    test_panel_data(df)

if __name__ == "__main__":
    df = pd.read_csv(PANEL_DATA_FILE)
    df = df.set_index(["entity_id", "time"])
    main(df)
