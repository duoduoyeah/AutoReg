# simulate VAR model
# This file is first generate by LLM, then manually modified

import pandas as pd
import numpy as np
from typing import Optional
import os


def generate_data_basic_structure(
    n_entities: int = 100,
    n_periods: int = 10,
) -> pd.DataFrame:
    """
    Generate a panel dataset with double index (entity and time)
    
    Args:
        n_entities: Number of entities (default 1000)
        n_periods: Number of time periods in years (default 10)
        
    Returns:
        pd.DataFrame: Empty DataFrame with MultiIndex (entity, time)
    """
    # Generate entity IDs
    entities = range(n_entities)
    
    # Generate time periods (years)
    # Generate time periods starting from 0
    times = range(n_periods)

    # Create MultiIndex
    index = pd.MultiIndex.from_product(
        [entities, times],
        names=['entity', 'time']
    )
    
    # Create empty DataFrame with the MultiIndex
    df = pd.DataFrame(index=index)
    
    return df

def generate_variables(
    var_name: str,
    df: pd.DataFrame,
    related_var: Optional[str] = None,
    correlation: Optional[float] = 0.3,
    dummy_var: Optional[bool] = False,
    group_var: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate a new variable in the DataFrame, optionally correlated with an existing variable
    
    Args:
        var_name: Name of the new variable to generate
        df: Panel DataFrame with MultiIndex (entity, time)
        related_var: Optional name of existing variable to correlate with
        correlation: Optional correlation coefficient (-1 to 1)
        
    Returns:
        pd.DataFrame: DataFrame with new variable added
    """
    
    
    n_samples = len(df)
    if dummy_var:
        # Generate dummy variable based on entity
        n_entities = len(df.index.get_level_values('entity').unique())
        # Generate random binary values for each entity
        entity_dummies = np.random.binomial(1, 0.5, n_entities)
        # Map entity dummies to each observation by entity index
        df[var_name] = entity_dummies[df.index.get_level_values('entity')]
        return df
    
    if group_var is not None:
        # Generate group variable based on entity
        n_groups = len(df.index.get_level_values('entity').unique())
        # Generate random binary values for each entity
        group_dummies = np.random.randint(0, group_var, n_groups)
        # Map entity dummies to each observation by entity index
        df[var_name] = group_dummies[df.index.get_level_values('entity')]
        return df

    if related_var is not None and correlation is not None:
        # Generate correlated random variable
        if var_name not in df.columns:
            existing_data = df[related_var].values
            
            # Generate random normal data
            new_data = np.random.normal(0, 1, n_samples)
            
            # Calculate the correlation coefficient
            rho = correlation
            
            # Apply correlation transformation
            new_data = rho * existing_data + np.sqrt(1 - rho**2) * new_data
            
            # Standardize the new variable
            new_data = (new_data - np.mean(new_data)) / np.std(new_data)
            
            df[var_name] = new_data
    else:
        # Generate independent random variable with time and entity effects
        entity_effect = np.random.normal(0, 0.4, len(df.index.get_level_values('entity').unique()))
        time_effect = np.random.normal(0, 0.4, len(df.index.get_level_values('time').unique()))
        
        # Map effects to each observation
        entity_component = pd.Series(entity_effect[df.index.get_level_values('entity')], index=df.index)
        time_component = pd.Series(time_effect[df.index.get_level_values('time')], index=df.index)
        
        # Combine random effect with entity and time components (0.8 weight on random component)
        df[var_name] = np.random.normal(0, 0.2, n_samples) + entity_component + time_component
    
    return df


def generate_new_csv(n_entities: int = 100, n_periods: int = 10):
    os.environ["RESEARCH_TOPIC"] = "The Impact of Extreme Temperatures on Stock Returns"
    df = generate_data_basic_structure(n_entities=100)
    # independent variable
    df = generate_variables(var_name='extreme_temperature', df=df)
    # dependent variable
    df = generate_variables(var_name='stock_revenue', related_var='extreme_temperature', df=df)
    # control variables
    control_vars = ['company_size', 'company_age', 'company_location', 'company_industry', 'company_revenue', 'company_profit']
    for control_var in control_vars:
        df = generate_variables(var_name=control_var, related_var='stock_revenue', correlation=0.1, df=df)
    
    # tool variable
    df = generate_variables(var_name='latitude', related_var='extreme_temperature', correlation=0.6, df=df)

    # heterogenous variable
    df = generate_variables(var_name='is_high_tech', related_var='stock_revenue', correlation=0.4, df=df, dummy_var=True)
    df = generate_variables(var_name='is_near_sea', related_var='stock_revenue', correlation=0.1, df=df, dummy_var=True)

    # stability test
    df = generate_variables(var_name='extreme_temperature_another_measure_method', related_var='extreme_temperature', correlation=0.9, df=df)

    df = generate_variables(var_name='stock_revenue_another_measure_method', related_var='stock_revenue', correlation=0.9, df=df)

    print(df.loc[df.index.get_level_values('entity') < 10])
    # Write dataframe to CSV file
    df.to_csv('temp/simulated_data.csv')

if __name__ == '__main__':
    import pandas as pd

    # Read the generated CSV file
    df = pd.read_csv('temp/simulated_data.csv')
    df = df.set_index(['company_id', 'year'])

    # Display the first few rows of the dataframe
    print(df.head())

    df = generate_variables(var_name='invester_mood',
                        related_var='extreme_temperature',
                        correlation=0.9,
                        df=df)
    print(df[['invester_mood']].head())
    df.to_csv('temp/simulated_data.csv')
