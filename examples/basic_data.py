import pandas as pd
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from auto_reg.regression.regression_config import ResearchConfig

def setup_basic_data() -> tuple[pd.DataFrame, ResearchConfig]:
    # Read and prepare data
    df = pd.read_csv('temp/simulated_data.csv')
    df = df.set_index(['company_id', 'year'])

    # header = [
    #     "company_id", "year", "extreme_temperature", "stock_revenue", "company_size", 
    #     "company_age", "company_distance_to_sea", "rain_amount", "dry_amount", "windy", 
    #     "company_latitude", "is_high_tech", "is_near_sea", "local_climate_variability", 
    #     "stock_revenue_another_measure_method", "industry", "invester_mood"
    # ]

    research_config = ResearchConfig(
        research_topic="The relationship between investment in extreme temperature and company stock abnormal return.",
        dependent_vars=["stock_revenue"],
        dependent_var_description=["Company's abnormal stock returns, calculated as the difference between actual and expected returns"],
        
        independent_vars=["extreme_temperature"],
        independent_var_description=["Frequency of extreme temperature events per year, standardized across all companies"],

        control_vars=["company_size", "company_age", "company_distance_to_sea", "rain_amount", "dry_amount"],
        control_vars_description=[
            "Size of the company, measured by natural logarithm of total assets",
            "Age of the company since establishment in years",
            "Distance of company headquarters to nearest coastline in kilometers",
            "Annual rainfall amount in millimeters of company's location",
            "Number of dry days per year of company's location",
        ],

        instrument_vars=["company_latitude"],
        instrument_vars_description=["Geographical latitude of company headquarters in decimal degrees"],

        group_vars=["is_high_tech", "is_near_sea"],
        group_vars_description=[
            "Binary indicator for high-tech industry classification (1 = high-tech, 0 = otherwise)",
            "Binary indicator for coastal proximity (1 = near sea, 0 = inland)"
        ],

        mediating_vars=["invester_mood"],
        mediating_vars_description=["Investor sentiment index measuring market optimism/pessimism for specific company"],

        extra_control_vars=["windy"],
        extra_control_vars_description=["Average wind speed in meters per second of company's location, used for robustness checks"],

        extra_effects=["time", "industry"],
        extra_effects_vars=["year", "industry"],

        replacement_x_vars=["local_climate_variability"],
        replacement_x_vars_description=["Local climate variability index of company's location, alternative measure for extreme weather exposure"],
        
        replacement_y_vars=["stock_revenue_another_measure_method"],
        replacement_y_vars_description=["Alternative measure of stock returns using market-adjusted method"],

        effects=["entity", "time"],
        effects_vars=["company_id", "year"],

        constant=True,
        run_another_regression_without_controls=True
    )

    research_config.validate_research_config(df)

    return df, research_config

def get_research_topic(research_config: ResearchConfig) -> str:
    return research_config.research_topic

if __name__ == "__main__":
    df, research_config = setup_basic_data()

    print(get_research_topic(research_config))