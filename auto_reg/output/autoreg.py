import pandas as pd
import json
import warnings

from ..regression.regression_config import ResearchConfig
from ..regression.panel_data import *
from ..analysis.generate_table import *
from ..analysis.design import *
from ..errors import *

async def autoreg(
    data_path: str,
    json_path: str,
    data_index: list[str],
    models: dict[str, ChatOpenAI],
    output_path: str = "./temp/",
    analaysis_language: str = "Chinese",
    verbose: bool = False,
    output_type: str = "latex",
):

    """Run automated regression analysis pipeline.

    Args:
        data_path: Path to dataset file (CSV format)
        json_path: Path to research configuration JSON file
        models: Dictionary of AI models for different tasks. Requires:
            - table_model: For table rendering
            - analysis_model: For table interpretation
        data_index: Column names/positions to use as DataFrame index. 
            Default [0,1] (first two columns)
        analaysis_language: Language for output analysis. 
            Default Chinese
    """

    data = setup_data(data_path, data_index)
    research_configuration = load_research_config(json_path)
    research_configuration.validate_research_config(data)
    
    # run regressions
    regression_results = run_regressions(
        data, 
        research_configuration.generate_regression_configs(),
    )

    # Design regression tables
    table_design: TableDesign | None = await design_regression_tables(
        research_configuration.research_topic, 
        regression_results, models["table_model"]
    )

    if table_design is None:
        return
    
    # user select tables
    table_design = select_table_design(table_design)
    if verbose:
        print(table_design)

    # draw tables
    table_results = ResultTables()
    await draw_tables(
        regression_results, table_design, models["table_model"], table_results
    )

    # User need to: adjust the language used(Any legit str is ok)
    await analyze_regression_results(
        regression_results,
        table_design,
        table_results,
        models["analysis_model"],
        language_used=analaysis_language,
    )

    # combine tables
    combined_table_results: ResultTables = await combine_tables(
        table_results, table_design, models["table_model"]
    )

    if output_type == "latex":
        pass
    else:
        raise ValueError(f"Output type {output_type} hasn't been defined.")
        
def setup_data(data_path, data_index):
    try:
        df = pd.read_csv(data_path)
        df = df.set_index(data_index)
        rows = df.shape[0]
        df.dropna(inplace=True)
        droped_rows = rows - df.shape[0]
        if droped_rows/rows > 0.2:
           warnings.warn("missing value rows is larger than 20%%")

        return df            
    except:
        raise DataFileError


def load_research_config(config_path: str) -> ResearchConfig:
    try:
        with open(config_path) as f:
            config_data = json.load(f)
        return ResearchConfig(**config_data)
    
    except:
        raise JsonFileError