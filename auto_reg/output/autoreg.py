import pandas as pd
import json
import warnings

from ..regression.regression_config import ResearchConfig
from ..regression.panel_data import *
from ..analysis.generate_table import *
from ..analysis.design import *
from ..errors import *
from ..output.document_generator import (
    create_tex, 
    generate_word, 
    generate_pdf,
    generate_tex,
)

async def autoreg(
    data: pd.DataFrame,
    research_configuration: ResearchConfig,
    models: dict[str, ChatOpenAI],
    output_path: str = "./temp/autoreg",
    analaysis_language: str = "Chinese",
    verbose: bool = False,
    output_types: list[str] = ["latex", "word", "pdf"],
):

    """Run automated regression analysis pipeline.

    Args:
        data_path: Path to dataset file (CSV format)
        json_path: Path to research configuration JSON file
        models: Dictionary of AI models for different tasks. Requires:
            - table_model: For table rendering
            - analysis_model: For table interpretation
        data_index: Column names/positions to use as DataFrame index
        output_path: Directory path for output files. Default "./temp/autoreg"
        analaysis_language: Language for output analysis. Default "Chinese"
        verbose: Whether to print table design info. Default False
        output_types: List of output formats to generate. Default ["latex", "word", "pdf"]

    Returns:
        None

    Changes:
        Creates output files in specified formats at output_path location

    Raises:
        ConfigError: If research configuration is invalid
        DataError: If data format is incorrect
        ModelError: If AI model responses are invalid
    """
    
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

    for output_type in output_types:
        try:
            if output_type == "latex":
                doc = create_tex(combined_table_results)
                generate_tex(doc, output_path)
            elif output_type == "word":
                generate_word(doc, combined_table_results)
            elif output_type == "pdf":
                generate_pdf(doc, output_path)
            else:
                print(f"Output type '{output_type}' is not supported. Supported types are: latex, word, pdf")
        except OutputFileError as e:
            print(e)
            continue

def setup_data(data_path, data_index, json_path):
    """
    Load data and research configuration from file paths.

    Args:
        data_path: Path to dataset file (CSV or Excel format)
        data_index: Column names/positions to use as DataFrame index
        json_path: Path to research configuration JSON file

    Returns:
        Tuple of (DataFrame, ResearchConfig)

    Raises:
        DataFileError: If data file format is incorrect
        JsonFileError: If research configuration file format is incorrect
    """
    try:
        if data_path.endswith(".csv"):
            df = pd.read_csv(data_path)
        elif data_path.endswith(".xlsx"):
            df = pd.read_excel(data_path)
        else:
            raise ValueError(f"Unsupported file type: {data_path}")
        df = df.set_index(data_index)
        rows = df.shape[0]
        df.dropna(inplace=True)
        droped_rows = rows - df.shape[0]
        if droped_rows/rows > 0.2:
           warnings.warn("missing value rows is larger than 20%%")
    except:
        raise DataFileError

    research_configuration = load_research_config(json_path)
    research_configuration.validate_research_config(df)

    return df, research_configuration

def load_research_config(config_path: str) -> ResearchConfig:
    try:
        with open(config_path) as f:
            config_data = json.load(f)
        return ResearchConfig(**config_data)
    
    except:
        raise JsonFileError