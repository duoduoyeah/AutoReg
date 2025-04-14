import pandas as pd
import json
import warnings
from ..regression.regression_config import ResearchConfig
from ..regression.panel_data import *
from ..analysis.generate_table import *
from ..analysis.design import *
from ..errors import *
from .document_generator import (
    create_tex, 
    generate_word, 
    generate_pdf,
    generate_tex
)

async def autoreg(
    data: pd.DataFrame,
    research_configuration: ResearchConfig,
    models: dict[str, ChatOpenAI],
    output_path: str = "./temp/autoreg",
    analaysis_language: str = "Chinese",
    verbose: bool = False,
    output_types: list[str] = ["latex", "word", "pdf"],
    all_table: bool = False,
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
        raise TableDesignError()
    elif verbose:
        print("table_design: ", table_design)
        
    # user select tables
    if all_table:
        max_table_num = table_design.number_of_tables
        table_design = select_table_design(table_design, number_of_tables=max_table_num)
    else:
        table_design = select_table_design(table_design)

    if verbose:
        print("User selected table design: ", table_design)

    # draw tables
    table_results = ResultTables()
    await draw_tables(
        regression_results, table_design, models["table_model"], table_results
    )
    
    if verbose:
        for i in range(table_results.get_length()):
            print(f"table_result {i}: ", str(table_results.get_tables([i])[0].latex_table)[:100])
            print(f"analysis {i}: ", table_results.get_analysis([i])[0].latex_analysis)

    # analyze regression results
    await analyze_regression_results(
        regression_results,
        table_design,
        table_results,
        models["analysis_model"],
        language_used=analaysis_language,
    )

    if verbose:
        for i in range(table_results.get_length()):
            print(f"analysis {i}: ", table_results.get_analysis([i])[0].latex_analysis)

    # combine tables
    combined_table_results: ResultTables = await combine_tables(
        table_results, table_design, models["table_model"]
    )

    if verbose:
        for i in range(combined_table_results.get_length()):
            print("combined_table_results: ", str(combined_table_results.get_tables([i])[0].latex_table)[:100])
        

    try:
        if "latex" in output_types:
            doc = create_tex(combined_table_results)
            generate_tex(doc, output_path + f"/regression_analysis.tex")
        if "word" in output_types:
            latex_file = output_path + f"/regression_analysis.tex"
            generate_word(latex_file, output_path + f"/regression_analysis.docx")
        if "pdf" in output_types:
            generate_pdf(doc, output_path + f"/regression_analysis.pdf")
    except OutputFileError as e:
        print(e)


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

    research_configuration = _load_research_config(json_path)
    research_configuration.validate_research_config(df)

    return df, research_configuration

def _load_research_config(config_path: str) -> ResearchConfig:
    try:
        with open(config_path) as f:
            config_data = json.load(f)
        return ResearchConfig(**config_data)
    
    except:
        raise JsonFileError

def check_models(models: dict[str, ChatOpenAI]):
    if "table_model" not in models or "analysis_model" not in models:
        raise ModelError("models must contain 'table_model' and 'analysis_model'")
    if not isinstance(models["table_model"], ChatOpenAI) or not isinstance(models["analysis_model"], ChatOpenAI):
        raise ModelError("models must be ChatOpenAI")
    
    try:
        strings = []
        strings.append(models["table_model"].invoke("test you, please return a string"))
        strings.append(models["analysis_model"].invoke("test you, please return a string"))
    except Exception as e:
        raise ModelError({"message":"models can't connect to the model server", "error": e})
    
    return strings
