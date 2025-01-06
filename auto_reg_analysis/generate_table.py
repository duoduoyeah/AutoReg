from pydantic import BaseModel, Field
from linearmodels.panel.results import PanelEffectsResults
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from ..static.langchain_query import LangchainQueries
from langchain_openai import ChatOpenAI
import os
from ..auto_reg_setup.regression_config import RegressionConfig, ResearchConfig

class RegressionResultTable(BaseModel):
    table: str = Field(description="regression result table", default="")


class TableDesign(BaseModel):
    number_of_tables: int = Field(description="the number of regression tables I need create")
    table_index: list[list[int]] = Field(description="the index used by each regression table using a list of list of integers. The list is as long as the number of regression tables. For each sublist, it contains the index of the regression results that should be combined into one table.")
    table_title: list[str] = Field(description="the title of each regression table using a list of strings. The list is as long as the number of regression tables.")


def generate_econometric_analysis_table(
    research_config: ResearchConfig,
    regression_config: RegressionConfig,
    regression_results: list[PanelEffectsResults],
    model: ChatOpenAI,
    table_template: str|None = None,
    query: str|None = None
) -> RegressionResultTable:
    """
    Generate econometric analysis table.
    
    Information to be given to the language model:
    - research topic
    - regression config
    - regression result
    - latex table template
    - number of regression results
    
    Infromation returned by the language model:
    - regression result table as a string in latex format

    Returns:
        RegressionResultTable: The regression result table.

    Raises:
        ValueError: If RESEARCH_TOPIC is not set.
    """
    # setup prompt
    parser = JsonOutputParser(pydantic_object=RegressionResultTable)

    if table_template is None:
        table_template = LangchainQueries.TABLE_EXAMPLE


    if query is None:
        query = LangchainQueries.format_query(
            LangchainQueries.BASIC_REGRESSION_TABLE_QUERY,
            research_topic=research_config.research_topic,
            regression_config=str(regression_config),
            regression_result = str(regression_results),
            latex_table_template=table_template,
            number_of_results=len(regression_results)
        )
    else:
        query = LangchainQueries.format_query(
            query,
            research_topic=research_config.research_topic,
            regression_config=str(regression_config),
            regression_result = str(regression_results),
            latex_table_template=table_template,
            number_of_results=len(regression_results)
        )

    prompt = PromptTemplate(
        template="Answer the user query.\n{format_instructions}\n{query}\n",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | model | parser

    output = chain.invoke({"query": query})

    return output


def draw_tables(regression_results: list[tuple[str, list[PanelEffectsResults]]]) -> dict[str, list[RegressionResultTable]]:
    """
    Draw tables for each regression result
    """
    pass


def design_regression_tables(
        research_topic: str,
        regression_results: list[tuple[str, list[PanelEffectsResults]]],
        model: ChatOpenAI,
        max_try_times: int = 3
) -> TableDesign:
    """
    Design regression tables
    """

    for _ in range(max_try_times):
        parser = JsonOutputParser(pydantic_object=TableDesign)
        
        combined_regression_descriptions = "\n".join([desc for desc, _ in regression_results])

        query = LangchainQueries.format_query(
            LangchainQueries.REGRESSION_TABLE_DESIGNER,
            research_topic=research_topic,
            regression_result=combined_regression_descriptions,
            number_of_results=len(regression_results) - 1
        )

        prompt = PromptTemplate(
            template="Answer the user query.\n{format_instructions}\n{query}\n",
            input_variables=["query"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        chain = prompt | model | parser

        output = chain.invoke({"query": query})

        if validate_design_regression_tables(output, len(regression_results)):
            return output

    return None

def validate_design_regression_tables(output: dict, number_of_results: int) -> bool:
    """
    Validate the design of regression tables.
    It verifies that all indices from 0 to the number of results are present without duplication.

    Args:
        output (TableDesign): The designed table output to validate.
        number_of_results (int): The total number of regression results.

    Returns:
        bool: True if the design is valid, False otherwise.
    """
    unique_numbers = set()

    for table in output["table_index"]:

        # Check if the table has more than 2 regression results
        if len(table) > 2:
            return False
        
        for index in table:
            # Check if index is within valid range
            if index < 0 or index >= number_of_results:
                return False
            
            # Check for duplicates
            if index in unique_numbers:
                return False
            
            unique_numbers.add(index)
    
    # Check if all indices from 0 to 'number_of_results - 1' are present
    return len(unique_numbers) == number_of_results