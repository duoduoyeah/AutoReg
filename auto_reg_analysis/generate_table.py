from pydantic import BaseModel, Field
from linearmodels.panel.results import PanelEffectsResults
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from ..static.langchain_query import LangchainQueries
from langchain_openai import ChatOpenAI
import os

class RegressionResultTable(BaseModel):
    table: str = Field(description="regression result table", default="")


def generate_econometric_analysis_table(
    regression_config: str,
    regression_results: list[PanelEffectsResults],
    model: ChatOpenAI,
    table_template: str|None = None
) -> RegressionResultTable:
    """
    Generate econometric analysis table.
    
    Returns:
        RegressionResultTable: The regression result table.

    Raises:
        ValueError: If RESEARCH_TOPIC is not set.
    """
    # setup prompt
    parser = JsonOutputParser(pydantic_object=RegressionResultTable)

    if table_template is None:
        table_template = LangchainQueries.TABLE_EXAMPLE

    

    RESEARCH_TOPIC = os.getenv("RESEARCH_TOPIC")
    if RESEARCH_TOPIC is None:
        raise ValueError("RESEARCH_TOPIC is not set")

    query = LangchainQueries.format_query(
        LangchainQueries.CREATE_REGRESSION_TABLE_QUERY,
        research_topic=RESEARCH_TOPIC,
        regression_config=regression_config,
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