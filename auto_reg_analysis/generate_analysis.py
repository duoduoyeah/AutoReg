from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import os
from ..static.langchain_query import LangchainQueries
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

class RegressionAnalysis(BaseModel):
    analysis: str = Field(description="regression result analysis", default="")

class RegressionEquation(BaseModel):
    equation: str = Field(description="regression equation", default="")
    analysis: str = Field(description="regression result analysis", default="")

def analyze_regression_results(
        regression_name: str,
        regression_result_table: str,
        model: ChatOpenAI,
        language_used: str = "Chinese",
) -> RegressionAnalysis:
    """
    Analyze regression results.
    
    Returns:
        RegressionAnalysis: The regression result analysis.
    """
    RESEARCH_TOPIC = os.getenv("RESEARCH_TOPIC")
    if RESEARCH_TOPIC is None:
        raise ValueError("RESEARCH_TOPIC is not set")
    
    parser = JsonOutputParser(pydantic_object=RegressionAnalysis)

    query = LangchainQueries.format_query(
        LangchainQueries.ANALYSIS_QUERY,
        research_topic=RESEARCH_TOPIC,
        analysis_type= regression_name,
        regression_result_table=regression_result_table,
        language_used=language_used
    )

    prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | model | parser

    output = chain.invoke({"query": query})

    return output