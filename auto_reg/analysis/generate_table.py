# Standard imports
import asyncio

# Langchain imports
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI

# Regression Package imports
from linearmodels.panel.results import PanelEffectsResults

# Local imports
from ..regression.regression_config import RegressionConfig
from ..regression.panel_data import *
from ..static.langchain_query import LangchainQueries
from .models import RegressionAnalysis, RegressionResultTable, ResultTables, TableDesign


async def draw_table(
    regression_description: str,
    regression_results: list[PanelEffectsResults],
    regression_config: RegressionConfig,
    model: ChatOpenAI,
    table_template: str,
    query: str,
    max_try_times: int = 2,
) -> RegressionResultTable:
    """
    Draw a table for the regression results.
    """
    for attempt in range(max_try_times):
        try:
            # setup prompt
            parser = JsonOutputParser(pydantic_object=RegressionResultTable)

            query = LangchainQueries.format_query(
                query,
                regression_config=str(regression_config),
                regression_description=regression_description,
                regression_result=str(regression_results),
                latex_table_template=table_template,
                number_of_results=len(regression_results),
            )

            prompt = PromptTemplate(
                template="Answer the user query.\n{format_instructions}\n{query}\n",
                input_variables=["query"],
                partial_variables={
                    "format_instructions": parser.get_format_instructions()
                },
            )

            chain = prompt | model | parser

            output = await chain.ainvoke({"query": query})

            output = RegressionResultTable.model_validate(output)

            return output
        except Exception as e:
            print(
                f"Error drawing table on attempt {attempt + 1}: {e}\n the tables is {regression_description}"
            )

    return RegressionResultTable(latex_table="")


def get_table_template(regression_type: str) -> str:
    """
    Get the table template based on the regression type.
    """
    if regression_type == get_function_name(panel_regression):
        return LangchainQueries.BASIC_TABLE
    elif regression_type == get_function_name(two_stage_regression):
        return LangchainQueries.IV_TABLE
    elif regression_type == get_function_name(group_regression):
        return LangchainQueries.GROUP_TABLE
    else:
        raise ValueError(f"Invalid regression type: {regression_type}")


async def generate_empty_tables():
    return RegressionResultTable(latex_table="")


async def draw_tables(
    all_reg_results: list[RegressionResult],
    design: TableDesign,
    model: ChatOpenAI,
    result_tables: ResultTables,
) -> None:
    """
    Draw tables for each regression result.
    """
    table_tasks = []
    table_descriptions = []
    used_regression_result: list[int] = get_used_regression_result(design)

    for i in range(len(all_reg_results)):
        if i not in used_regression_result:
            table_tasks.append(generate_empty_tables())
            table_descriptions.append(all_reg_results[i].description)
        else:
            regression_description: str = all_reg_results[i].description
            regression_results: list[PanelEffectsResults] = all_reg_results[i].results
            regression_config: RegressionConfig = all_reg_results[i].regression_config
            table_template = get_table_template(all_reg_results[i].regression_type)
            query = LangchainQueries.REGRESSION_TABLE_QUERY

            table_tasks.append(
                draw_table(
                    regression_description=regression_description,
                    regression_results=regression_results,
                    regression_config=regression_config,
                    model=model,
                    table_template=table_template,
                    query=query,
                )
            )
            table_descriptions.append(regression_description)

    results: list[RegressionResultTable] = await asyncio.gather(*table_tasks)
    assert len(results) == len(table_descriptions)

    result_tables.tables = results
    result_tables.description = table_descriptions


async def combine_table(
    table_title: str,
    combine_tables: list[RegressionResultTable],
    model: ChatOpenAI,
    query: str,
    max_try_times: int = 2,
) -> RegressionResultTable:
    """
    Combine multiple tables into one table.
    """
    if len(combine_tables) == 1:
        return combine_tables[0]

    for attempt in range(max_try_times):
        try:
            # setup prompt
            parser = JsonOutputParser(pydantic_object=RegressionResultTable)

            query = LangchainQueries.format_query(
                query,
                table_title=table_title,
                regression_tables="\n".join(
                    [table.latex_table for table in combine_tables]
                ),
            )

            prompt = PromptTemplate(
                template="Answer the user query.\n{format_instructions}\n{query}\n",
                input_variables=["query"],
                partial_variables={
                    "format_instructions": parser.get_format_instructions()
                },
            )

            chain = prompt | model | parser

            output = await chain.ainvoke({"query": query})
            output = RegressionResultTable.model_validate(output)
            return output
        except Exception as e:
            print(
                f"Error drawing table on attempt {attempt + 1}: {e}\n the tables is {table_title}"
            )

    return RegressionResultTable(latex_table="")


async def combine_tables(
    tables: ResultTables,
    design: TableDesign,
    model: ChatOpenAI,
) -> ResultTables:
    """
    Combine tables together.
    """
    combine_tasks = []
    analysis: list[RegressionAnalysis] = []
    for i in range(design.number_of_tables):
        combine_tasks.append(
            combine_table(
                table_title=design.table_title[i],
                combine_tables=tables.get_tables(design.table_index[i]),
                model=model,
                query=LangchainQueries.COMBINE_REGRESSION_TABLE_QUERY,
            )
        )
        analysis.append(tables.get_analysis(design.table_index[i]))

    combined_tables: list[RegressionResultTable] = await asyncio.gather(*combine_tasks)

    assert len(combined_tables) == len(analysis)
    assert len(combined_tables) == design.number_of_tables

    result_tables = ResultTables(
        tables=combined_tables,
        index=None,
        description=design.table_title,
        analysis=analysis,
    )
    return result_tables


def get_used_regression_result(design: TableDesign) -> list[int]:
    """
    Get the regression results that are used to create the tables.
    """
    used_regression_result = []
    used_regression_result = []
    for i in range(len(design.table_index)):
        used_regression_result.extend([index for index in design.table_index[i]])
    return used_regression_result


async def analyze_regression_result(
    regression_config: RegressionConfig,
    regression_description: str,
    regression_table: str,
    model: ChatOpenAI,
    language_used: str = "Chinese",
) -> RegressionAnalysis:
    """
    Analyze regression results.

    Information to be given to the language model:
    - research topic
    - previous analysis as reference
    - regression result table
    - language used

    Information returned by the language model:
    - regression result analysis as a string in latex format

    Returns:
        RegressionAnalysis: The regression result analysis.
    """
    try:
        parser = JsonOutputParser(pydantic_object=RegressionAnalysis)

        query = LangchainQueries.format_query(
            LangchainQueries.ANALYSIS_QUERY,
            regression_config=str(regression_config),
            regression_description=regression_description,
            regression_table=regression_table,
            language_used=language_used,
        )

        prompt = PromptTemplate(
            template="Answer the user query.\n{format_instructions}\n{query}\n",
            input_variables=["query"],
            partial_variables={
                "format_instructions": parser.get_format_instructions()
            },
        )

        chain = prompt | model | parser

        output = await chain.ainvoke({"query": query})

        output = RegressionAnalysis.model_validate(output)

        return output
    except Exception as e:
        print(
            f"Error analyzing regression result: {e}"
        )

    return RegressionAnalysis(analysis="")


async def generate_empty_analysis() -> RegressionAnalysis:
    """
    Generate an empty analysis.
    """
    return RegressionAnalysis(analysis="")


async def analyze_regression_results(
    regression_results: list[RegressionResult],
    design: TableDesign,
    result_tables: ResultTables,
    model: ChatOpenAI,
    language_used: str = "English",
) -> None:
    """
    Analyze regression results with table
    """
    analysis_tasks = []
    used_regression_result: list[int] = get_used_regression_result(design)

    for i in range(len(result_tables.tables)):
        # if the regression result is not used, generate an empty analysis
        if i not in used_regression_result:
            analysis_tasks.append(generate_empty_analysis())
        else:
            analysis_tasks.append(
                analyze_regression_result(
                    regression_config=regression_results[i].regression_config,
                    regression_description=result_tables.description[i],
                    regression_table=result_tables.tables[i].latex_table,
                    model=model,
                    language_used=language_used,
                )
            )

    result_tables.analysis = await asyncio.gather(*analysis_tasks)
