# Standard imports
from pydantic import BaseModel, Field
import asyncio

# Langchain imports
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI

# Regression Package imports
from linearmodels.panel.results import PanelEffectsResults

# Local imports
from ..auto_reg_setup.regression_config import RegressionConfig, ResearchConfig
from ..reg_model.panel_data import *
from ..static.langchain_query import LangchainQueries



class RegressionResultTable(BaseModel):
    latex_table: str = Field(description="regression result table", default="")


class ResultTables(BaseModel):
    tables: list[RegressionResultTable|None] # the regression result tables
    index: dict[int, int] | None # the key is the index of original regression results, the value is the index of the regression result tables
    description: list[str]  # the description of the regression results that are used to create the tables

    def get_tables(self, index: list[int]) -> list[RegressionResultTable]:
        if self.index is None:
            return self.tables
        else:
            return [self.tables[self.index[i]] for i in index]

    def get_description(self, index: list[int]) -> list[str]:
        if self.index is None:
            return self.description
        else:
            return [self.description[self.index[i]] for i in index]
    

class TableDesign(BaseModel):
    number_of_tables: int = Field(description="the number of regression tables I need create")
    table_index: list[list[int]] = Field(description="the index used by each regression table using a list of list of integers. The list is as long as the number of regression tables. For each sublist, it contains the index of the regression results that should be combined into one table.")
    table_regression_nums: list[int] = Field(description="the number of regression for each table using a list of integers. The list is as long as the number of regression tables.")
    table_title: list[str] = Field(description="the title of each regression table using a list of strings, as well as how many columns(the regression numbers). The list is as long as the number of regression tables.")


async def draw_table(
        regression_description: str,
        regression_results: list[PanelEffectsResults],
        regression_config: RegressionConfig,
        model: ChatOpenAI,
        table_template: str,
        query: str,
        max_try_times: int = 2
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
                regression_config=regression_config.get_vars(),
                regression_description=regression_description,
                regression_result=str(regression_results),
                latex_table_template=table_template,
                number_of_results=len(regression_results),
            )

            prompt = PromptTemplate(
                template="Answer the user query.\n{format_instructions}\n{query}\n",
                input_variables=["query"],
                partial_variables={"format_instructions": parser.get_format_instructions()},
            )

            chain = prompt | model | parser

            output = await chain.ainvoke({"query": query})

            output = RegressionResultTable.model_validate(output)

            return output
        except Exception as e:
            print(f"Error drawing table on attempt {attempt + 1}: {e}\n the tables is {regression_description}")

    return None



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

async def draw_tables(
        all_reg_results: list[RegressionResult],
        design: TableDesign,
        model: ChatOpenAI,
) -> ResultTables:
    """
    Draw tables for each regression result.
    """
    table_tasks = []
    table_descriptions = []
    used_regression_result: list[int] = get_used_regression_result(design)
    index_dict: dict[int, int] = {}
    count = 0
    for i in used_regression_result:
        index_dict[i] = count
        count += 1
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
                query=query
            )
        )
        table_descriptions.append(regression_description)
        # TODO: delete this print
        print(f"Table {count} is {regression_description}")

    results: list[RegressionResultTable] = await asyncio.gather(*table_tasks)
    
    assert count == len(used_regression_result)

    table_results = ResultTables(
        tables=results,
        index=index_dict,
        description=table_descriptions
    )
    return table_results


async def combine_table(
        table_title: str,
        combine_tables: list[RegressionResultTable],
        model: ChatOpenAI,
        query: str,
        max_try_times: int = 2
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
                regression_tables="\n".join([table.latex_table for table in combine_tables]),
            )

            prompt = PromptTemplate(
                template="Answer the user query.\n{format_instructions}\n{query}\n",
                input_variables=["query"],
                partial_variables={"format_instructions": parser.get_format_instructions()},
            )

            chain = prompt | model | parser

            output = await chain.ainvoke({"query": query})

            output = RegressionResultTable.model_validate(output)

            return output
        except Exception as e:
            print(f"Error drawing table on attempt {attempt + 1}: {e}\n the tables is {table_title}")

    return None


async def combine_tables(
        tables: ResultTables,
        design: TableDesign,
        model: ChatOpenAI,
        ) -> ResultTables:
    """
    Combine tables together.
    """
    combine_tasks = []
    for i in range(design.number_of_tables):
        combine_tasks.append(
            combine_table(
                table_title=design.table_title[i],
            combine_tables=tables.get_tables(design.table_index[i]),
            model=model,
            query=LangchainQueries.COMBINE_REGRESSION_TABLE_QUERY
        )
    )


    combined_tables: list[RegressionResultTable] = await asyncio.gather(*combine_tasks)
    result_tables = ResultTables(
        tables=combined_tables,
        index=None,
        description=design.table_title
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



async def design_regression_tables(
        research_topic: str,
        regression_results: list[RegressionResult],
        model: ChatOpenAI,
        max_try_times: int = 2
        ) -> TableDesign | None:
    """
    Design regression tables.
    This will use to combine tables together.
    """
    add_reg_descriptions(regression_results)
    for _ in range(max_try_times):
        parser = JsonOutputParser(pydantic_object=TableDesign)
        
        combined_regression_descriptions = "\n".join([results.description for results in regression_results])

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

        output = await chain.ainvoke({"query": query})

        output = TableDesign.model_validate(output)

        if validate_design_regression_tables(output, len(regression_results)):
            print(f"Get valid table design")
            break

    remove_reg_descriptions(regression_results)
    return output

def validate_design_regression_tables(output: TableDesign, number_of_results: int) -> bool:
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

    for table in output.table_index:
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
    if len(unique_numbers) != number_of_results:
        return False
    elif output.number_of_tables != len(output.table_index):
        return False
    return True


def select_table_design(
        table_design: TableDesign
        ) -> TableDesign:
    """
    Allow the user to use the command line to select multiple table designs to keep.

    Args:
        table_design (TableDesign): The designed table output to select from.

    Returns:
        TableDesign: The selected table designs.
    """

    def display_table_designs(table_design: TableDesign):
        for i, table in enumerate(table_design.table_index):
            print(f"Table {i + 1}: {table} {table_design.table_title[i]}")

    def get_user_selection() -> list[int]:
        while True:
            try:
                selection = input("Enter the numbers of the table designs to keep (e.g., 1,2,3) separated by commas: ")
                selected_indices = [int(num.strip()) - 1 for num in selection.split(',')]
                if all(0 <= index < len(table_design.table_index) for index in selected_indices):
                    return selected_indices
                else:
                    print(f"Invalid selection. Please enter numbers between 1 and {len(table_design.table_index)} separated by commas.")
            except ValueError:
                print("Invalid input. Please enter valid numbers separated by commas.")

    display_table_designs(table_design)
    selected_indices = get_user_selection()

    selected_table_design = TableDesign(
        table_index=[table_design.table_index[i] for i in selected_indices],
        table_regression_nums=[table_design.table_regression_nums[i] for i in selected_indices],
        table_title=[table_design.table_title[i] for i in selected_indices],
        number_of_tables=0
    )

    selected_table_design.number_of_tables = len(selected_table_design.table_index)

    return selected_table_design

