from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import copy

from .models import *
from ..static.langchain_query import LangchainQueries
from ..regression.panel_data import *


async def design_regression_tables(
    research_topic: str,
    regression_results: list[RegressionResult],
    model: ChatOpenAI,
    max_try_times: int = 2,
) -> TableDesign | None:
    """
    Design regression tables.
    This will use to combine tables together.
    """
    add_reg_descriptions(regression_results)
    for _ in range(max_try_times):
        parser = JsonOutputParser(pydantic_object=TableDesign)

        combined_regression_descriptions = "\n".join(
            [results.description for results in regression_results]
        )

        query = LangchainQueries.format_query(
            LangchainQueries.REGRESSION_TABLE_DESIGNER,
            research_topic=research_topic,
            regression_result=combined_regression_descriptions,
            number_of_results=len(regression_results) - 1,
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


def validate_design_regression_tables(
    output: TableDesign, number_of_results: int
) -> bool:
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
    table_design: TableDesign, number_of_tables: int = 0
) -> TableDesign:
    """
    Allow the user to use the command line to select multiple table designs to keep.
    Create a new TableDesign object with the selected table designs.

    Args:
        table_design (TableDesign): The designed table output to select from.
        number_of_tables (int): when larger than 0, the function will not ask the user to select the table designs. But return the first number_of_tables table designs.

    Returns:
        TableDesign: The selected table designs.
    """
    if number_of_tables > 0:
        assert number_of_tables <= len(
            table_design.table_index
        ), f"The number of tables to select is larger than the number of tables designed, which is {len(table_design.table_index)}."
        return TableDesign(
            table_index=copy.deepcopy(table_design.table_index[:number_of_tables]),
            table_regression_nums=copy.deepcopy(
                table_design.table_regression_nums[:number_of_tables]
            ),
            table_title=copy.deepcopy(table_design.table_title[:number_of_tables]),
            number_of_tables=number_of_tables,
        )

    def display_table_designs(table_design: TableDesign):
        for i, table in enumerate(table_design.table_index):
            print(f"Table {i + 1}: {table} {table_design.table_title[i]}")

    def get_user_selection() -> list[int]:
        while True:
            try:
                selection = input(
                    "Enter the numbers of the table designs to keep (e.g., 1,2,3) separated by commas: "
                )
                selected_indices = [
                    int(num.strip()) - 1 for num in selection.split(",")
                ]
                if all(
                    0 <= index < len(table_design.table_index)
                    for index in selected_indices
                ):
                    return selected_indices
                else:
                    print(
                        f"Invalid selection. Please enter numbers between 1 and {len(table_design.table_index)} separated by commas."
                    )
            except ValueError:
                print("Invalid input. Please enter valid numbers separated by commas.")

    display_table_designs(table_design)
    selected_indices = get_user_selection()

    selected_table_design = TableDesign(
        table_index=[table_design.table_index[i].copy() for i in selected_indices],
        table_regression_nums=[
            table_design.table_regression_nums[i] for i in selected_indices
        ],
        table_title=[table_design.table_title[i] for i in selected_indices],
        number_of_tables=0,
    )

    selected_table_design.number_of_tables = len(selected_table_design.table_index)

    return selected_table_design
