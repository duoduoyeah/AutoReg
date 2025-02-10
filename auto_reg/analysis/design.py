from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import copy

from .models import *
from ..static import LangchainQueries
from ..regression.panel_data import *
from ..errors import DesignError, ChainConfigurationError
from ..language_model import run_chain

async def design_regression_tables(
    research_topic: str,
    regression_results: list[RegressionResult],
    model: ChatOpenAI,
    max_try_times: int = 3,
) -> TableDesign | None:
    """
    Design regression tables.
    This will use to combine tables together.
    """
    add_reg_descriptions(regression_results)
    for _ in range(max_try_times):
        try:
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
        except Exception:
            raise ChainConfigurationError(extra_info={
                "error place": "design regression",
                })

        try:
            output = await run_chain(
                chain, 
                query,
                RegressionResultTable, 
                "draw_table",)
            validate_design_regression_tables(output, len(regression_results))
            return output
        except Exception as e:
            print(e)
            continue
        
    remove_reg_descriptions(regression_results)
    return output


def validate_design_regression_tables(
    output: TableDesign, number_of_results: int
) -> None:
    """
    Validate the design of regression tables.
    It verifies that all indices from 0 to the number of results
    are present without duplication.
    
    Args:
        output (TableDesign): The designed table output to validate.
        number_of_results (int): The total number of regression results.
        
    Raises:
        DesignError: If the table design is invalid
    """
    unique_numbers = set()

    for table in output.table_index:
        # Check if the table has more than 2 regression results
        if len(table) > 2:
            raise DesignError({"error": "Table cannot have more than 2 regression results"})

        for index in table:
            # Check if index is within valid range
            if index < 0 or index >= number_of_results:
                raise DesignError({"error": f"Invalid index {index} outside range 0 to {number_of_results-1}"})

            # Check for duplicates
            if index in unique_numbers:
                raise DesignError({"error": f"Duplicate index {index} found"})

            unique_numbers.add(index)

    # Check if all indices from 0 to 'number_of_results - 1' are present
    if len(unique_numbers) != number_of_results:
        raise DesignError({"error": "Not all regression results are used in tables"})
    elif output.number_of_tables != len(output.table_index):
        raise DesignError({"error": "Number of tables does not match table index length"})


def select_table_design(
    table_design: TableDesign, number_of_tables: int = 0
) -> TableDesign:
    """
    Allow the user to use the command line to select multiple table designs to keep.
    Create a new TableDesign object with the selected table designs.

    Args:
        table_design (TableDesign): The designed table output to select from.
        number_of_tables (int): when larger than 0, the function will not ask
        the user to select the table designs.
        But return the first number_of_tables table designs.

    Returns:
        TableDesign: The selected table designs.
    """
    if number_of_tables > 0:
        assert number_of_tables <= len(
            table_design.table_index
        ), f"The number of tables to select is larger than the number of \
            tables designed, which is {len(table_design.table_index)}."
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
                    "Enter the numbers of the table designs to keep (e.g., 1,2,3)\
                    separated by commas: "
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
                        f"Invalid selection. Please enter numbers \
                    between 1 and {len(table_design.table_index)} separated by commas."
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
