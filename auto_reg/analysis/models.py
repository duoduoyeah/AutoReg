from pydantic import BaseModel, Field
from ..errors import ResultTableError

class RegressionEquation(BaseModel):
    equation: str
    analysis: str


class RegressionAnalysis(BaseModel):
    latex_analysis: str


class RegressionResultTable(BaseModel):
    latex_table: str = Field(description="regression result table", default="")


class ResultTables(BaseModel):
    tables: list[RegressionResultTable] = Field(
        default_factory=list
    )  # the regression result tables
    description: list[str] = Field(
        default_factory=list
    ) 
    analysis: list[RegressionAnalysis] = Field(
        description="the analysis of the regression results that are used to create the tables",
        default_factory=list,
    )
    
    def get_length(self):
        return len(self.tables)

    def assert_valid(self):
        if not (len(self.tables) == len(self.description) == len(self.analysis)):
            raise ResultTableError
        
    def get_tables(self, index: list[int]) -> list[RegressionResultTable]:
        return [self.tables[i] for i in index]

    def get_description(self, index: list[int]) -> list[str]:
        return [self.description[i] for i in index]

    def combine_analysis(self, index: list[int]) -> RegressionAnalysis:
        try: 
            combined_analysis = ""
            for i in index:
                combined_analysis += self.analysis[i].latex_analysis + "\n"
            return RegressionAnalysis(
                latex_analysis=combined_analysis.strip()  # Remove trailing newline
            )
        except Exception as e:
            print(f"error in ResultTables.combine_analysis({index}): {e}")
            return RegressionAnalysis(latex_analysis="") # Return empty analysis instead of None

    def get_analysis(self, index: list[int]) -> list[RegressionAnalysis]:
        try:
            return [self.analysis[i] for i in index]
        except Exception as e:
            print(f"Special error in ResultTables.get_analysis({index}): {e}")
            return None

    def iterate_table(self):
        for i in range(len(self.tables)):
            yield (self.tables[i], self.description[i], self.analysis[i])


class TableDesign(BaseModel):
    """
    Table design for regression tables.

    Attributes:
        number_of_tables (int): The number of regression tables to create.
        table_index (list[list[int]]): The index used by each regression table using a list of list of integers. The list is as long as the number of regression tables. For each sublist, it contains the index of the regression results that should be combined into one table.
        table_regression_nums (list[int]): The number of regressions for each table using a list of integers. The list is as long as the number of regression tables.
        table_title (list[str]): The title of each regression table using a list of strings, as well as how many columns (the regression numbers). The list is as long as the number of regression tables.
    """
    number_of_tables: int = Field(description="the number of regression tables to create")
    table_index: list[list[int]] = Field()
    table_regression_nums: list[int] = Field()
    table_title: list[str] = Field()
