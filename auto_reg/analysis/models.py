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
        default=[]
    ) 
    analysis: list[RegressionAnalysis] = Field(
        description="the analysis of the regression results that are used to create the tables",
        default=[],
    )

    def assert_valid(self):
        if not (len(self.tables) == len(self.description) == len(self.analysis)):
            raise ResultTableError
        
    def get_tables(self, index: list[int]) -> list[RegressionResultTable]:
        return [self.tables[i] for i in index]

    def get_description(self, index: list[int]) -> list[str]:
        return [self.description[i] for i in index]

    def get_analysis(self, index: list[int]) -> RegressionAnalysis:
        selected_analysis = [self.analysis[i] for i in index]
        return RegressionAnalysis(
            latex_analysis="\n".join([analysis.latex_analysis for analysis in selected_analysis])
        )

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
    number_of_tables: int = Field()
    table_index: list[list[int]] = Field()
    table_regression_nums: list[int] = Field()
    table_title: list[str] = Field()
