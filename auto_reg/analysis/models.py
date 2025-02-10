from pydantic import BaseModel, Field


class RegressionEquation(BaseModel):
    equation: str
    analysis: str


class RegressionAnalysis(BaseModel):
    analysis: str


class RegressionResultTable(BaseModel):
    latex_table: str = Field(description="regression result table", default="")


class ResultTables(BaseModel):
    tables: list[RegressionResultTable | None] = Field(
        default_factory=list
    )  # the regression result tables
    description: list[str] = Field(
        default=[]
    )  # the description of the regression results that are used to create the tables
    analysis: list[RegressionAnalysis | None] = Field(
        description="the analysis of the regression results that are used to create the tables",
        default=[],
    )

    def get_tables(self, index: list[int]) -> list[RegressionResultTable]:
        """
        Given a list of indices, return the corresponding regression result tables.
        """
        return [self.tables[i] for i in index]

    def get_description(self, index: list[int]) -> list[str]:
        """
        Given a list of indices, return the corresponding regression result descriptions.
        """
        return [self.description[i] for i in index]

    def get_analysis(self, index: list[int]) -> RegressionAnalysis:
        """
        Given a list of indices, return the corresponding regression result analysis.
        """
        selected_analysis = [self.analysis[i] for i in index]
        return RegressionAnalysis(
            analysis="\n".join([analysis.analysis for analysis in selected_analysis])
        )


class TableDesign(BaseModel):
    number_of_tables: int = Field(
        description="the number of regression tables I need create"
    )
    table_index: list[list[int]] = Field(
        description="the index used by each regression table using a list of list of integers. The list is as long as the number of regression tables. For each sublist, it contains the index of the regression results that should be combined into one table."
    )
    table_regression_nums: list[int] = Field(
        description="the number of regression for each table using a list of integers. The list is as long as the number of regression tables."
    )
    table_title: list[str] = Field(
        description="the title of each regression table using a list of strings, as well as how many columns(the regression numbers). The list is as long as the number of regression tables."
    )
