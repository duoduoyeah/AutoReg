import os


class DefaultDict(dict):
    def __missing__(self, key):
        return ""


class LangchainQueries:
    """Class for storing prompts used with langchain."""

    REGRESSION_TABLE_DESIGNER = """
    My research topic is: {research_topic}
    
    The following is the regression result, I devide them by index.
    Under each index, the number of regressions may not be one.
    The regressions:
    {regression_result}

    Your task is to tell me how many regression result tables should be created. The requirement is that:
    1. Include all index. The index number you use is from 0 to {number_of_results}.
    2. For each table, the number of regressions should be two.(Not index amounts, but the number of regressions)
    3. If there is only one regression available, then the number of regressions should be one.
    4. For each table, the regressions must be the same type(the same test or analysis type). For example, you could combine two robust regression indices into one table if they are both contain one regression result.
    
    You should return:
    1. the number of regression tables I need create.
    2. the index used by each regression table using a list of list of integers. The list is as long as the number of regression tables.
    For each sublist, it contains the index of the regression results that should be combined into one table.
    3. The number of regression for each table.
    4. Table title for each regression table.
    """

    RESEARCH_TOPIC_PROMPT = """
    I will conduct financial research and perform panel data regression to validate my findings.
    Your task is to propose an engaging financial-related research topic focused on: {research_topic}

    Please specify which independent variable influences which dependent variable. 
    Ensure that both the independent and dependent variables are measured per 
    entity over time, suitable for panel data regression analysis.
    Only provide one independent variable and one dependent variable.

    Also provide what entity level should be used for the regression analysis.

    For each independent variable, dependent variable provide a description of the variable.
    In the description, please include the unit of the variables, also from which data source the variables are obtained.
    """

    CONTROL_VARIABLE_GENERATION_PROMPT = """
    Your task is to generate six control variables that are relevant to the research topic.
    Our research topic is: {research_topic}
    Our main variables are: {main_variables}
    Our entity level is: {entity_level}

    For each control variable, provide a description of the variable.
    In the description, please include the unit of the variables, also from which data source the variables are obtained.

    The control variables should be entity-level variables.
    Entity-level means that the variable is measured per entity over time.
    For example, if the variable is the size of the company, it is measured per company over time.

    """

    NEW_VARIABLE_GENERATION_PROMPT = """
    Our main research topic is:{research_topic}.
    Our main variables are: {main_variables}

    Your task is to generate a new variable that: {new_variable_description}.
    Your generated variable will be used in {regression_type} analysis.

    You should also provide the most relevant variable(use variable name) that your generated variable is correlated with.
    For example, if your generated variable is another measure method of the dependent variable, then the most relevant variable is the dependent variable.
    If no variable is relevant, then the most relevant variable is "None".

    The new variable name should be descriptive.
    The new variable description should concisely describe the unit of the variable and the data source of the variable.
    """

    BASIC_REGRESSION_CONFIG_PROMPT = """Your task is to use the variable descriptions to set up a {model_type}.
    Our main topic is:{research_topic}.

    Available variables and their descriptions:
    {variables_description}

    For your reference, we have completed a {previous_model_type}.
    The configuration of the previous model is:
    {previous_result_configuration}

    Please analyze these variables and set up an appropriate {model_type} structure."""

    REGRESSION_TABLE_QUERY = """
    Your task is to create one single regression table in latex format for the following regression {number_of_results} results.
    
    The title of the table should describe the regression purpose.
===============================================
For your reference: 
    The regression settings are:
    {regression_config}
    The regression's description is:
    {regression_description}
    The regression's results are in below:
    {regression_result}
===============================================
    You should return the latex table following template format:
    {latex_table_template}
===============================================
    The requirements are:
    1. Replace all variable names in the first column of the table with actual variable names.
    2. Replace all placeholder with actual parameter values.
    3. Don't use underline in latex, use space instead.
    4. One column, one regression result.
    """

    COMBINE_REGRESSION_TABLE_QUERY = """
    Your task is to create one single regression table in latex format by combining the following regression tables.
    Requirement:
    The width of return table should be within 10cm. 
    The title of the new table is: {table_title}.

    The tables you should combine are:
    {regression_tables}
    
    
    """

    ANALYSIS_QUERY = """
    Your task is to analyze the regression result and write analysis.
    
    The configuration of the regression is:
    {regression_config}

    The description of the regression is:
    {regression_description}

    The result of the analysis in table format is:
    {regression_table}

    Your task is to analyze the result by the table column by column and provide a summary of the analysis. Please focus on:
    1. The statistical significance and magnitude of coefficients
    2. The economic interpretation of the results
    3. How the results change when control variables are added

    Your analysis requirement:
    1. Provide a clear and concise summary that highlights the key findings and their implications.
    2. Writing style should be concise, right-branch, and easy to understand.
    3. Your response language should be {language_used}.
    4. Your response should use latex format.
    5. Don't analyze control variables and constant.
    6. Within 200 words for each regression column in the table.
    """

    EQUATION_QUERY = """
    Our topic is: {research_topic}
    Currently, we are running the {analysis_type} analysis.
    The configuration of the regression analysis is:
    {regression_config}
    Your task is to write the regression equation in the latex format.
    """

    current_folder = os.path.dirname(os.path.abspath(__file__))
    latex_folder = os.path.join(current_folder, "latex")
    with open(os.path.join(latex_folder, "basic.tex"), "r") as file:
        BASIC_TABLE = file.read()

    with open(os.path.join(latex_folder, "iv.tex"), "r") as file:
        IV_TABLE = file.read()

    with open(os.path.join(latex_folder, "group.tex"), "r") as file:
        GROUP_TABLE = file.read()

    @staticmethod
    def format_query(query: str, **kwargs) -> str:
        return query.format_map(DefaultDict(kwargs))


if __name__ == "__main__":
    print(LangchainQueries.BASIC_TABLE, end="\n\n\n\n\n\n")
    print(LangchainQueries.IV_TABLE, end="\n\n\n\n\n\n")
    print(LangchainQueries.GROUP_TABLE, end="\n\n\n\n\n\n")
