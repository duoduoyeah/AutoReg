class DefaultDict(dict):
    def __missing__(self, key):
        return ""


class LangchainQueries:
    """Class for storing prompts used with langchain."""
    
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
    
    CREATE_REGRESSION_TABLE_QUERY = (
        """
    Our topic is: {research_topic}
    Currently, we are running a regression analysis. 
    The configuration of the regression analysis is:
    {regression_config}

    Your task is to create a regression table in latex format for the following regression {number_of_results} results:
    {regression_result}

    The latex table should be in the following template format:
    {latex_table_template}
    You should write the regression results in the table.
    If there are even number of regression results, then there should be one regression result without controls and one with controls.
    Always use the odd column to write the regression result without controls.
    """
    )

    ANALYSIS_QUERY = """
    Our topic is: {research_topic}
    Currently, we are running the {analysis_type} analysis.
    The result of the analysis in table format is:
    {regression_result_table}

    Your task is to analyze the result by the table column by column and provide a summary of the analysis. Please focus on:
    1. The statistical significance and magnitude of coefficients
    2. The economic interpretation of the results
    3. How the results change when control variables are added
    4. Any notable patterns or trends across specifications
    Provide a clear and concise summary that highlights the key findings and their implications.
    Writing style should be concise, right-branch, and easy to understand.
    Your response language should be {language_used}.
    Your response should use latex format.
    """


    EQUATION_QUERY = """
    Our topic is: {research_topic}
    Currently, we are running the {analysis_type} analysis.
    The configuration of the regression analysis is:
    {regression_config}
    Your task is to write the regression equation in the latex format.

    """

    # r string is a raw string literal in Python that treats backslashes as literal characters
    TABLE_EXAMPLE = r"""
            \begin{table}[htbp]
            \caption{Regression Results Template Table}
            \label{Use the regression name as the label}
            \centering
            \begin{tabular}{p{3.6cm}p{3.6cm}p{3.6cm}} % Three columns of equal width totaling ~11cm
            \toprule
            & (1) & (2) \\
            Dependent Variable  & Name(replace_with_actual_variable_name)  & Name(replace_with_actual_variable_name) \\
            \midrule
            Independent Variable(replace_with_actual_variable_name)  & $\beta_1$*** & $\beta_2$*** \\
                        & ($t_1$) & ($t_2$) \\
            Control Variable(replace_with_actual_variable_name)     &  & $\beta_3$*** \\  % Each control variable should be on a new line
                        &  & ($t_3$) \\
            Constant    & $\beta_4$* & $\beta_5$ \\
                        & ($t_4$) & ($t_5$) \\

            Number of id       & X,XXX        & X,XXX \\
            Individual FE      & YES          & YES \\
            Year FE            & YES          & YES \\
            Observations       & XX,XXX       & XX,XXX \\
            R-squared          & 0.XXX        & 0.XXX \\
            \bottomrule
            \end{tabular}
            \begin{tablenotes}
            \small
            \item \textit{Note:} t-statistics are in parentheses; *, **, *** denote significance at the 10\%, 5\%, and 1\% levels, respectively.
            \end{tablenotes}
            \end{table}
            """

    TABLE_TEMPLATE_FOUR_COLUMNS = r"""

    """


    @staticmethod
    def format_query(query: str, **kwargs) -> str:
        return query.format_map(DefaultDict(kwargs))

