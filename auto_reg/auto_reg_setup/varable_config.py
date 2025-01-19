from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from ..static.langchain_query import LangchainQueries
from typing import Optional

# TODO: This file need to be rewrite

# Define regression model data structure
class RegressionModel(BaseModel):
    type: str = Field(default="basic_regression", description="Type of regression model")
    dependent_vars: list[str] = Field(description="List of dependent variables", default_factory=list)
    independent_vars: list[str] = Field(description="List of independent variables", default_factory=list) 
    effects: list[str] = Field(description="Fixed effects to include in regression, at most two effects", default_factory=list)
    control_vars: list[str] = Field(description="List of control variables", default_factory=list)
    constant: bool = Field(default=True, description="Whether to include constant term")


class ResearchTopic(BaseModel):
    research_topic: str = Field(description="Research topic", default="")
    entity_level: str = Field(description="One entity-level variable", default="")
    dependent_var_name: str = Field(description="Name of dependent variable", default="")
    dependent_var_description: str = Field(description="Description of dependent variable", default="")
    independent_var_name: str = Field(description="The only one name of independent variable", default="")
    independent_var_description: str = Field(description="The only one description of independent variable", default="")


class ControlVariables(BaseModel):
    control_vars_name: list[str] = Field(description="List of control variables", default_factory=list)
    control_vars_description: list[str] = Field(description="List of control variables", default_factory=list)


class NewVariable(BaseModel):
    new_variable_name: str = Field(description="Name of new variable", default="")
    is_dummy: bool = Field(description="Whether the new variable is a dummy variable", default=False)
    new_variable_description: str = Field(description="Description of new variable", default="")
    most_relevant_variable: str = Field(description="Most relevant variable - either dependent var, independent var, or None", default="None")

                                                                        
def generate_research_topic(
    description: str,
    language_model: ChatOpenAI,
) -> ResearchTopic:
    
    parser = JsonOutputParser(pydantic_object=ResearchTopic)
    query = LangchainQueries.format_query(
        LangchainQueries.RESEARCH_TOPIC_PROMPT,
        research_topic=description
    )

    prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | language_model | parser

    output = chain.invoke({"query": query})

    return output

def generate_control_variables(
    research_topic: ResearchTopic,
    language_model: ChatOpenAI,
) -> ControlVariables:
    
    parser = JsonOutputParser(pydantic_object=ControlVariables)

    independent_variable = research_topic["independent_var_name"] + " " + research_topic["independent_var_description"]
    dependent_variable = research_topic["dependent_var_name"] + " " + research_topic["dependent_var_description"]
    main_variables = independent_variable + " " + dependent_variable

    query = LangchainQueries.format_query(
        LangchainQueries.CONTROL_VARIABLE_GENERATION_PROMPT,
        research_topic=research_topic["research_topic"],
        main_variables=main_variables,
        entity_level=research_topic["entity_level"],
    )

    prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | language_model | parser

    output = chain.invoke({"query": query})

    return output


def generate_new_variable(
    research_topic: ResearchTopic,
    regression_type: str,
    new_variable_description: str,
    language_model: ChatOpenAI,
) -> NewVariable:

    parser = JsonOutputParser(pydantic_object=NewVariable)

    independent_variable = research_topic["independent_var_name"] + " " + research_topic["independent_var_description"]
    dependent_variable = research_topic["dependent_var_name"] + " " + research_topic["dependent_var_description"]
    main_variables = independent_variable + " " + dependent_variable

    query = LangchainQueries.format_query(
        LangchainQueries.NEW_VARIABLE_GENERATION_PROMPT,
        research_topic=research_topic["research_topic"],
        main_variables=main_variables,
        entity_level=research_topic["entity_level"],
        new_variable_description=new_variable_description,
        regression_type=regression_type,
    )

    prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | language_model | parser

    output = chain.invoke({"query": query})

    return output



def regression_model_config(
    vars_name: list[str],          # List of variable names available in dataset
    vars_description: list[str],    # List of descriptions for each variable
    language_model: ChatOpenAI,
) -> RegressionModel:              # Returns the RegressionModel structure
    
    # Setup prompt
    regression_input_query = f""" """
    prompt = prompt_setup()

    # Setup parser
    parser = JsonOutputParser(pydantic_object=RegressionModel)

    # Setup chain
    chain = prompt | language_model | parser

    # Run chain
    output = chain.invoke({"query": regression_input_query})

    return output

def prompt_setup():
    pass
