import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = os.getenv("DEEPSEEK_API_KEY")
    os.environ["OPENAI_API_BASE"] = os.getenv("DEEPSEEK_API_BASE")



from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

model = ChatOpenAI(
    model="deepseek-chat",
    temperature=0
)


# Define regression model data structure
class RegressionModel(BaseModel):
    type: str = Field(default="basic_regression", description="Type of regression model")
    dependent_vars: list[str] = Field(description="List of dependent variables", default_factory=list)
    independent_vars: list[str] = Field(description="List of independent variables", default_factory=list) 
    effects: list[str] = Field(description="Fixed effects to include in regression, at most two effects", default_factory=list)
    control_vars: list[str] = Field(description="List of control variables", default_factory=list)
    constant: bool = Field(default=True, description="Whether to include constant term")

vars_name: list[str] = ["entity_id", "time", "cloud_investment", "revenue", "total_assets", "market_share", "rd_spend", "op_costs", "gdp_growth", "employee_count", "regional_econ", "city"]

vars_description: list[str] = ["Unique identifier for each company in the dataset","Year of observation for each company","Investment in cloud computing by company 'i' in year 't'","Revenue of company 'i' in year 't'","Total assets of the company","Market share or competitive position within the industry","R&D expenditure","Operational costs","Economic conditions (e.g., GDP growth rate)","Employee count or workforce size","Geographic factors (e.g., regional economic conditions)","City where company 'i' is located"]


# And a query intented to prompt a language model to populate the data structure.
regression_input_query = f"""Your task is to use the variable descriptions to set up a regression model for me.

I aims to investigate the relationship between a company's investment in cloud computing and its revenue over time using panel data regression analysis.

Available variables and their descriptions:
{[f"{name}: {desc}" for name, desc in zip(vars_name, vars_description)]}

Please analyze these variables and set up an appropriate regression model structure."""

# Set up a parser + inject instructions into the prompt template.
parser = JsonOutputParser(pydantic_object=RegressionModel)

prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = prompt | model | parser

output = chain.invoke({"query": regression_input_query})

print(output)
