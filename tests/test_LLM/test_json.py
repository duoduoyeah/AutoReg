import os
from getpass import getpass
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


# Define your desired data structure.
class Joke(BaseModel):
    setup: str = Field(description="question to set up a joke")
    punchline: str = Field(description="answer to resolve the joke")
    total_words: int = Field(description="total number of words in the punchline")


# And a query intented to prompt a language model to populate the data structure.
joke_query = "Tell me a joke."

# Set up a parser + inject instructions into the prompt template.
parser = JsonOutputParser(pydantic_object=Joke)

prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = prompt | model | parser

output = chain.invoke({"query": joke_query})

print(output)