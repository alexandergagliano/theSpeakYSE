from langchain.llms import OpenAI, Ollama, HuggingFaceHub
from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_sql_agent
from langchain.agents.agent_types import AgentType
from langchain.agents import tool
from astropy.time import Time
from datetime import date
import os
from langchain.tools.render import format_tool_to_openai_function

db = SQLDatabase.from_uri("sqlite:////Users/alexgagliano/Documents/Research/LLMs/data/2021mwb.db")

@tool
def current_date_in_mjd() -> float:
    """Computes the current mjd date."""
    t_mjd = Time(date.today().strftime("%Y-%m-%dT%H:%M:%S"), format='isot', scalfe='utc').mjd

    return t_mjd

@tool
def savePhotometry(SNname) -> string:
    """Saves ."""

@tool
def fitGP_toPhotometry(SNname) -> float:
    """Returns the length of a word."""

    t_mjd = Time(date.today().strftime("%Y-%m-%dT%H:%M:%S"), format='isot', scalfe='utc').mjd

    return t_mjd

#llm_with_tools = llm.bind(functions=[SQLDatabaseToolkit, current_date_in_mjd])
custom_tools = [current_date_in_mjd]

toolkit = SQLDatabaseToolkit(db=db, llm=OpenAI(temperature=0))

agent_executor = create_sql_agent(
    llm=OpenAI(temperature=0),
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    extra_tools=custom_tools
)


agent_executor.run("The data is photometry associated with a supernova in apparent magnitudes, and date in mjd. How many days ago relative to today was the brightest observation in the dataset, and in what filter?"%t_mjd)
