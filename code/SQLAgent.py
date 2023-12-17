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
from pydantic import BaseModel, Field

db = SQLDatabase.from_uri("sqlite:////Users/alexgagliano/Documents/Research/LLMs/data/2021mwb.db")

#class PhaseInput(BaseModel):#
#    obsMJD: float = Field(description="The MJD date of an observation.")

@tool("getPhase")#, args_schema=PhaseInput)
def getPhase(obsMJD: float) -> float:
    """Get the date of an observation relative to the current date."""
    t_mjd = Time(date.today().strftime("%Y-%m-%dT%H:%M:%S"), format='isot', scale='utc').mjd
    return t_mjd - float(obsMJD)

@tool
def savePhotometry(SNname: str) -> str:
    """Saves sql-returned photometry for a supernova for use later with light curve fitting."""

@tool
def fitGP_toPhotometry(SNname: str, redshift: float, extinction: float) -> float:
    """Fits a gaussian process model to a set of photometry."""

    fn_call = "extrabol '%s.dat' -z %.3f -verbose --ebv %.2f"%(SNname, redshift, extinction)
    os.system(fn_call)

    #read in output

    return t_mjd

custom_tools = [getPhase]

toolkit = SQLDatabaseToolkit(db=db, llm=OpenAI(temperature=0))

agent_executor = create_sql_agent(
    llm=OpenAI(temperature=0),
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    extra_tools=custom_tools
)


agent_executor.run("The data is photometry associated with a supernova in apparent magnitudes, and date in mjd. How many days ago was the brightest observation in the dataset, and in what filter?")