from langchain.utilities import SQLDatabase
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import LlamaCpp
from langchain.agents import (
    tool,
    Tool,
    AgentExecutor,
)
from astropy.time import Time
from datetime import date
import os
from astropy.io import ascii
import numpy as np
import ast

from langchain.chains import LLMChain

from ravenAPI import *
from ravenAgent import *

import langchain
langchain.verbose = True

n_gpu_layers = 1  # Metal set to 1 is enough.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

db = SQLDatabase.from_uri("sqlite:////Users/kdesoto/python_repos/theSpeakYSE/data/2021mwb.db")

@tool
def get_phase(obsMJD: float) -> float:
    """Get the date of an observation relative to the current date."""
    t_mjd = Time(date.today().strftime("%Y-%m-%dT%H:%M:%S"), format='isot', scale='utc').mjd
    return t_mjd - float(obsMJD)

@tool
def save_photometry(photometry: str) -> str:
    """Saves a photometry query result to a text file for later manipulation."""
    print(photometry)
    photList = ast.literal_eval(photometry)
    photList = list(zip(*photList))
    #assume AB mag for now - change later!
    testList+ [('AB',)*len(testList)]
    ascii.write(photList, 'tempFile.txt', comment=False, overwrite=True)
    return 'Done.'

@tool
def fitGP_toPhotometry(SNname: str, redshift: float, extinction: float) -> float:
    """Fits a gaussian process model to a set of photometry."""

    fn_call = "extrabol '%s.dat' -z %.3f -verbose --ebv %.2f"%(SNname, redshift, extinction)
    os.system(fn_call)

    #read in output

    return t_mjd

custom_tools = [get_phase, save_photometry]

n_gpu_layers = 1  # Metal set to 1 is enough.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

LLAMA_MODEL_PATH = "/Users/kdesoto/Downloads/nexusraven-v2-13b.Q5_K_S.gguf"
# Make sure the model path is correct for your system!

tools = custom_tools

llm = LlamaCpp(
    model_path=LLAMA_MODEL_PATH,
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    n_ctx=2048,
    f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
    callback_manager=callback_manager,
    verbose=True,
    temperature=0.001,
    max_tokens=2000
)

sql_toolkit = SQLDatabaseToolkit(db=db, llm=llm)

for sql_tool in sql_toolkit.get_tools():
    tools.append(
        sql_tool
    )

raven_prompt = RavenPromptTemplate(
    template=RAVEN_PROMPT, tools=tools, input_variables=["input", "agent_scratchpad"]
)

llm_chain = LLMChain(
    llm=llm,
    prompt=raven_prompt,
    callback_manager=callback_manager,
)

agent = RavenAgent(
    llm_chain=llm_chain,
    allowed_tools=[tool.name for tool in tools],
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    callback_manager=callback_manager,
    handle_parsing_errors=True
)

#agent_executor.run("The data is photometry associated with a supernova in apparent magnitudes, and date in mjd. How many days ago was the brightest observation in the dataset, and in what filter?")
testList = [(59406.217, 'g-ZTF', 19.712, 0.144, 'P48'), (59406.255, 'r-ZTF', 18.495, 0.054, 'P48')]

agent_executor.run("Please get all photometry within the last 3 days from the database, phase each datapoint relative to today, and save all augmented photometry to a file.")

