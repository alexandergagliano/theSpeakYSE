from langchain.llms import OpenAI,LlamaCpp
from astropy.time import Time
from datetime import date
import os
from langchain.tools.render import format_tool_to_openai_function
from pydantic import BaseModel, Field
from astropy.io import ascii
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_experimental.tools import PythonREPLTool
from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.agents.agent_toolkits import SQLDatabaseToolkit,create_retriever_tool
from langchain.agents import tool, Tool, AgentType, create_sql_agent
from langchain.embeddings import LlamaCppEmbeddings
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.memory import ConversationBufferMemory

n_gpu_layers = 1
n_batch = 512
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

LLAMA_MODEL_PATH = "/Users/alexgagliano/Documents/Research/LLMs/Models/llama-2-13b.Q5_K_M.gguf"

llm = LlamaCpp(
    model_path=LLAMA_MODEL_PATH,
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    n_ctx=2048,
    f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
    callback_manager=callback_manager,
    verbose=True,
)

LLAMA_MODEL_CHAT_PATH = "/Users/alexgagliano/Documents/Research/LLMs/Models/llama-2-13b-chat.Q5_K_M.gguf"

llm_chat = LlamaCpp(
    model_path=LLAMA_MODEL_CHAT_PATH,
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    n_ctx=2048,
    f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
    callback_manager=callback_manager,
    verbose=True,
)

db = SQLDatabase.from_uri("mysql+mysqlconnector://root:password@localhost/YSE?port=53306")
embedder = LlamaCppEmbeddings(model_path=LLAMA_MODEL_PATH)

vector_db = FAISS.load_local("/Users/alexgagliano/Documents/Research/LLMs/theSpeakYSE/data/YSEPZQueries_Faiss_index", embedder)
retriever = vector_db.as_retriever(search_type='mmr')

tool_description = """
This tool will help you understand similar examples to adapt them to the user question.
Input to this tool should be the user question.
"""

retriever_tool = create_retriever_tool(
    retriever, name="sql_get_similar_examples", description=tool_description
)

@tool
def make_plot(plot_type: str, x_name: str, x_values: list, y_name: str, y_values: list) -> None:
    """
    A function for plotting the results of a SQL query.
    Input: plot_type (string): Can be one of 'histogram' or 'scatter plot'.
    Input: x_name (string): the name of the variable associated with x_values.
    Input: x_values (array): the x-values for the plot. If plot_type == 'histogram',
            x_values contain the data for the plot.
    Input: y_name (string): the name of the variable associated with y_values.
    Input: y_values (array): the y-values for the plot.
    Input:
    """
    sns.set_context("poster")
    if plot_type == 'histogram':
        plt.hist(x_values)
        plt.xlabel(x_name)
    elif plot_type == 'scatter':
        plt.plot(x_values, y_values, 'o-')
        plt.xlabel(x_name)
        plt.ylabel(y_name)
    plt.show()

toolkit = SQLDatabaseToolkit(db=db, llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, max_tokens=1000))
custom_tool_list = [retriever_tool]

custom_suffix = """
 Consider the following context to construct the SQL query:
 YSE_App_transient table contains info about a transient/supernova, including its name, coordinates (ra,dec), discovery date (disc_date), spectroscopic redshift (redshift), spectroscopic class (TNS_spec_class) if available, its associated host galaxy as host_id, etc.
 YSE_App_host table contains info about a transient's host galaxy, matched to a transient on YSE_App_transient.host_id = YSE_App_host.id.
 YSE_App_transientphotometry table solely links the general properties of a transient to its raw photometric data (in YSE_App_transientphotdata), and contains the field transient_id (associated with YSE_App_transient.id).
 YSE_App_transientphotdata table contains info about photometry of each transient, including its observed date (obs_date), filter (band_id) brightness and brightness error (in magnitudes (mag and mag_err) and flux (flux and flux_err)). This table is matched
 by first linking YSE_App_transient to YSE_App_transientphotometry via YSE_App_transient.id=YSE_App_transientphotometry.transient_id, and then via YSE_App_transientphotdata.photometry_id = YSE_App_transientphotometry.id.
 When answering questions about brightness or observations, you must always JOIN YSE_App_transient to YSE_App_transientphotometry to YSE_App_transientphotdata tables.
 YSE_App_photometricband table contains the name of each observing filter, matched to YSE_App_transientphotdata via YSE_App_photometricband.id = YSE_App_transientphotdata.band_id.
 YSE_App_transient_tags table contains info about associated tags for that transient, matched to the transient on YSE_App_transient_tags.transient_id = YSE_App_transient.id.
 YSE_App_transienttag table contains the labels for each of the tags in YSE_App_transient_tags ("Young", "Dave", etc), matched on YSE_App_transienttag.id = YSE_App_transient_tags.transienttag_id.
 YSE_App_transientstatus table contains the primary status of the transient (if we think it is Interesting, Watch for might be interesting, Following for actively observing, FollowupRequested for requested additional data,
 FollowupFinished when we have stopped studying the event, and Ignore if not interesting). This table is matched to the transient through YSE_App_transientstatus.id = YSE_App_transient.status_id.
 YSE_App_transientspectrum table links the general properties of a transient to its raw spectroscopic data (in YSE_App_transientspecdata).
 When answering questions about spectra, you must always JOIN YSE_App_transient to YSE_App_transientspectrum on YSE_App_transientspectrum.transient_id = transient.id, then to YSE_App_transientspecdata on YSE_App_transientspecdata.spectrum_id = YSE_App_transientspectrum.id.

You MUST first get similar examples to your SQL query using your retriever tool.
If the examples are enough to construct the query, you can build it.
If there are no similar exmaples, you must look at the tables in the database to see what you can query (without printing out the extensive schema).
Then you should query the schema of the most relevant tables.
If there is chat history, you must also base your answers on the previous SQL results.
"""

agent = create_sql_agent(
    llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, max_tokens=1000),
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    #agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    extra_tools=custom_tool_list,
    suffix=custom_suffix,
    max_iterations=3,
    return_intermediate_steps=False,
)

TEMPLATE = """You are a SQL Agent for the database YSE-PZ.
Given an input question, create a syntactically correct SQL query, then execute the query and return the result.
Execute all SQL queries without permission - do not return generic information.
You must format your output as below:

"Input": "Input question"
"SQLQuery": "SQL Query to run"
"SQLResult": "Result of the Query"
"Answer": "The final answer".

Use your current conversation below for additional context.
Drop all "SN" and "supernova" prefixes from each transient's name before constructing the query; as an example,
'SN 2020oi' should be queried as '2020oi'. Do not provide any false information; if the SQL query returns no results
or an error, you must return the SQL Query and state honestly that you do not know.
Assume the variable 'z' corresponds to redshift - you must select spectroscopic redshift from YSE_App_transient table if available, or the YSE_App_host photometric redshift (photo_z) value if not.
Only return the first 10 results of any SQL Query (LIMIT 10).
For the number of days relative to today, pay close attention to the commands DATEDIFF(), CURDATE(), and TO_DAYS().

Current conversation:
{chat_history}

Question: {input}
"""

CUSTOM_PROMPT = PromptTemplate(
    input_variables=["input", "chat_history"],
    template=TEMPLATE,
)

def asktheSpeakYSE(question, history=''):
    prompt = CUSTOM_PROMPT.format(input=question,chat_history=history)
    result = agent.run(input=prompt)
    history += str('Human: ' + question + '\n' + 'AI Assistant: ' + result)
    return result, history

#prompt, history = asktheSpeakYSE(How many transients are currently tagged as `Young`?')
#prompt, history = asktheSpeakYSE('Give me the names of all transients discovered in 2023 with a `TESS` tag.')
#prompt, history = asktheSpeakYSE('Give me the names and discovery dates of all spectroscopically-confirmed supernovae discovered between March 1st, 2021 and March 2nd, 2022.')
#prompt, history = asktheSpeakYSE('Of all the transients with a `TESS` tag, which is currently the brightest?')
#prompt, history = asktheSpeakYSE('Give me the names of all transients with a status of `Watch`.')
#prompt, history = asktheSpeakYSE('Give me the names of all supernovae with host galaxy information.')
#prompt, history = asktheSpeakYSE("Of the transients current tagged as 'Interesting', which was discovered most recently?")
#prompt, history = asktheSpeakYSE('How many supernovae are in the database?')
#prompt, history = asktheSpeakYSE('Give me the names of all supernovae spectroscopically classified as an SN Ia, ordered by descending discovery date.')
#prompt, history = asktheSpeakYSE('What is name of the the galaxy that hosted SN 2020oi?')
#prompt, history = asktheSpeakYSE("Return the photometry for supernova 13EYSEbpq.")
#prompt, history = asktheSpeakYSE('What transient has the highest redshift in the database?')
#prompt, history = asktheSpeakYSE('Do we have any host galaxy information for SN 2023ixf?')
#prompt, history = asktheSpeakYSE('In what galaxy did SN2022xxf happen?')
#prompt, history = asktheSpeakYSE('Create a histogram of the redshifts of 10 transients in the database, where redshifts are available.')
#prompt, history = asktheSpeakYSE("What's the phone number of the PI of the KAST program?")
#prompt, history = asktheSpeakYSE("Give me the data for the last spectrum taken for 2023ixf.")

history = ''
response, history = asktheSpeakYSE("What transient occurred at the highest redshift?", history)
response, history = asktheSpeakYSE("What was the spectroscopic class of that transient?", history)
response, history = asktheSpeakYSE("Wow - and that classification was provided by TNS?", history)

history = ''
prompt, history = asktheSpeakYSE("Who's the PI of the currently-running DECam program?", history)
prompt, history = asktheSpeakYSE("What's his phone number?", history)

history = ''
result, history = asktheSpeakYSE("What is the name of the galaxy that hosted SN 2020oi?", history)
result, history = asktheSpeakYSE("What can you tell me about that galaxy, NGC 4321?", history)
result, history = asktheSpeakYSE("What are its coordinates?", history)

history = ''
result, history = asktheSpeakYSE("How many spectra exist for SN 2023ixf?", history)
result, history = asktheSpeakYSE('When was the last one taken and with which instrument?', history)
result, history = asktheSpeakYSE('How many days after discovery was that?', history)

#Some examples with a history:
history = ''
result, history = asktheSpeakYSE("What spectroscopically-confirmed SN Ic took place at the lowest redshift? Ignore null values.", history)
result, history = asktheSpeakYSE('What was its redshift?', history)
result, history = asktheSpeakYSE('And how many spectra were obtained for that supernova?', history)
result, history = asktheSpeakYSE('When was the latest spectrum obtained for that supernovas?', history)

history = ''
result, history = asktheSpeakYSE("How many supernovae are there beyond a redshift of 1?", history)
result, history = asktheSpeakYSE("What are the basic properties of these supernovae?", history)


#needs an example in the training set
#prompt = askTheSpeakYSE('What transients had new photometry taken at any time in the last 2 days?')
#prompt = askTheSpeakYSE('Give me the names of all transients within z < 0.5 where the spectroscopic and photometric classifications disagree.')
