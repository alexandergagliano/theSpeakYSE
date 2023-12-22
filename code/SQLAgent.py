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
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.memory import ConversationBufferMemory

#os.environ['OPENAI_API_KEY']='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'

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

toolkit = SQLDatabaseToolkit(db=db, llm=llm)
custom_tool_list = [retriever_tool, PythonREPLTool(), make_plot]

custom_suffix = """
Limit 10 in the query always. Assume the words `transient` and `supernova` are interchangeable.
If any queries return no results, tell me that; do not invent any new information.
I should first get the similar examples I know using my retriver tool.
If the examples are enough to construct the query, I can build it.
Otherwise, I can then look at the tables in the database to see what I can query.
Then I should query the schema of the most relevant tables.
"""

memory = ConversationBufferMemory(memory_key="chat_history")

agent = create_sql_agent(
    llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    extra_tools=custom_tool_list,
    suffix=custom_suffix,
    max_iterations=3,
    memory=memory,
    return_intermediate_steps=False,
)

TEMPLATE = """Given an input question, create a syntactically correct SQL query to run, then execute the query and analyze the data as necessary to answer the question.
If the user asks for a `light curve`, you will need to first get its photometry and then create a scatter plot with MJD date as x_values and mag as y_values.
If the user asks for a classification, always give the spectroscopic classification first if available.

Format your output as below:

"Question": "Question here"
"SQLQuery": "SQL Query to run"
"Answer": "The final answer".

Question: {input}"""

CUSTOM_PROMPT = PromptTemplate(
    input_variables=["input"],
    template=TEMPLATE,
)

#prompt = CUSTOM_PROMPT.format(input='Give me the names of all transients tagged as `Interesting`')
#prompt = CUSTOM_PROMPT.format(input='Give me the names of all transients with a status of `Watch`.')
#prompt = CUSTOM_PROMPT.format(input='Give me the names of all supernovae with host galaxy information.')
#prompt = CUSTOM_PROMPT.format(input='How many transients of any kind are in the database?')
#prompt = CUSTOM_PROMPT.format(input='How many supernovae are in the database?')
#prompt = CUSTOM_PROMPT.format(input='Give me the names of all supernovae spectroscopically classified as a Type SN Ia.')
#prompt = CUSTOM_PROMPT.format(input='What is name of the the galaxy that hosted SN 2020oi?')
#prompt = CUSTOM_PROMPT.format(input="Return the photometry for supernova 13EYSEbpq.")
#prompt = CUSTOM_PROMPT.format(input='What transient has the highest redshift in the database?')

#prompt = CUSTOM_PROMPT.format(input='What transients had new photometry taken in the last week?')
#struggled...

#prompt = CUSTOM_PROMPT.format(input='Create a histogram of all redshifts of all transients in the database where redshift > 0.')

result = agent.run(input=prompt)
