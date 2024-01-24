from langchain.llms import OpenAI
from datetime import date
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from langchain.utilities import SQLDatabase
from langchain.agents.agent_toolkits import SQLDatabaseToolkit,create_retriever_tool
from langchain.agents import tool, Tool, AgentType, create_sql_agent
from langchain.prompts.prompt import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document
from langchain.vectorstores import FAISS
import sys
import traceback
from langchain.embeddings import OpenAIEmbeddings

embedder = OpenAIEmbeddings()

db = SQLDatabase.from_uri("mysql+mysqlconnector://root:password@localhost/YSE?port=53306")
vector_db = FAISS.load_local("/home/gaglian2/theSpeakYSE/data/YSEPZQueries_index_OAI", embedder)
retriever = vector_db.as_retriever()


tool_description = """
This tool will help you understand similar examples to adapt them to the user question.
Input to this tool should be the question to ask the SQL database.
"""

retriever_tool = create_retriever_tool(
    retriever, name="sql_get_similar_examples", description=tool_description
)

toolkit = SQLDatabaseToolkit(db=db, llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2, max_tokens=1000))
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
    llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.0, max_tokens=1000),
    toolkit=toolkit,
    verbose=False,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    extra_tools=custom_tool_list,
    suffix=custom_suffix,
    max_iterations=3,
    return_intermediate_steps=False,
)

TEMPLATE = """You are a conversational AI designed to answer questions on behalf of the Young Supernova Experiment using a database.
Given an input, decide whether additional info is needed. If not, repond directly. If so,
create a syntactically correct SQL query, then execute the query and return the result.
You must format your logic as follows, but only print out the final answer unless there is a problem with the SQL query:

'Input': 'Input question',
'SQLQuery': 'SQL query to run',
'SQLResult': 'Result of the query',
'Answer': 'The final answer'

Use your current conversation below for additional context.
Drop all "SN" and "supernova" prefixes from each transient's name before constructing the query; as an example,
'SN 2020oi' should be queried as '2020oi'. Do not provide any false information; if the SQL query returns no results
or an error, you must return the SQL Query and state honestly that you do not know.
Assume the variable 'z' corresponds to redshift - you must select spectroscopic redshift from YSE_App_transient table if available, or the YSE_App_host photometric redshift (photo_z) value if not. Assume that redshift must be a positive value.
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
    return result

if __name__ == '__main__':
    question = sys.argv[1]
    chat_history = eval(sys.argv[2])
    try:
        response = asktheSpeakYSE(question, chat_history)
        print(response,flush=True)
    except Exception as e:
        # Log or print the error and traceback for debugging
        print(f"Error: {str(e)}")
