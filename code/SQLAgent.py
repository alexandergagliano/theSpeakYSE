from langchain.llms import OpenAI, Ollama, HuggingFaceHub, Replicate
from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import tool, Tool
from astropy.time import Time
from datetime import date
import os
from langchain.tools.render import format_tool_to_openai_function
from pydantic import BaseModel, Field
from astropy.io import ascii
import numpy as np
import ast
from langchain_experimental.tools import PythonREPLTool
import csv
from langchain.embeddings import LlamaCppEmbeddings
from langchain.llms import LlamaCpp
import pandas as pd
from langchain.prompts.prompt import PromptTemplate
from langchain.agents import AgentType, create_sql_agent
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.memory import ConversationBufferMemory

#export LANGCHAIN_TRACING_V2="true"
#export LANGCHAIN_API_KEY="<your-api-key>"

db = SQLDatabase.from_uri("sqlite:////Users/alexgagliano/Documents/Research/LLMs/YSEPZ_sqlFiles/miniYSEPZ_noNull.db")

few_shots = {
"Give me spectroscopically-classified supernovae within z<0.015.":"SELECT t.*, h.* FROM YSE_App_transient t INNER JOIN YSE_App_host h ON h.id = t.host_id WHERE t.host_id IS NOT NULL AND (t.redshift OR h.redshift) IS NOT NULL AND COALESCE(t.redshift, h.redshift) < 0.015 AND (t.TNS_spec_class like '%SN%');",

"Find me all fast and young supernovae.":"SELECT DISTINCT t.name, t.TNS_spec_class AS `classification`, g.first_detection AS `first_detection`, g.latest_detection AS `latest_detection`, g.number_of_detection AS `number_of_detection`, og.name AS `group_name` FROM (SELECT DISTINCT	t.id, MIN(pd.obs_date) AS `first_detection`, MAX(pd.obs_date) AS `latest_detection`, COUNT(pd.obs_date) AS `number_of_detection` FROM YSE_App_transient t INNER JOIN YSE_App_transientphotometry tp ON tp.transient_id = t.id INNER JOIN YSE_App_transientphotdata pd ON pd.photometry_id = tp.id WHERE (pd.flux/pd.flux_err > 2 OR pd.mag_err< 0.2 OR ((pd.mag_err IS NULL) AND (pd.mag IS NOT NULL))) GROUP BY t.id) g INNER JOIN YSE_App_transient t ON t.id=g.id INNER JOIN YSE_App_observationgroup og ON og.id = t.obs_group_id WHERE t.TNS_spec_class IS NULL AND t.name NOT LIKE '%YSEAGN%' AND TO_DAYS(CURDATE())- TO_DAYS(first_detection) < 6 AND TO_DAYS(CURDATE())- TO_DAYS(latest_detection) < 3 -- AND TO_DAYS(latest_detection) - TO_DAYS(first_detection) > 0.01 AND number_of_detection > 1 ORDER BY g.first_detection DESC;",

"Give me the properties of all supernovae spectroscopically classified as a Type SN Ibn.": "SELECT t.* FROM YSE_App_transient t WHERE (t.TNS_spec_class LIKE 'SN Ibn');",

"Give me the names of all supernovae spectroscopically classified as a Type SN Ic.": "SELECT t.name FROM YSE_App_transient t WHERE (t.TNS_spec_class LIKE 'SN Ic');",

"Give me all transients in the database with associated host galaxy information.":"SELECT t.* FROM YSE_App_transient t INNER JOIN YSE_App_host h ON h.id = t.host_id;",

"Give me the names of all galaxies matched to named supernovae in 2023.":"SELECT h.name , t.name FROM YSE_App_host h INNER JOIN YSE_App_transient h ON t.host_id=h.id WHERE t.name LIKE '2023%';",

"Show me all transients with HST observations.":"SELECT t.name, t.ra, t.`dec`, t.disc_date, t.redshift, t.TNS_spec_class, t.has_hst FROM YSE_App_transient t WHERE t.has_hst > 0;",

"How many transients did YSE discover in the year 2021?":"SELECT COUNT(*) FROM DISTINCT t.* FROM YSE_App_transient t INNER JOIN YSE_App_transient_tags tt ON tt.transient_id = t.id INNER JOIN YSE_App_transienttag tg ON tg.id = tt.transienttag_id INNER JOIN YSE_App_transientstatus ts ON ts.id = t.status_id WHERE (tg.name = 'YSE' OR tg.name = 'YSE Forced Phot') AND (t.name LIKE '2021%');",

"Give me the properties of all transients with one of the following statuses:`Interesting/Watch/Following/FollowupRequested/New`":"SELECT t.* FROM YSE_App_transient t INNER JOIN YSE_App_transientstatus ts ON ts.id = t.status_id WHERE (ts.name = 'New' OR ts.name = 'Interesting' OR ts.name = 'Watch' OR ts.name = 'Following' OR ts.name = 'FollowupRequested');",

"What supernovae were given a status of `Interesting` in 2021?":"SELECT t.name FROM YSE_App_transient t INNER JOIN YSE_App_transientstatus ts ON ts.id = t.status_id WHERE (ts.name = 'Interesting') AND (t.name LIKE '2021%');",

"What transients currently have follow-up requested?":"SELECT t.name FROM YSE_App_transient t INNER JOIN YSE_App_transientstatus ts ON ts.id = t.status_id WHERE (ts.name = 'FollowupRequested');",

"What is the assigned status of SN 2023mjo?":"SELECT ts.name FROM YSE_App_transient t INNER JOIN YSE_App_transientstatus ts ON ts.id = t.status_id WHERE (t.name LIKE '2023mjo');",

"What transients currently set to 'Watch' status?":"SELECT t.name FROM YSE_App_transient t INNER JOIN YSE_App_transientstatus ts ON ts.id = t.status_id WHERE (ts.name = 'Watch');",

"Return the properties of all stripped envelope supernovae (SESNe) discovered in the last 2 years.":"SELECT t.* FROM YSE_App_transient t AND DATEDIFF(curdate(),t.disc_date) < 730 AND t.TNS_spec_class IN ('SN Ic', 'SN Ibc', 'SN Ib', 'SN IIb');",

"Get all photometry for the transient SN 2019ehk.":"SELECT DISTINCT pd.obs_date AS `observed_date`, TO_DAYS(pd.obs_date) AS `count_date`, pb.name AS `filter`, pd.mag, pd.mag_err,t.mw_ebv, pd.forced FROM YSE_App_transient t INNER JOIN YSE_App_transientphotometry tp ON tp.transient_id = t.id INNER JOIN YSE_App_transientphotdata pd ON pd.photometry_id = tp.id INNER JOIN YSE_App_photometricband pb ON pb.id = pd.band_id WHERE t.name LIKE '2019ehk';",

"Which supernovae have photometry in the database?":"SELECT DISTINCT t.name FROM YSE_App_transient t INNER JOIN YSE_App_transientphotometry tp ON tp.transient_id = t.id INNER JOIN YSE_App_transientphotdata pd ON pd.photometry_id = tp.id;",

"Get the properties of all transients discovered in the last 40 days.":"SELECT DISTINCT	t.name, t.ra,t.dec, pd.obs_date AS `observed_date`, TO_DAYS(pd.obs_date) AS `count_date`, pb.name AS `filter`, pd.mag, pd.mag_err,t.mw_ebv, og.name AS `group_name`, pd.forced FROM YSE_App_transient t INNER JOIN YSE_App_observationgroup og ON og.id = t.obs_group_id INNER JOIN YSE_App_transientphotometry tp ON tp.transient_id = t.id INNER JOIN YSE_App_transientphotdata pd ON pd.photometry_id = tp.id INNER JOIN YSE_App_photometricband pb ON pb.id = pd.band_id WHERE t.disc_date IS NOT NULL AND TO_DAYS(CURDATE())- TO_DAYS(t.disc_date) < 40 ORDER BY t.name ASC, TO_DAYS(pd.obs_date) DESC;",

"Give me the general properties of SN2022xxf.":"SELECT DISTINCT t.* FROM YSE_App_transient t INNER JOIN YSE_App_transient_tags tt ON tt.transient_id = t.id INNER JOIN YSE_App_transienttag tg ON tg.id = tt.transienttag_id INNER JOIN YSE_App_transientstatus ts ON ts.id = t.status_id WHERE (tg.name = 'YSE' OR tg.name = 'YSE Forced Phot') AND (t.name LIKE '2022xxf');",

"At what redshift did SN2020oi occur?":"SELECT DISTINCT t.redshift FROM YSE_App_transient t INNER JOIN YSE_App_transient_tags tt ON tt.transient_id = t.id INNER JOIN YSE_App_transienttag tg ON tg.id = tt.transienttag_id INNER JOIN YSE_App_transientstatus ts ON ts.id = t.status_id WHERE (tg.name = 'YSE' OR tg.name = 'YSE Forced Phot') AND (t.name LIKE '2020oi');",

"What are the coordinates (in degrees) of SN 1987A?":"SELECT DISTINCT t.ra, t.`dec` FROM YSE_App_transient t WHERE t.name LIKE '1987A';",

"What transients peaked brighter than 17th magnitude? Order the results by discovery date.":"SELECT DISTINCT t.name, t.ra AS transient_RA, t.`dec` AS transient_Dec, t.disc_date AS disc_date, pd.mag, t.TNS_spec_class AS spec_class, t.redshift AS transient_z FROM YSE_App_transient t, YSE_App_transientphotdata pd, YSE_App_transientphotometry p WHERE  pd.photometry_id = p.id AND pd.id = (SELECT pd2.id FROM YSE_App_transientphotdata pd2 JOIN YSE_App_transientphotometry p2 ON pd2.photometry_id = p2.id LEFT JOIN YSE_App_transientphotdata_data_quality pdq ON pdq.transientphotdata_id = pd2.id WHERE p2.transient_id = t.id AND pdq.dataquality_id IS NULL AND ISNULL(pd2.mag) = False ORDER BY pd2.mag ASC LIMIT 1)  AND pd.mag < 17 ORDER BY t.disc_date DESC",

"Get me the names and properties of all spectroscopically-confirmed tidal disruption events (TDEs)":"SELECT t.name, t.ra AS transient_RA, t.`dec` AS transient_Dec, t.disc_date AS disc_date, t.redshift AS ‘redshift’,  t.TNS_spec_class FROM YSE_App_transient t WHERE t.TNS_spec_class ='TDE';",

"What is the name and coordinates of the galaxy that hosted supernova 2023bee?":"SELECT h.name, h.ra, h.`dec` FROM YSE_App_host h INNER JOIN YSE_App_transient t ON t.host_id = h.id WHERE t.name LIKE '2023bee';"}

n_gpu_layers = 1
n_batch = 512
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

LLAMA_MODEL_PATH = "/Users/alexgagliano/Documents/Research/LLMs/Models/llama-2-13b.Q5_K_M.gguf"
LLAMA_CHAT_MODEL_PATH = "/Users/alexgagliano/Documents/Research/LLMs/Models/llama-2-13b-chat.Q5_K_M.gguf"

llm = LlamaCpp(
    model_path=LLAMA_MODEL_PATH,
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    n_ctx=2048,
    f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
    callback_manager=callback_manager,
    verbose=True,
)

llm_chat = LlamaCpp(
    model_path=LLAMA_CHAT_MODEL_PATH,
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    n_ctx=2048,
    f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
    callback_manager=callback_manager,
    verbose=True,
)

embedder = LlamaCppEmbeddings(model_path=LLAMA_MODEL_PATH)

few_shot_docs = [
    Document(page_content=question, metadata={"sql_query": few_shots[question]})
    for question in few_shots.keys()
]
vector_db = FAISS.from_documents(few_shot_docs, embedder)
retriever = vector_db.as_retriever()

from langchain.agents.agent_toolkits import create_retriever_tool

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
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
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
custom_tool_list = [retriever_tool, make_plot]

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
    return_intermediate_steps=True,
)

TEMPLATE = """Given an input question, create a syntactically correct SQL query to run, then look at the results of the query.
You are bad at writing code; if the user needs you to write code for calculating or plotting, you must look for functions in your toolkit to do so.
If the user asks for a `light curve`, you will need to first get its photometry and then create a scatter plot with MJD date as x_values and mag as y_values.
Input data into all python functions as arrays. Execute all functions without asking permission first.
Format your output as below, as succinctly as possible:

"Question": "Question here"
"SQLQuery": "SQL Query to run"
"SQLResult": "Result of the SQLQuery"
"Code": "Any code needed"
"Answer": "The final answer to the input query".

Question: {input}"""

CUSTOM_PROMPT = PromptTemplate(
    input_variables=["input"],
    template=TEMPLATE,
)

#prompt = CUSTOM_PROMPT.format(input='Give me the names of all supernovae spectroscopically classified as a Type SN Ia.')
#prompt = CUSTOM_PROMPT.format(input='Give me the names of all supernovae with host galaxy information.')
#prompt = CUSTOM_PROMPT.format(input='Give me the names of all transients tagged as `Interesting`.')
#prompt = CUSTOM_PROMPT.format(input='Give me the names of all transients with a status of `Watch`.')
#prompt = CUSTOM_PROMPT.format(input="Return the photometry for supernova 13EYSEbpq.")
#prompt = CUSTOM_PROMPT.format(input='Give me the redshifts of all transients in the database.')
#prompt = CUSTOM_PROMPT.format(input='Create a histogram of all redshifts of all transients in the database where redshift > 0.')
#prompt = CUSTOM_PROMPT.format(input='Plot a light curve for SN 2023yjp.')
prompt = CUSTOM_PROMPT.format(input='What supernovae have photometry in the database?')

result = agent.run(input=prompt)
result
