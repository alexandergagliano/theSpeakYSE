from flask import Flask, render_template, request, jsonify, Response, stream_with_context, stream_template, session
import subprocess
import traceback
from langchain.llms import OpenAI
from datetime import date
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import seaborn as sns
from langchain.utilities import SQLDatabase
from langchain.agents.agent_toolkits import SQLDatabaseToolkit,create_retriever_tool
from langchain.agents import tool, Tool, AgentType, create_sql_agent
from langchain_community.llms import HuggingFaceHub
from langchain.prompts.prompt import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document
from langchain.vectorstores import FAISS
import time
import sys
import threading
import queue
from pathlib import Path
import json
from langchain.callbacks.streaming_stdout_final_only import FinalStreamingStdOutCallbackHandler
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import traceback
from langchain.embeddings import OpenAIEmbeddings
import redis

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ['FLASK_TOKEN']
embedder = OpenAIEmbeddings()

redis_client = redis.Redis(host='localhost', port=6379, db=0)

db = SQLDatabase.from_uri("mysql+mysqlconnector://root:password@localhost/YSE?port=53306",sample_rows_in_table_info=0)
vector_db = FAISS.load_local("/home/gaglian2/theSpeakYSE/data/YSEPZQueries_index_OAI", embedder)
retriever = vector_db.as_retriever()

tool_description = """
This tool will help you understand similar examples to adapt them to the user question.
Input to this tool should be the question to ask the SQL database.
"""

retriever_tool = create_retriever_tool(
    retriever, name="sql_get_similar_examples", description=tool_description
)

repo_id = 'machinists/Mistral-7B-Instruct-SQL'
#repo_id_functioncalling = 'gorilla-llm/gorilla-7b-hf-delta-v1'
#repo_id_functioncalling = 'meetkai/functionary-small-v2.2'
mistral_sql_llm = HuggingFaceHub(
    repo_id=repo_id, model_kwargs={"temperature": 0.2, "max_length": 1024}
)

# Make sure the model path is correct for your system!
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

llm = LlamaCpp(
    model_path="/home/gaglian2/Models/functionary-7b-v2.1.q4_0.gguf",
    temperature=0.2,
    max_tokens=2000,
    top_p=1,
    callback_manager=callback_manager,
    verbose=True,  # Verbose is required to pass to the callback manager
)


toolkit = SQLDatabaseToolkit(db=db, llm=mistral_sql_llm)

custom_tool_list = [retriever_tool]

custom_suffix = Path('custom_suffix_mod.txt').read_text().replace('\n', '')

TEMPLATE = """You are a friendly SQL Agent for the database YSE-PZ.
Given an input question, create a syntactically correct SQL query, then execute the query and return the result.

Use the chat history below for additional context.
If the SQL query returns 0 results or an error, you must state honestly that there are no results. 
Assume the variable 'z' corresponds to redshift - you must select spectroscopic redshift from YSE_App_transient table if available, or the YSE_App_host photometric redshift (photo_z) value if not. Ignore all negative redshift values. Assume a supernova is only spectroscopically confirmed if t.TNS_spec_class starts with "SN". 
If sorting by a property, you must only consider entries WHERE property IS NOT NULL. 
For the number of days relative to today, use the commands DATEDIFF(), CURDATE(), and TO_DAYS().

Chat history:
{chat_history}

Question: {input}
Answer: 
"""

CUSTOM_PROMPT = PromptTemplate(
    input_variables=["input", "chat_history"],
    template=TEMPLATE,
)


#define stuff here 
userText = 'Get me the names of all type II supernovae.'
chat_history=''

prompt = CUSTOM_PROMPT.format(input=userText,chat_history=str(chat_history))

 
agent = create_sql_agent(
# llm=ChatOpenAI(model_name="gpt-4", temperature=0.0, streaming=True, max_tokens=1000),
 gorilla_llm,
 toolkit=toolkit,
 verbose=False,
 agent_type=AgentType.OPENAI_FUNCTIONS,
 top_k=50,
 extra_tools=custom_tool_list,
 suffix=custom_suffix,
 return_intermediate_steps=False,
 handle_parsing_errors=True)

# Assuming ChainStreamHandler sends tokens or chunks of response to the generator
complete_response = ''
for response_chunk in agent.run(input=prompt,handle_parsing_errors=True):
    complete_response += response_chunk  # Collect each chunk
    print(response_chunk)
