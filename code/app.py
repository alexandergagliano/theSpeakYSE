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
from langchain.prompts.prompt import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain_community.llms import HuggingFaceHub
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

"""Toolkit for interacting with an SQL database."""
from typing import List

from langchain_core.language_models import BaseLanguageModel
from langchain_core.pydantic_v1 import Field

from langchain_community.agent_toolkits.base import BaseToolkit
from langchain_community.tools import BaseTool
from langchain_community.tools.sql_database.tool import (
    InfoSQLDatabaseTool,
    ListSQLDatabaseTool,
    QuerySQLCheckerTool,
    QuerySQLDataBaseTool,
)
from langchain_community.utilities.sql_database import SQLDatabase

# toolkit that allows an LLM to construct and validate SQL queries before executing them.
class SQLDatabaseToolkit(BaseToolkit):
    """Toolkit for interacting with SQL databases."""

    db: SQLDatabase = Field(exclude=True)
    llm: BaseLanguageModel = Field(exclude=True)

    @property
    def dialect(self) -> str:
        """Return string representation of SQL dialect to use."""
        return self.db.dialect

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        list_sql_database_tool = ListSQLDatabaseTool(db=self.db)
        info_sql_database_tool_description = (
            "Input to this tool is a comma-separated list of tables, output is the "
            "schema and sample rows for those tables. "
            "Be sure that the tables actually exist by calling "
            f"{list_sql_database_tool.name} first! "
            "Example Input: table1, table2, table3"
        )
        info_sql_database_tool = InfoSQLDatabaseTool(
            db=self.db, description=info_sql_database_tool_description
        )
        query_sql_database_tool_description = (
            "Input to this tool is a detailed and correct SQL query, output is a "
            "result from the database. If the query is not correct, an error message "
            "will be returned. If an error is returned, rewrite the query, check the "
            "query, and try again. If you encounter an issue with Unknown column "
            f"'xxxx' in 'field list', use {info_sql_database_tool.name} "
            "to query the correct table fields."
        )
        query_sql_database_tool = QuerySQLDataBaseTool(
            db=self.db, description=query_sql_database_tool_description
        )
        query_sql_checker_tool_description = (
            "Use this tool to double check if your query is correct before executing "
            "it. Always use this tool before executing a query with "
            f"{query_sql_database_tool.name}!"
        )
        query_sql_checker_tool = QuerySQLCheckerTool(
            db=self.db, llm=self.llm, description=query_sql_checker_tool_description
        )
        return [
            query_sql_database_tool
            #info_sql_database_tool,
            #list_sql_database_tool,
            #query_sql_checker_tool,
        ]

    def get_context(self) -> dict:
        """Return db context that you may want in agent prompt."""
        return self.db.get_context()

app = Flask(__name__)

# create a unique token for this app
app.config['SECRET_KEY'] = os.environ['FLASK_TOKEN']
embedder = OpenAIEmbeddings()

# the database where the anonymized chat histories will be stored
redis_client = redis.Redis(host='localhost', port=6379, db=0)

# the mySQL connector, linked to the dockerized YSE-PZ image
db = SQLDatabase.from_uri("mysql+mysqlconnector://root:password@localhost/YSE?port=53306",sample_rows_in_table_info=0)

# embeddings of previous successful YSE-PZ SQL queries - see storeSQLEmbeddings.py for full dataset
vector_db = FAISS.load_local("/home/gaglian2/theSpeakYSE/data/YSEPZQueries_index_OAI", embedder)
retriever = vector_db.as_retriever()

# basic description of the similar queries retriever tool
tool_description = """
This tool will help you understand similar examples to adapt them to the user question.
"""

retriever_tool = create_retriever_tool(
    retriever, name="sql_get_similar_examples", description=tool_description
)


# switching to a HuggingFace-hosted open-source Mistral model to conduct the queries for now
#toolkit = SQLDatabaseToolkit(db=db, llm=ChatOpenAI(model_name="gpt-4", temperature=0.2))
repo_id = 'machinists/Mistral-7B-Instruct-SQL'
mistral_sql_llm = HuggingFaceHub(
    repo_id=repo_id, model_kwargs={"temperature": 0.2, "max_length": 100}
)

# define the SQL agent, and the embedding retrieval tool
toolkit = SQLDatabaseToolkit(db=db, llm=mistral_sql_llm)
custom_tool_list = [retriever_tool]

# add description of YSE-PZ tables to model prompt
custom_suffix = Path('custom_suffix.txt').read_text().replace('\n', '')

# Prompt for the model, with chat history.
TEMPLATE = """You are an enthusiastic SQL Agent for the database YSE-PZ.
Given an input question, create a syntactically correct SQL query, then execute the query and succinctly answer the question.

Use the chat history below for additional context.
If the SQL query returns 0 results or an error, you must state honestly that there are no results. 
Assume the variable 'z' corresponds to redshift - you must select spectroscopic redshift from YSE_App_transient table if available, or the YSE_App_host photometric redshift (photo_z) value if not. Ignore all negative redshift values. Assume a supernova is only spectroscopically confirmed if t.TNS_spec_class starts with "SN". 
If sorting by a property, you must only consider entries WHERE property IS NOT NULL. 
For the number of days relative to today, use the commands DATEDIFF(), CURDATE(), and TO_DAYS().

Chat history:
{chat_history}

Question: {input}
"""

CUSTOM_PROMPT = PromptTemplate(
    input_variables=["input", "chat_history"],
    template=TEMPLATE,
)

# Initialize an empty list to store chat history
chat_histories_lock = threading.Lock()

# some helper functions to store user chat histories in the redis database.
def get_chat_history(user_id):
    chat_history = redis_client.get(user_id)
    if chat_history is not None:
        return json.loads(chat_history.decode('utf-8'))
    return []

def zero_chat_history(user_id):
    # Serialize and store the updated history back into Redis
    redis_client.set(user_id, json.dumps([]))

def store_chat_history(user_id, new_message):
    existing_history = redis_client.get(user_id)
    if existing_history is not None:
        chat_history = json.loads(existing_history.decode('utf-8'))
    else:
        chat_history = []

    # Append the new message to the chat history
    chat_history.append(new_message)

    # Serialize and store the updated history back into Redis
    redis_client.set(user_id, json.dumps(chat_history))

# associate each user with a random 16-bit string
def get_user_id():
    if 'user_id' not in session:
        session['user_id'] = os.urandom(16).hex()  # Generates a random user_id
    user_id = session['user_id']
    return user_id

# the generator object that allows for tokens to be passed to the html webpage asynchronously
# (in the case of response generation).
class ThreadedGenerator:
    def __init__(self):
        self.queue = queue.Queue()

    def __iter__(self):
        return self

    def __next__(self):
        item = self.queue.get()
        if item is StopIteration: raise item
        return item

    def send(self, data):
        self.queue.put_nowait(data)

    def close(self):
        self.queue.put(StopIteration)

# translating the history from a redis-stored json object to a string 
# that can be used in the prompt
def format_history(chat_history):
   str_history = """"""
   for msg in chat_history:
       str_history += msg['sender'] + ': ' + msg['message'] + '\n'
   print(str_history)
   return str_history

# custom handler to change what text gets written out to the webpage 
# (e.g., just the final answer, all the intermediate steps, etc).
class ChainStreamHandler(FinalStreamingStdOutCallbackHandler):
    def __init__(self, gen):
        super().__init__()
        self.gen = gen

    def on_llm_new_token(self, token: str, **kwargs):
        # Remember the last n tokens, where n = len(answer_prefix_tokens)
        self.append_to_last_tokens(token)
    
        # Check if the last n tokens match the answer_prefix_tokens list ...
        if self.check_if_answer_reached():
            self.answer_reached = True

        # ... if yes, then print tokens from now on
        if not self.answer_reached:
             self.gen.send(token)
        return

# Create the agent, ask a question, and send the response to the webpage.
def llm_thread(g, userText, chat_history, user_id):
    complete_response = ''
    try:
        prompt = CUSTOM_PROMPT.format(input=userText,chat_history=str(chat_history))

        agent = create_sql_agent(
        llm=ChatOpenAI(model_name="gpt-4", temperature=0.0, streaming=True,callbacks=[ChainStreamHandler(g)]),
        toolkit=toolkit,
        verbose=False,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        top_k=50,
        extra_tools=custom_tool_list,
        suffix=custom_suffix,
        return_intermediate_steps=False,
        handle_parsing_errors=True
        )

        # Assuming ChainStreamHandler sends tokens or chunks of response to the generator
        for response_chunk in agent.run(input=prompt,handle_parsing_errors=True):
            complete_response += response_chunk  # Collect each chunk
    except Exception as e:
        g.send(f"Error: {str(e)}")

    finally:
        # Once the full response is generated, add this piece of the conversation to the redis db
        store_chat_history(user_id, {"sender": "User", "message": userText})
        store_chat_history(user_id, {"sender": "AI", "message": complete_response})
        g.close()

# creates the unique user thread for a conversation
def chain(userText, chat_history):
    g = ThreadedGenerator()
    user_id = get_user_id()
    print("User ID: %s"%user_id)
    threading.Thread(target=llm_thread, args=(g, userText, chat_history, user_id)).start()

    return g

# load the website
@app.route('/')
def index():
    return render_template('index.html')

# reset the chat history - just deletes the user-id, a new one is created elsewhere
@app.route('/reset_chat', methods=['POST'])
def reset_chat():
    session.pop('user_id', None)
    return jsonify(success=True)
 
# stream the response to the website, with some light error handling
@app.route('/stream', methods=['POST'])
def stream():
    user_id = get_user_id()
    userText = request.json.get("message")

    # Error handling for missing userText
    if not userText:
        return "No text provided", 400

    if len(userText) > 150:
        return "Please ask a shorter question!"

    # After streaming, update session
    with chat_histories_lock:
        chat_history = get_chat_history(user_id)
    # printing chat history to the terminal to make sure it gets stored and reset correctly
    print(chat_history)

    #add userText to the history 
    #convert to a string object for use in the prompt
    history_prompt = format_history(chat_history)

    if request.method == 'POST':    
        resp = Response(stream_with_context(chain(userText, chat_history)), mimetype='text/event-stream')
        resp.headers['X-Accel-Buffering'] = 'no'
        return resp

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0', port=5000)
