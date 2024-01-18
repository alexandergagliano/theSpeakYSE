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

toolkit = SQLDatabaseToolkit(db=db, llm=ChatOpenAI(model_name="gpt-4", temperature=0.2, max_tokens=1000))

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

# Initialize an empty list to store chat history
chat_histories_lock = threading.Lock()

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

def get_user_id():
    if 'user_id' not in session:
        session['user_id'] = os.urandom(16).hex()  # Generates a random user_id
    user_id = session['user_id']
    return user_id

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

def format_history(chat_history):
   str_history = """"""
   for msg in chat_history:
       str_history += msg['sender'] + ': ' + msg['message'] + '\n'
   print(str_history)
   return str_history

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
            #if self.stream_prefix:
            #    for t in self.last_tokens:
            #        self.gen.send(t)
            #return

        # ... if yes, then print tokens from now on
        if self.answer_reached:
             print("do nothing")
        #    self.gen.send(token)
        else: 
             self.gen.send(token)
        return

def llm_thread(g, userText, chat_history, user_id):
    complete_response = ''
    try:
        prompt = CUSTOM_PROMPT.format(input=userText,chat_history=str(chat_history))

        agent = create_sql_agent(
        llm=ChatOpenAI(model_name="gpt-4", temperature=0.0, streaming=True,callbacks=[ChainStreamHandler(g)], max_tokens=1000),
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
        #only append when it's done!
        store_chat_history(user_id, {"sender": "User", "message": userText})
        store_chat_history(user_id, {"sender": "AI", "message": complete_response})
        g.close()

def chain(userText, chat_history):
    g = ThreadedGenerator()
    user_id = get_user_id()
    print("User ID: %s"%user_id)
    threading.Thread(target=llm_thread, args=(g, userText, chat_history, user_id)).start()

    return g

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/reset_chat', methods=['POST'])
def reset_chat():
    session.pop('user_id', None)
    return jsonify(success=True)
 
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

    #add userText to the history 
    #convert to a string object for use in the prompt
    history_prompt = format_history(chat_history)

    if request.method == 'POST':    
        resp = Response(stream_with_context(chain(userText, chat_history)), mimetype='text/event-stream')
        resp.headers['X-Accel-Buffering'] = 'no'
        return resp

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0', port=5000)
