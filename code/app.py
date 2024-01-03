from flask import Flask, render_template, request, jsonify, Response, stream_with_context, stream_template
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
from langchain.callbacks.streaming_stdout_final_only import FinalStreamingStdOutCallbackHandler
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import traceback
from langchain.embeddings import OpenAIEmbeddings

app = Flask(__name__)
embedder = OpenAIEmbeddings()

db = SQLDatabase.from_uri("mysql+mysqlconnector://root:password@localhost/YSE?port=53306", sample_rows_in_table_info=0)
vector_db = FAISS.load_local("/home/gaglian2/theSpeakYSE/data/YSEPZQueries_index_OAI", embedder)
retriever = vector_db.as_retriever()

tool_description = """
This tool will help you understand similar examples to adapt them to the user question.
Input to this tool should be the question to ask the SQL database.
"""

retriever_tool = create_retriever_tool(
    retriever, name="sql_get_similar_examples", description=tool_description
)

toolkit = SQLDatabaseToolkit(db=db, llm=ChatOpenAI(model_name="gpt-4", temperature=0.2, max_tokens=2500))

custom_tool_list = [retriever_tool]

custom_suffix = Path('custom_suffix.txt').read_text().replace('\n', '')

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
chat_history = []

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

class ChainStreamHandler(FinalStreamingStdOutCallbackHandler):
    def __init__(self, gen):
        super().__init__()
        self.gen = gen
        self.startDone = False
        self.gapSent = False

    def on_llm_new_token(self, token: str, **kwargs):
        # Remember the last n tokens, where n = len(answer_prefix_tokens)
        #self.gen.send(token)
        #self.append_to_last_tokens(token)
    
        if not self.startDone:
           self.gen.send(token)
           self.startDone = '.' in token

        if self.startDone and not self.gapSent:
           self.gen.send("\n\n")
           self.gapSent = True
        
        # Check if the last n tokens match the answer_prefix_tokens list ...
        #if self.check_if_answer_reached():
        #    if self.stream_prefix:
        #        for t in self.last_tokens:
        #            self.gen.send(t)
        #    self.answer_reached = True
        #    return

        # ... if yes, then print tokens from now on
        #if self.answer_reached:
        #    self.gen.send(token)

def llm_thread(g, userText, chat_history):
    complete_response = ''
    try:
        prompt = CUSTOM_PROMPT.format(input=userText,chat_history=str(chat_history))

        agent = create_sql_agent(
        llm=ChatOpenAI(model_name="gpt-4", temperature=0.0, streaming=True, callbacks=[ChainStreamHandler(g)], max_tokens=500),
        toolkit=toolkit,
        verbose=False,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        top_k=1000,
        extra_tools=custom_tool_list,
        suffix=custom_suffix,
        return_intermediate_steps=False,
        handle_parsing_errors=True
        )

        # Assuming ChainStreamHandler sends tokens or chunks of response to the generator
        for response_chunk in agent.run(input=prompt):
            complete_response += response_chunk  # Collect each chunk
            g.send(response_chunk)  # Send chunk for streaming
    except Exception as e:
        g.send(f"Error: {str(e)}")

    finally:
        #only append when it's done!
        chat_history.append("ai: " + complete_response + "\n")
        g.close()

def chain(userText, chat_history):
    g = ThreadedGenerator()
    threading.Thread(target=llm_thread, args=(g, userText, chat_history)).start()
    return g

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/reset_chat', methods=['POST'])
def reset_chat():
    chat_history = []
    return jsonify(success=True)
 
@app.route('/stream', methods=['POST'])
def stream():
    userText = request.json.get("message")

    # Error handling for missing userText
    if not userText:
        return "No text provided", 400

    if len(userText) > 150:
        return "Please ask a shorter question!"

    if request.method == 'POST':    
        resp = Response(stream_with_context(chain(userText, chat_history)), mimetype='text/event-stream')
        resp.headers['X-Accel-Buffering'] = 'no'
        return resp

if __name__ == '__main__':
    app.run(debug=True)
