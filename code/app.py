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
from openai import OpenAI
import sys
import threading
import queue
from pathlib import Path
from langchain.callbacks.streaming_stdout_final_only import FinalStreamingStdOutCallbackHandler
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import traceback
from langchain.embeddings import OpenAIEmbeddings

#!pip install outlines

app = Flask(__name__)
embedder = OpenAIEmbeddings()
client = OpenAI()

db = SQLDatabase.from_uri("mysql+mysqlconnector://root:password@localhost/YSE?port=53306", sample_rows_in_table_info=0)
vector_db = FAISS.load_local("/home/gaglian2/theSpeakYSE/data/YSEPZQueries_index_OAI", embedder)
query = "Do we have TESS data for SN 2020oi?"
docs_and_scores = vector_db.similarity_search_with_score(query)

tool_description = """
This tool will help you understand similar examples to adapt them to the user question.
Input to this tool should be the question to ask the SQL database.
"""

custom_suffix = Path('custom_suffix_mod.txt').read_text().replace('\n', '')

TEMPLATE = """You are a friendly SQL Agent for the database YSE-PZ.
Given an input question, create a syntactically correct SQL query for a database YSE-PZ. 

Here are a few examples: 
{one_shot_one}
{one_shot_two}
"""

CUSTOM_PROMPT = PromptTemplate(
    input_variables=["one_shot_one", "one_shot_two"],
    template=TEMPLATE,
)

# Initialize an empty list to store chat history
chat_history = []


system_prompt = CUSTOM_PROMPT.format(one_shot_one=docs_and_scores[0][0], one_shot_two=docs_and_scores[1][0])

# start by just trying for a sql agent
#sql_agent = ChatOpenAI(model_name="gpt-4", temperature=0.0, streaming=True, max_tokens=500)
response = client.chat.completions.create(
  max_tokens=300, temperature=0.0,
  n=1,
  model="gpt-3.5-turbo-1106",
  messages=[{"role": "system", "content": system_prompt},{"role": "user", "content": query}]
 )

model = outlines.models.transformers("mistralai/Mistral-7B-v0.1", device="cuda")
generator = outlines.generate.cfg(model, arithmetic_grammar)
sequence = generator("Write a formula that returns 5 using only additions and subtractions.")

generated_response = response.choices[0].message.content
print(generated_response)
sys.exit()

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
        if not self.startDone:
           self.gen.send(token)
           self.startDone = '.' in token

        if self.startDone and not self.gapSent:
           self.gen.send("\n\n")
           self.gapSent = True
        
def llm_thread(g, userText, chat_history):
    complete_response = ''
    try:
        prompt = CUSTOM_PROMPT.format(input=userText,one_shot_one=docs_and_scores[0][0], one_shot_two=docs_and_scores[1][0])

        # start by just trying for a sql agent
        #sql_agent = ChatOpenAI(model_name="gpt-4", temperature=0.0, streaming=True, max_tokens=500)
        sql_agent = openai.ChatCompletion.create(**kwargs)
        
        #  
        #interpreting_agent = ChatOpenAI(model_name="gpt-4", temperature=0.0, streaming=True, callbacks=[ChainStreamHandler(g)], max_tokens=500)

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

#if __name__ == '__main__':
#    app.run(debug=True)
