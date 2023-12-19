"""
This file handles loading in arXiV articles, converting them
to VectorStore objects for processing by LangChain VectorStore agents,
and decoding query responses about the VectorStore representations.
"""
"""IN PROGRESS - IGNORE."""

from langchain.document_loaders import ArxivLoader
from langchain import PromptTemplate
from langchain.agents.agent_toolkits import (
    VectorStoreInfo,
    VectorStoreToolkit,
    create_vectorstore_agent,
)
from langchain.embeddings import LlamaCppEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import LlamaCpp

ARXIV_TEMPLATE = """
<s>[INST] <<SYS>>
You are an astronomer interested in supernova research. When summarizing and analyzing publications,
your main objectives should be:
(1) Identify whether the paper studies one supernova in detail, or a collection of supernovae.
(2) Identify the names of the supernovae being studied, if any.
(3) Identify whether the dataset used contains observed or simulated supernovae.
(4) Identify whether the paper uses photometry, spectroscopy, and/or host galaxy information.
(5) Identify the main analysis techniques used, with a special emphasis on machine learning methods.
(6) Identify the main takeaways of the paper, and how results differ from previous works.
<</SYS>>
 
{text} [/INST]
"""

ARXIV_PROMPT = PromptTemplate(
    input_variables=["text"],
    template=ARXIV_TEMPLATE,
)

n_gpu_layers = 1  # Metal set to 1 is enough.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

LLAMA_MODEL_PATH = "/Users/kdesoto/Downloads/llama-2-70b.Q4_K_M.gguf"
# Make sure the model path is correct for your system!
LLM = LlamaCpp(
    model_path=LLAMA_MODEL_PATH,
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    n_ctx=2048,
    f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
    callback_manager=callback_manager,
    verbose=True,
)

EMBEDDER = LlamaCppEmbeddings(model_path=LLAMA_MODEL_PATH)

def arxiv_loader(query, num=10):
    """Loader for arxiv metadata + file content. Returns associated
    vector store."""
    docs = ArxivLoader(query=query, load_max_docs=num).load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(docs)

    #Use Llama model for embedding
    llama_model_path = LLAMA_MODEL_PATH
    
    arxiv_store = Chroma.from_documents(
        texts, EMBEDDER, collection_name="arxiv-query"
    )
    return arxiv_store

def create_agent_from_query(query):
    """Creates agent executor from arxiv query.
    """
    vectorstore = arxiv_loader(query)

    vectorstore_info = VectorStoreInfo(
        name="state_of_union_address",
        description="the most recent state of the Union adress",
        vectorstore=vectorstore,
    )
    toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)
    agent_executor = create_vectorstore_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True
    )
    return agent_executor

agent_executor = create_agent_from_query("Villar")
agent_executor.run("Describe recent advances in machine learning applied to supernovae.")





    
    
    

    
    