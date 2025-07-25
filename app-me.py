import os
import re
import logging
from typing import List, Dict, Optional, Set
from fastapi import FastAPI, APIRouter, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate

# --- Basic Setup & Configuration ---

# Configure logging
logging.basicConfig(level=logging.INFO,filename='myapp.log' ,format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="SRC Cluster Knowledge Base API",
    description="An API to query documentation about Stanford's high-performance computing clusters.",
    version="1.0.0",
)

origins = ['http://localhost:5000', 'http://127.0.0.1:5000']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models for API Data Validation ---

class QueryRequest(BaseModel):
    """Request model for a user query."""
    query: str
    cluster: Optional[str] = None  # Allow user to explicitly specify the cluster

class Source(BaseModel):
    """Model for a single source document."""
    name: str
    url: Optional[str] = None

class QueryResponse(BaseModel):
    """Response model for a successful query."""
    answer: str
    cluster: str
    sources: List[Source]

# --- Global Configuration & Constants ---

# Define paths to the directories containing the markdown files for different clusters
CLUSTERS = {
    "sherlock": "sherlock/",
    "farmshare": "farmshare/",
     "oak": "oak/", 
     "elm": "elm/"
}


# LOCAL_MODEL_PATH = os.getenv("MODEL_PATH", "/default/path/to/model")
LOCAL_MODEL_PATH = "/oak/stanford/groups/ruthm/bcritt/.cache/huggingface/hub/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/e0bc86c23ce5aae1db576c8cca6f06f1f73af2db"



# --- Helper Functions ---

# Ingest markdown files from a specified directory
def ingest_markdown_files(corpusdir: str) -> List[Document]:
    documents = []
    for infile in os.listdir(corpusdir):
        if infile.endswith(".md"):
            with open(os.path.join(corpusdir, infile), 'r', errors='ignore') as fin:
                content = fin.read()
                documents.append(Document(page_content=content))
    return documents


def identify_cluster(user_query: str) -> str:
    """Identifies the target cluster from the user's query."""
    user_query_lower = user_query.lower()
    for cluster_name in CLUSTERS:
        if cluster_name in user_query_lower:
            return cluster_name
    return "unknown"


# --- Application Startup: Model & Data Loading ---
# This code runs ONCE when the application starts.

logging.info("Starting application setup...")

# Setup SQLite cache for LLM responses
try:
    set_llm_cache(SQLiteCache(database_path=".langchain.db"))
    logging.info("LangChain LLM cache enabled with SQLite.")
except Exception as e:
    logging.error(f"Could not set up LLM cache: {e}")

# Initialize dictionaries to hold models, retrievers, and documents for each cluster
llm = None
retrievers: Dict[str, BM25Retriever] = {}
all_docs: Dict[str, List[Document]] = {}

try:
    logging.info(f"Loading model from {LOCAL_MODEL_PATH}...")
    model = AutoModelForCausalLM.from_pretrained(
        LOCAL_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,  # Increased for potentially more detailed answers
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        return_full_text=False
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    logging.info("Model and pipeline loaded successfully.")

except Exception as e:
    logging.critical(f"FATAL: Could not load the model from path {LOCAL_MODEL_PATH}. API cannot start. Error: {e}")
    # In a real scenario, you might want the app to exit or handle this more gracefully.
    # For this script, we'll let it fail at the query endpoint if llm is None.

# Ingest documents and create a retriever for each cluster
for cluster_name, path in CLUSTERS.items():
    if not os.path.isdir(path):
        logging.warning(f"Directory not found for cluster '{cluster_name}': {path}. Skipping.")
        continue
    
    logging.info(f"Ingesting documents for cluster: {cluster_name}")
    documents = ingest_markdown_files(path)
    if not documents:
        logging.warning(f"No documents found for cluster '{cluster_name}'.")
        continue

    # Using a simple text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = text_splitter.split_documents(documents)
    
    all_docs[cluster_name] = split_docs
    retrievers[cluster_name] = BM25Retriever.from_documents(split_docs)
    logging.info(f"Ingested {len(documents)} documents and created retriever for '{cluster_name}'.")

logging.info("Application setup complete. API is ready.")


# --- API Endpoints ---

@app.get("/", summary="Root endpoint", description="Simple health check endpoint.")
async def root():
    return {"message": "SRCC Knowledge Base API is running"}

@app.post("/query/", response_model=QueryResponse, summary="Query the knowledge base")
async def query_kb(request: QueryRequest):
    """
    Accepts a user query and returns a synthesized answer based on retrieved documents.
    
    The API will attempt to automatically detect the relevant cluster (e.g., 'sherlock')
    from the query text. You can also explicitly specify the cluster in the request body.
    """
    if llm is None or not retrievers:
        raise HTTPException(status_code=503, detail="Service Unavailable: Model or retrievers are not loaded.")

    # 1. Determine the cluster
    if request.cluster and request.cluster in retrievers:
        cluster = request.cluster
        logging.info(f"Using specified cluster: {cluster}")
    else:
        cluster = identify_cluster(request.query)
        logging.info(f"Identified cluster from query: {cluster}")

    if cluster == "unknown" or cluster not in retrievers:
        cluster_list = list(retrievers.keys())
        cluster_list_comma = ', '.join(cluster_list)
        return QueryResponse(
        answer=f"I need to know which cluster you are using. Please ask your question again and specify one of: {cluster_list_comma.capitalize()}",
        cluster="",
        sources=[]
    )

    # 2. Retrieve relevant documents
    retriever = retrievers[cluster]
    retrieved_docs: List[Document] = retriever.invoke(request.query)
    log.info(f"retrieved_docs: {retrieved_docs}")
    if not retrieved_docs:
        logging.warning(f"No documents found for query: '{request.query}' in cluster '{cluster}'")
        return QueryResponse(
            answer=f"I could not find any relevant information for your query in the {cluster.capitalize()} documentation. Please try rephrasing your question or contact srcc-support@stanford.edu.",
            cluster=cluster,
            sources=[]
        )
    
    # 3. Generate the prompt for the LLM
    retrieved_content = "\n\n".join(
        [f"--- Document: {doc.metadata.get('url', 'N/A')} ---\n{doc.page_content}" for doc in retrieved_docs]
    )


    prompt = f"""
### Task:
Summarize the user's query based on the information provided in the retrieved documents. Include inline citations in the format [source: Document X] where X represents the document number. Do not provide an answer unless it is supported by the context in the retrieved documents.

### User Query:
{request.query}

### Retrieved Documents:
{retrieved_content}


### Response:
"""

    # 4. Invoke the LLM to get the answer
    try:
        logging.info(f"Invoking LLM for cluster '{cluster}' with query '{request.query[:50]}...'")
        response_text = llm.invoke(prompt)
        logging.info(f"LLM invocation successful {response_text}.")
    except Exception as e:
        logging.error(f"Error during LLM invocation: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate a response from the model.")

    # 5. Process the response and extract sources
    final_answer = response_text.strip()
    
    # Use a regex to find all cited sources like [source: filename.md]
    cited_sources_names = set(re.findall(r'\[source:\s*([^\]]+)\]', final_answer))
    
    # Replace the citations with a simpler format for the user, e.g., (Source 1) - not doing for now.
    # For now, let's just extract them and leave the answer as is.
    
    # Create a list of source objects based on the retrieved docs that were cited
    source_objects = []
    for doc in retrieved_docs:
        doc_source_name = doc.metadata.get('url')
        if doc_source_name in cited_sources_names:
            source_objects.append(
                Source(name=doc_source_name, url=doc.metadata.get('url'))
            )
            cited_sources_names.remove(doc_source_name) # Avoid duplicates

    return QueryResponse(
        answer=final_answer,
        cluster=cluster,
        sources=source_objects
    )