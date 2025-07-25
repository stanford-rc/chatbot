# main.py

import logging
import os
import re
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import frontmatter
import torch
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.cache import SQLiteCache
from langchain_community.retrievers import BM25Retriever
from langchain_core.globals import set_llm_cache
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_huggingface import HuggingFacePipeline
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# --- 1. Configuration Management ---

logging.basicConfig(level=logging.INFO, filename='myapp.log', format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
load_dotenv()

class Settings(BaseSettings):
    APP_TITLE: str = "SRC Cluster Knowledge Base API"
    APP_DESCRIPTION: str = "An API to query documentation about Stanford's high-performance computing clusters."
    APP_VERSION: str = "1.0.0"
    MODEL_PATH: str = Field(..., env="MODEL_PATH")
    CLUSTERS: Dict[str, str] = {"sherlock": "sherlock/", "farmshare": "farmshare/", "oak": "oak/", "elm": "elm/"}
    CORS_ORIGINS: List[str] = ['http://localhost:5000', 'http://127.0.0.1:5000']
    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

settings = Settings()

# --- 2. Pydantic Models for API ---

class QueryRequest(BaseModel):
    query: str
    cluster: Optional[str] = None

class Source(BaseModel):
    title: str
    url: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    cluster: str
    sources: List[Source]

# --- 3. Core Application Logic (RAG Service) ---

class RAGService:
    def __init__(self, config: Settings):
        self.settings = config
        self.llm = None
        self.retrievers: Dict[str, BM25Retriever] = {}
        self.chain = None

    def _ingest_markdown_files(self, corpus_dir: str) -> List[Document]:
        documents = []
        for filename in os.listdir(corpus_dir):
            if filename.endswith(".md"):
                file_path = os.path.join(corpus_dir, filename)
                try:
                    post = frontmatter.load(file_path)
                    metadata = post.metadata
                    metadata['source'] = filename
                    doc = Document(page_content=post.content, metadata=metadata)
                    documents.append(doc)
                except Exception as e:
                    logger.warning(f"Could not read or parse front matter from file {file_path}: {e}")
        return documents

    def initialize(self):
        logger.info("Initializing RAG Service...")
        try:
            set_llm_cache(SQLiteCache(database_path=".langchain.db"))
            logger.info("LangChain LLM cache enabled with SQLite.")
        except Exception as e:
            logger.error(f"Could not set up LLM cache: {e}")

        try:
            logger.info(f"Loading model from {self.settings.MODEL_PATH}...")
            model = AutoModelForCausalLM.from_pretrained(
                self.settings.MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto"
            )
            tokenizer = AutoTokenizer.from_pretrained(self.settings.MODEL_PATH)
            pipe = pipeline(
                "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=400,
                temperature=0.5, do_sample=True, pad_token_id=tokenizer.eos_token_id,
                return_full_text=False
            )
            self.llm = HuggingFacePipeline(pipeline=pipe)
            logger.info("Model and pipeline loaded successfully.")
        except Exception as e:
            logger.critical(f"FATAL: Could not load the model. Error: {e}")
            raise

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        for cluster_name, path in self.settings.CLUSTERS.items():
            if not os.path.isdir(path):
                logger.warning(f"Directory not found for cluster '{cluster_name}': {path}. Skipping.")
                continue
            
            logger.info(f"Ingesting documents for cluster: {cluster_name}")
            documents = self._ingest_markdown_files(path)
            if not documents:
                logger.warning(f"No documents found for cluster '{cluster_name}'.")
                continue

            split_docs = text_splitter.split_documents(documents)
            self.retrievers[cluster_name] = BM25Retriever.from_documents(split_docs)
            logger.info(f"Created retriever for '{cluster_name}'.")

            prompt_template = ChatPromptTemplate.from_template(
            """<s>[INST] You are an expert assistant for the Stanford Research Computing Center (SRCC).
Your task is to answer the user's query based ONLY on the provided documentation context.
- Your answer must be grounded in the facts from the CONTEXT below.
- If the context does not contain the answer, state that you could not find the information.
- Do not reference any specific documents by their filenames. If you must refer to a file, look up the title in the file's metadata.
- Answer ONLY the user's query. Do not add any extra information, questions, or conversational text after the answer is complete.

CONTEXT:
{context}

USER QUERY:
{query} [/INST]"""
        )

        def retrieve_and_format_context(inputs: Dict) -> str:
            query, cluster = inputs['query'], inputs['cluster']
            retriever = self.retrievers[cluster]
            retrieved_docs = retriever.invoke(query)
            inputs['retrieved_docs'] = retrieved_docs
            if not retrieved_docs:
                return "No relevant documents were found."
            return "\n\n".join(
                f"--- Document: {doc.metadata['source']} ---\n{doc.page_content}"
                for doc in retrieved_docs
            )
        
        self.chain = (
            RunnablePassthrough()
            | RunnablePassthrough.assign(context=RunnableLambda(retrieve_and_format_context))
            | prompt_template
            | self.llm
            | StrOutputParser()
        )
        logger.info("RAG service initialization complete.")

    def _identify_cluster(self, user_query: str) -> str:
        user_query_lower = user_query.lower()
        for cluster_name in self.settings.CLUSTERS.keys():
            if cluster_name in user_query_lower:
                return cluster_name
        return "unknown"


    def query(self, request: QueryRequest) -> QueryResponse:
        """Processes a user query, returning a clean answer and a separate list of sources."""
        if not self.chain or not self.retrievers:
            raise HTTPException(status_code=503, detail="Service Unavailable: RAG service is not initialized.")

        cluster = request.cluster if request.cluster in self.retrievers else self._identify_cluster(request.query)
        
        if cluster == "unknown" or cluster not in self.retrievers:
            cluster_list_str = ", ".join(self.retrievers.keys())
            return QueryResponse(
                answer=f"I couldn't determine which cluster you're asking about. Please specify one of: {cluster_list_str.capitalize()}.",
                cluster="unknown",
                sources=[]
            )
        
        logger.info(f"Processing query for cluster '{cluster}': '{request.query[:100]}...'")

        chain_input = {"query": request.query, "cluster": cluster, "retrieved_docs": []}
        try:
            llm_answer_with_placeholders = self.chain.invoke(chain_input)
            logger.info(f"llm_answer_with_placeholders: {llm_answer_with_placeholders}")
            retrieved_docs = chain_input['retrieved_docs']
            logger.info(f"retrieved_docs: {retrieved_docs}")
        except Exception as e:
            logger.error(f"Error during RAG chain invocation: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to generate a response from the model.")

        # 1. Extract cited filenames and build the list of Source objects
        cited_filenames = set(re.findall(r'\[\s*([^\]]+)\]', llm_answer_with_placeholders))
        metadata_lookup: Dict[str, Dict] = {
            doc.metadata['source']: doc.metadata
            for doc in retrieved_docs if 'source' in doc.metadata
        } 

        source_objects = []
        for filename in sorted(list(cited_filenames)):
            if filename in metadata_lookup:
                metadata = metadata_lookup[filename]
                title = metadata.get('title', filename)
                url = metadata.get('url')
                source_objects.append(Source(title=title, url=url))
        
        # 2. Clean the LLM's answer by removing all internal citation placeholders.
        final_answer = re.sub(r'\s*\[source:\s*[^\]]+\]', '', llm_answer_with_placeholders).strip()

        # 3. Return the clean answer and the separate source list.
        return QueryResponse(
            answer=final_answer,
            cluster=cluster,
            sources=source_objects
        )


# --- 4. FastAPI Application Setup ---
rag_service = RAGService(settings)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application startup...")
    rag_service.initialize()
    yield
    logger.info("Application shutdown.")

app = FastAPI(
    title=settings.APP_TITLE,
    description=settings.APP_DESCRIPTION,
    version=settings.APP_VERSION,
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# --- 5. API Endpoints ---
@app.get("/", summary="Root endpoint")
async def root():
    return {"message": f"{settings.APP_TITLE} is running."}

@app.post("/query/", response_model=QueryResponse, summary="Query the knowledge base")
async def query_kb(request: QueryRequest):
    """
    Accepts a user query and returns a synthesized answer.
    The response body includes a separate list of source documents used to generate the answer.
    """
    return rag_service.query(request)