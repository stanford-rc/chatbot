import logging
import os
import re
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import frontmatter
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.cache import SQLiteCache
from langchain_community.retrievers import BM25Retriever
from langchain_core.globals import set_llm_cache
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- 1. Configuration Management ---

logging.basicConfig(level=logging.INFO, filename='myapp.log', format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
load_dotenv()

class Settings(BaseSettings):
    APP_TITLE: str = "SRC Cluster Knowledge Base API"
    APP_DESCRIPTION: str = "An API to query documentation about Stanford's high-performance computing clusters."
    APP_VERSION: str = "1.0.0"
    MODEL_PATH: str = Field(default="/srv/scratch/bcritt/chatbot2/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659", env="MODEL_PATH")
    CLUSTERS: Dict[str, str] = {"sherlock": "docs/sherlock/", "farmshare": "docs/farmshare/", "oak": "docs/oak/", "elm": "docs/elm/"}
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
                    logger.info(f"ok loading from {file_path}")
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
            
            # Determine device and dtype
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            logger.info("Using Device: %s, dtype: %s", device, torch_dtype)
            
            # Load tokenizer from local directory with local_files_only
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.settings.MODEL_PATH,
                local_files_only=True
            )
            
            # Load model with automatic device placement
            self.model = AutoModelForCausalLM.from_pretrained(
                self.settings.MODEL_PATH,
                torch_dtype=torch_dtype,
                device_map="auto" if torch.cuda.is_available() else None,
                local_files_only=True
            )
            
            self.llm = RunnableLambda(self.llama_runnable_llm)
            
            logger.info("Model and tokenizer loaded successfully.")
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
- Determine which cluster documentation to consult based on the user's input. If they don't supply an identifiable cluster, you may ask for more information. 
- If the context does not contain the answer, state that you could not find the information and refer the user to srcc-support@stanford.edu.
- Do not reference any specific documents by their filenames. If you must refer to a file, look up the title in the file's metadata.
- Answer ONLY the user's query. Do not add any extra information, questions, or conversational text after the answer is complete.
- Prioritize bulleted steps for the practical completion of a user's task.
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

    def llama_runnable_llm(self, inputs: Dict) -> str:
        try:
            logger.info(f"llama_runnable_llm received inputs: {inputs}")
            query = str(inputs)
            
            # Use Llama 3.1's chat template
            messages = [{"role": "user", "content": query}]
            prompt = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # Tokenize and move to device
            inputs_encoded = self.tokenizer(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs_encoded = inputs_encoded.to("cuda")
            
            # Generate response
            outputs = self.model.generate(
                **inputs_encoded,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Decode only the new tokens (exclude the prompt)
            response = self.tokenizer.decode(
                outputs[0][inputs_encoded['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            return response
            
        except RuntimeError as e:
            # Catch CUDA errors and retry with reduced parameters
            logger.error(f"CUDA error during inference: {e}, retrying with reduced tokens")
            try:
                outputs = self.model.generate(
                    **inputs_encoded,
                    max_new_tokens=256,  # Reduced
                    temperature=0.1,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                response = self.tokenizer.decode(
                    outputs[0][inputs_encoded['input_ids'].shape[1]:],
                    skip_special_tokens=True
                )
                return response
            except Exception as retry_error:
                logger.error(f"Retry failed: {retry_error}")
                raise ValueError(f"LLM generation error after retry: {retry_error}")
        except Exception as e:
            logger.error(f"Error during LLM generation: {e}")
            raise ValueError(f"LLM generation error: {e}")

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
        
        final_answer = re.sub(r'\s*\[source:\s*[^\]]+\]', '', llm_answer_with_placeholders).strip()

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

    try:
        rag_service.initialize()
        logger.info("RAG service initialized successfully.")
    except Exception as e:
        logger.critical(f"FATAL: RAG service initialization failed. Error: {e}")
        raise

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

@app.get("/", summary="Root endpoint")
async def root():
    return {"message": f"{settings.APP_TITLE} is running."}

@app.post("/query/", response_model=QueryResponse, summary="Query the knowledge base")
async def query_kb(request: QueryRequest):
    try:
        result = rag_service.query(request)
        return result
    except Exception as e:
        logger.error(f"Error handling the query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")
