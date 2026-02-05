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
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Semantic caching - optional dependency
try:
    from app.semantic_cache import SemanticResponseCache
    SEMANTIC_CACHE_AVAILABLE = True
except ImportError:
    SEMANTIC_CACHE_AVAILABLE = False
    SemanticResponseCache = None

# --- 1. Configuration Management ---

logging.basicConfig(level=logging.INFO, filename='myapp.log', format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
load_dotenv()

# Load centralized configuration
def load_config():
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

config = load_config()

class Settings(BaseSettings):
    APP_TITLE: str = config['app']['title']
    APP_DESCRIPTION: str = config['app']['description']
    APP_VERSION: str = config['app']['version']
    MODEL_PATH: str = Field(default=config['model']['path'], env="MODEL_PATH")
    MODEL_TYPE: str = config['model']['type']
    # WORKER_GPU env var overrides config for multi-worker (Python 3.14 Pydantic compat issue)
    MODEL_DEVICE: str = os.environ.get('WORKER_GPU', config['model']['device'])
    USE_QUANTIZATION: bool = config['model']['use_quantization']
    LOCAL_FILES_ONLY: bool = config['model']['local_files_only']
    MAX_NEW_TOKENS: int = config['generation']['max_new_tokens']
    CLUSTERS: Dict[str, str] = config['clusters']
    CORS_ORIGINS: List[str] = config['api']['cors_origins']
    
    # Caching settings
    SEMANTIC_CACHE_ENABLED: bool = config.get('caching', {}).get('SEMANTIC_CACHE_ENABLED', True)
    SEMANTIC_CACHE_THRESHOLD: float = config.get('caching', {}).get('SEMANTIC_CACHE_THRESHOLD', 0.70)
    SEMANTIC_CACHE_DB: str = config.get('caching', {}).get('SEMANTIC_CACHE_DB', '.response_cache.db')
    LANGCHAIN_CACHE_DB: str = config.get('caching', {}).get('LANGCHAIN_CACHE_DB', '.langchain.db')
    
    # Retrieval settings
    MAX_RETRIEVED_DOCS: int = config.get('retrieval', {}).get('MAX_RETRIEVED_DOCS', 5)
    
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
        self.semantic_cache = None  # Initialized in initialize()

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
            set_llm_cache(SQLiteCache(database_path=self.settings.LANGCHAIN_CACHE_DB))
            logger.info(f"LangChain LLM cache enabled: {self.settings.LANGCHAIN_CACHE_DB}")
        except Exception as e:
            logger.error(f"Could not set up LLM cache: {e}")
        
        # Initialize semantic response cache
        if SEMANTIC_CACHE_AVAILABLE and self.settings.SEMANTIC_CACHE_ENABLED:
            try:
                self.semantic_cache = SemanticResponseCache(
                    db_path=self.settings.SEMANTIC_CACHE_DB,
                    similarity_threshold=self.settings.SEMANTIC_CACHE_THRESHOLD
                )
                logger.info(f"✓ Semantic cache initialized (threshold: {self.settings.SEMANTIC_CACHE_THRESHOLD})")
            except Exception as e:
                logger.error(f"Failed to initialize semantic cache: {e}")
                self.semantic_cache = None
        else:
            logger.warning("Semantic cache disabled (sentence-transformers not installed)")
            self.semantic_cache = None


        try:
            logger.info(f"Loading model from {self.settings.MODEL_PATH}...")
            logger.info(f"Model type: {self.settings.MODEL_TYPE}, Device: {self.settings.MODEL_DEVICE}, Quantization: {self.settings.USE_QUANTIZATION}")
            
            # Determine device and dtype
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            logger.info("Using Device: %s, dtype: %s", device, torch_dtype)
            
            # Load tokenizer from local directory
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.settings.MODEL_PATH,
                local_files_only=self.settings.LOCAL_FILES_ONLY
	    )
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model - FP16 without quantization for ARM compatibility
            if self.settings.USE_QUANTIZATION:
                logger.warning("Quantization enabled - may cause slowdown on ARM architecture")
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.settings.MODEL_PATH,
                    quantization_config=bnb_config,
                    device_map="auto" if torch.cuda.is_available() else None,
                    local_files_only=self.settings.LOCAL_FILES_ONLY,
                    torch_dtype=torch.float16,
                )
            else:
                logger.info("Loading model in FP16 (recommended for ARM)")
                # Use config device instead of device_map='auto' to avoid multi-GPU overhead
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.settings.MODEL_PATH,
                    torch_dtype=torch_dtype,
                    local_files_only=self.settings.LOCAL_FILES_ONLY
                )
                if torch.cuda.is_available():
                    logger.info(f"Loading model to device: {self.settings.MODEL_DEVICE}")
                    logger.info(f"WORKER_GPU env var: {os.environ.get('WORKER_GPU', 'NOT SET')}")
                    self.model = self.model.to(self.settings.MODEL_DEVICE)
                    logger.info(f"Model moved to {self.settings.MODEL_DEVICE}")

            self.model.eval()  # Set to inference mode
            
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

        # Adjust prompt template based on model type
        if self.settings.MODEL_TYPE == "llama":
            prompt_template = ChatPromptTemplate.from_template(
                """<s>[INST] You are an expert assistant for the Stanford Research Computing Center (SRCC).
Your task is to answer the user's query based ONLY on the provided documentation context.
- Your answer must be grounded in the facts from the CONTEXT below.
- Determine which cluster documentation to consult based on the user's input. If they don't supply an identifiable cluster, you may ask for more information. 
- If the context does not contain the answer, state that you could not find the information and refer the user to srcc-support@stanford.edu.
- When you reference information from a document, cite it using [Title] where Title is the exact title from the document's metadata.
- Answer ONLY the user's query. Do not add any extra information, questions, or conversational text after the answer is complete.
- Prioritize bulleted steps for the practical completion of a user's task.
CONTEXT:
{context}

USER QUERY:
{query} [/INST]"""
            )
        else:
            # Gemma and other models - simpler prompt
            prompt_template = ChatPromptTemplate.from_template(
                """You are an expert assistant for the Stanford Research Computing Center (SRCC).
Your task is to answer the user's query based ONLY on the provided documentation context.
- Your answer must be grounded in the facts from the CONTEXT below.
- Determine which cluster documentation to consult based on the user's input. If they don't supply an identifiable cluster, you may ask for more information. 
- If the context does not contain the answer, state that you could not find the information and refer the user to srcc-support@stanford.edu.
- When you reference information from a document, cite it using [Title] where Title is the exact title from the document's metadata.
- Answer ONLY the user's query. Do not add any extra information, questions, or conversational text after the answer is complete.
- Prioritize bulleted steps for the practical completion of a user's task.
CONTEXT:
{context}

USER QUERY:
{query}"""
            )

        def retrieve_and_format_context(inputs: Dict) -> str:
            query, cluster = inputs['query'], inputs['cluster']
            retriever = self.retrievers[cluster]
            retrieved_docs = retriever.invoke(query)
            retrieved_docs = retrieved_docs[:self.settings.MAX_RETRIEVED_DOCS]
            inputs['retrieved_docs'] = retrieved_docs
            if not retrieved_docs:
                return "No relevant documents were found."
            # Format with title so LLM can cite by title
            return "\n\n".join(
                f"--- Document: {doc.metadata.get('title', doc.metadata.get('source', 'Unknown'))} ---\n{doc.page_content}"
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
            # Extract content from LangChain message object
            if isinstance(inputs, dict) and 'messages' in inputs:
                query = inputs['messages'][0].content
            else:
                query = str(inputs)
            
            # Format prompt based on model type
            if self.settings.MODEL_TYPE == "gemma":
                # Gemma uses simple tokenization without chat template
                inputs_encoded = self.tokenizer(query, return_tensors="pt")
            elif self.settings.MODEL_TYPE in ["llama", "tinyllama"]:
                # Llama models use chat template
                messages = [{"role": "user", "content": query}]
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                inputs_encoded = self.tokenizer(prompt, return_tensors="pt")
            else:
                # Default: simple tokenization
                inputs_encoded = self.tokenizer(query, return_tensors="pt")
            
            if torch.cuda.is_available():
                inputs_encoded = inputs_encoded.to(self.settings.MODEL_DEVICE)
            
            # Generate response with optimized parameters
            logger.info("Starting model generation...")
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs_encoded,
                    max_new_tokens=self.settings.MAX_NEW_TOKENS,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                )
            logger.info(f"Generation complete. Tokens generated: {outputs.shape[1] - inputs_encoded['input_ids'].shape[1]}")
            
            # Decode only the new tokens (exclude the prompt)
            response = self.tokenizer.decode(
                outputs[0][inputs_encoded['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error during LLM generation: {e}")
            raise ValueError(f"LLM generation error: {e}")

    def _identify_cluster(self, user_query: str) -> str:
        user_query_lower = user_query.lower()
        for cluster_name in self.settings.CLUSTERS.keys():
            if cluster_name in user_query_lower:
                return cluster_name
        return "unknown"

    def _format_sources(self, documents: List[Document]) -> List[Source]:
        """
        Format documents into Source objects with titles and URLs.
        Extracts URL from document metadata (added by file_magic.py).
        """
        sources = []
        seen_titles = set()
        
        for doc in documents:
            title = doc.metadata.get('title', 'Unknown')
            url = doc.metadata.get('url', None)  # Extract URL from frontmatter
            
            # Avoid duplicate sources with same title
            if title not in seen_titles:
                sources.append(Source(title=title, url=url))
                seen_titles.add(title)
        
        return sources

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
        
        # Check semantic cache first
        if self.semantic_cache:
            cached_response = self.semantic_cache.get(request.query, cluster)
            if cached_response:
                return QueryResponse(**cached_response)
        
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


        # Extract citations from LLM answer (format: [Title])
        cited_titles = set(re.findall(r'\[([^\]]+)\]', llm_answer_with_placeholders))
        logger.info(f"LLM cited {len(cited_titles)} sources: {cited_titles}")
        
        # Build title → URL mapping from retrieved docs
        title_to_url = {}
        title_to_doc = {}
        for doc in retrieved_docs:
            title = doc.metadata.get('title', 'Unknown')
            url = doc.metadata.get('url', None)
            title_to_url[title] = url
            title_to_doc[title] = doc
        
        # Convert [Title] citations to inline markdown links [Title](URL)
        final_answer = llm_answer_with_placeholders
        for title in cited_titles:
            if title in title_to_url and title_to_url[title]:
                # Replace [Title] with [Title](url)
                final_answer = final_answer.replace(
                    f'[{title}]',
                    f'[{title}]({title_to_url[title]})'
                )
        
        # Build sources list from only cited documents
        cited_docs = [title_to_doc[title] for title in cited_titles if title in title_to_doc]
        source_objects = self._format_sources(cited_docs) if cited_docs else []
        
        # Append source URLs as markdown reference list
        if source_objects:
            source_list = "\n\n📚 **Sources:**\n" + "\n".join(
                f"- [{src.title}]({src.url})" if src.url else f"- {src.title}"
                for src in source_objects
            )
            final_answer = final_answer + source_list

        response = QueryResponse(
            answer=final_answer,
            cluster=cluster,
            sources=source_objects
        )
        
        # Cache the response for future similar queries
        if self.semantic_cache:
            try:
                self.semantic_cache.set(
                    request.query,
                    cluster,
                    response.model_dump()
                )
            except Exception as e:
                logger.error(f"Failed to cache response: {e}")
        
        return response

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
