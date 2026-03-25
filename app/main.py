import os
os.environ.setdefault('HF_HUB_OFFLINE', '1')

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException # pyright: ignore[reportMissingImports]
from fastapi.middleware.cors import CORSMiddleware # pyright: ignore[reportMissingImports]

from app.config import settings
from app.models import QueryRequest, QueryResponse
from app.rag_service import RAGService

# Setup logging
os.makedirs(settings.LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    filename=os.path.join(settings.LOG_DIR, 'myapp.log'),
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize RAG service
rag_service = RAGService(settings)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - startup and shutdown"""
    logger.info("Application startup...")
    try:
        rag_service.initialize()
        logger.info("RAG service initialized successfully.")
    except Exception as e:
        logger.critical(f"FATAL: RAG service initialization failed. Error: {e}")
        raise
    
    yield
    
    logger.info("Application shutdown.")


# Create FastAPI application
app = FastAPI(
    title=settings.APP_TITLE,
    description=settings.APP_DESCRIPTION,
    version=settings.APP_VERSION,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


# Routes
@app.get("/", summary="Root endpoint")
async def root():
    """Root endpoint - returns service status"""
    return {"message": f"{settings.APP_TITLE} is running."}


@app.get("/health", summary="Health check endpoint")
async def health_check():
    """Health check endpoint - returns OK when model is loaded and ready"""
    if rag_service.model is None or rag_service.chain is None:
        raise HTTPException(
            status_code=503,
            detail="Service not ready - model still loading"
        )
    if not rag_service.retrievers:
        raise HTTPException(
            status_code=503,
            detail="Service not ready - no document retrievers loaded"
        )
    return {
        "status": "healthy",
        "model_loaded": True,
        "device": settings.MODEL_DEVICE,
        "clusters": list(rag_service.retrievers.keys())
    }


@app.post("/query/", response_model=QueryResponse, summary="Query the knowledge base")
async def query_kb(request: QueryRequest):
    """
    Query the knowledge base with a question.
    
    - **query**: The question to ask
    - **cluster**: Optional cluster name (sherlock, farmshare, oak, elm)
    """
    try:
        result = rag_service.query(request)
        return result
    except Exception as e:
        logger.error(f"Error handling query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.post("/cache/clear", summary="Clear the semantic response cache")
async def clear_cache():
    """
    Flush the semantic response cache.  Useful before running the test suite
    to ensure test queries get fresh model responses rather than cached ones.
    """
    if rag_service.semantic_cache is None:
        return {"cleared": False, "reason": "semantic cache not enabled"}
    try:
        rag_service.semantic_cache.clear()
        logger.info("Semantic cache cleared via API")
        return {"cleared": True}
    except Exception as e:
        logger.error(f"Failed to clear semantic cache: {e}")
        raise HTTPException(status_code=500, detail=f"Cache clear failed: {e}")


@app.get("/stats", summary="Worker statistics")
async def worker_stats():
    """Return worker-specific statistics"""
    return {
        "status": "healthy",
        "device": settings.MODEL_DEVICE,
        "model_type": settings.MODEL_TYPE,
        "clusters": list(rag_service.retrievers.keys()),
        "cache_enabled": settings.SEMANTIC_CACHE_ENABLED
    }
