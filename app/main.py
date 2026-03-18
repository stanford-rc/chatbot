import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.models import QueryRequest, QueryResponse
from app.rag_service import RAGService

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    filename='/workspace/logs/myapp.log',
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
