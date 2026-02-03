#!/usr/bin/env python3
"""
Load Balancer with Request Queue for Multi-GPU Workers
Accepts unlimited concurrent requests and queues them intelligently
"""
import asyncio
import httpx
import logging
from collections import deque
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Worker configuration
WORKERS = [
    {"url": "http://localhost:8001", "gpu": "cuda:0"},
    {"url": "http://localhost:8002", "gpu": "cuda:1"},
]

# Queue and worker state
request_queue = deque()
worker_busy = {worker["url"]: False for worker in WORKERS}
worker_stats = {worker["url"]: {"processed": 0, "errors": 0} for worker in WORKERS}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    logger.info("Load Balancer starting...")
    logger.info(f"Managing {len(WORKERS)} GPU workers")
    for worker in WORKERS:
        logger.info(f"  - {worker['url']} ({worker['gpu']})")
    yield
    logger.info("Load Balancer shutting down...")


app = FastAPI(
    title="Multi-GPU Load Balancer",
    description="Queues and routes requests to available GPU workers",
    lifespan=lifespan
)


async def get_available_worker():
    """Find first available worker, or None if all busy"""
    for worker in WORKERS:
        if not worker_busy[worker["url"]]:
            return worker
    return None


async def process_request_on_worker(worker, request_data):
    """Send request to specific worker and return response"""
    worker_url = worker["url"]
    worker_busy[worker_url] = True
    
    try:
        logger.info(f"Routing request to {worker_url} ({worker['gpu']})")
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"{worker_url}/query/",
                json=request_data
            )
            response.raise_for_status()
            worker_stats[worker_url]["processed"] += 1
            return response.json()
    except Exception as e:
        logger.error(f"Error from {worker_url}: {e}")
        worker_stats[worker_url]["errors"] += 1
        raise HTTPException(status_code=500, detail=f"Worker error: {str(e)}")
    finally:
        worker_busy[worker_url] = False


@app.post("/query/")
async def query_with_queue(request: Request):
    """
    Accept query and route to available worker
    If all workers busy, wait and retry
    """
    request_data = await request.json()
    
    # Keep trying until we get a worker
    while True:
        # Check for available worker
        worker = await get_available_worker()
        
        if worker:
            # Worker available - process immediately
            return await process_request_on_worker(worker, request_data)
        else:
            # All workers busy - wait a bit and retry
            await asyncio.sleep(0.5)


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "workers": len(WORKERS),
        "worker_status": {
            url: "busy" if busy else "available"
            for url, busy in worker_busy.items()
        },
        "stats": worker_stats
    }


@app.get("/stats")
async def stats():
    """Detailed statistics"""
    return {
        "total_workers": len(WORKERS),
        "available_workers": sum(1 for busy in worker_busy.values() if not busy),
        "worker_stats": worker_stats,
        "workers": [
            {
                "url": worker["url"],
                "gpu": worker["gpu"],
                "status": "busy" if worker_busy[worker["url"]] else "available",
                "processed": worker_stats[worker["url"]]["processed"],
                "errors": worker_stats[worker["url"]]["errors"]
            }
            for worker in WORKERS
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
