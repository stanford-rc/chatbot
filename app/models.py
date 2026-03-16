from typing import Optional, List
from pydantic import BaseModel

class QueryRequest(BaseModel):
    """Request model for query endpoint"""
    query: str
    cluster: Optional[str] = None

class Source(BaseModel):
    """Source document with title and URL"""
    title: str
    url: Optional[str] = None

class QueryResponse(BaseModel):
    """Response model for query endpoint"""
    answer: str
    cluster: str
    sources: List[Source]
