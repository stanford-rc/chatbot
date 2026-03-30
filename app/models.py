from typing import Optional, List
from pydantic import BaseModel

class QueryRequest(BaseModel):
    """Request model for query endpoint"""
    query: str
    # TODO(production): make cluster required (str, no default) once the upstream
    # portal/routing layer is passing it in every request. For now it stays optional
    # so internal test calls without a cluster field don't break.
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
