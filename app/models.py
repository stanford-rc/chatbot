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
    query_id: str          # UUID — pass back to /feedback to correlate ratings
    answer: str
    cluster: str
    sources: List[Source]


# ── Feedback ──────────────────────────────────────────────────────────────────

FEEDBACK_TAGS = {
    "wrong_answer",
    "incomplete",
    "outdated",
    "wrong_cluster",
}

class FeedbackRequest(BaseModel):
    """User rating for a query response."""
    query_id: str                          # from QueryResponse.query_id
    rating: int                            # 1 = thumbs up, -1 = thumbs down
    tags: Optional[List[str]] = None       # subset of FEEDBACK_TAGS (downvotes only)
    cluster_correction: Optional[str] = None  # correct cluster if wrong_cluster tagged
    comment: Optional[str] = None         # free-text, optional


class FeedbackResponse(BaseModel):
    received: bool
    query_id: str
