"""
stats.py — Usage statistics tracker.

Every recorded query is persisted to the shared SQLite db at
`settings.STATS_DB` (see stats_db.py). Aggregates for the dashboard
(`/stats`) are computed by querying the db with service/scope/date
filters. Semantic top-queries clustering is kept in-memory per-process
(embedding lookups are too slow to run per dashboard request).

Public API — unchanged for compatibility with existing callers:
    stats_tracker.set_log_path(path)          # now points at the sqlite db path
    stats_tracker.set_embedding_model(model)
    stats_tracker.set_feedback_log_path(path) # no-op; kept for backwards compat
    stats_tracker.record_query(
        cluster, query, latency_s, cache_hit,
        error=False, service=None, scope=None, tool=None, caller=None,
    )
    stats_tracker.record_feedback(query_id, cluster, rating, tags,
                                  cluster_correction, comment,
                                  service=None, scope=None)
    stats_tracker.get_stats(services=None, scopes=None, since=None, until=None) -> dict

Each service that instantiates this tracker should call
    StatsTracker(default_service='chatbot',    default_scope='public')
or set the defaults after construction via `set_service_defaults()`. Then
individual call sites can override via kwargs where needed.
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from contextlib import closing
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

from app import stats_db

logger = logging.getLogger(__name__)


class StatsTracker:
    def __init__(
        self,
        similarity_threshold: float = 0.70,
        default_service: str = "",
        default_scope: str = "unknown",
    ):
        self._lock = threading.Lock()
        self.start_time = datetime.now(timezone.utc)
        self._similarity_threshold = similarity_threshold

        # Service defaults — call sites can still override per-call.
        self._default_service = default_service
        self._default_scope = default_scope

        # Path to the shared SQLite db (set via set_log_path)
        self._db_path: str = ""

        # Semantic query clustering (in-memory, per-process).
        # _canonical_queries: list of (canonical_text, embedding, count)
        self._canonical_queries: List[Tuple[str, np.ndarray, int]] = []

        # Embedding model — injected after startup via set_embedding_model()
        self._embedding_model = None

    # ── Configuration ─────────────────────────────────────────────────────

    def set_embedding_model(self, model) -> None:
        """Inject the sentence-transformers model after it's loaded at startup."""
        self._embedding_model = model

    def set_log_path(self, path: str) -> None:
        """
        Point the tracker at the shared SQLite db. Name kept for
        backwards compatibility with the legacy JSONL API — path is now
        expected to be a `.sqlite` file, not `.jsonl`.
        """
        self._db_path = path
        if path:
            try:
                stats_db.init_db(path)
            except Exception as e:
                logger.error(f"Failed to init stats db at {path}: {e}")

    def set_feedback_log_path(self, path: str) -> None:
        """No-op — feedback now writes to the same SQLite db as queries."""
        pass

    def set_service_defaults(self, service: str, scope: str) -> None:
        self._default_service = service
        self._default_scope = scope

    # ── Recording ─────────────────────────────────────────────────────────

    def _embed(self, text: str) -> Optional[np.ndarray]:
        if self._embedding_model is None:
            return None
        try:
            vec = self._embedding_model.encode(text, convert_to_numpy=True)
            return vec / (np.linalg.norm(vec) + 1e-10)
        except Exception:
            return None

    def _find_or_create_canonical(self, query: str) -> None:
        """
        Find the most similar existing canonical query and increment its count,
        or create a new canonical entry if no match exceeds the threshold.
        Embedding happens outside the lock to minimise contention.
        """
        normalised = (query or "").strip().lower()[:200]
        if not normalised:
            return
        embedding = self._embed(normalised)

        with self._lock:
            if embedding is not None and self._canonical_queries:
                canonicals_emb = np.stack([e for _, e, _ in self._canonical_queries])
                sims = canonicals_emb @ embedding
                best_idx = int(np.argmax(sims))
                if sims[best_idx] >= self._similarity_threshold:
                    text, emb, count = self._canonical_queries[best_idx]
                    self._canonical_queries[best_idx] = (text, emb, count + 1)
                    return

            self._canonical_queries.append(
                (normalised, embedding if embedding is not None else np.array([]), 1)
            )

    def record_query(
        self,
        cluster: str = "",
        query: str = "",
        latency_s: float = 0.0,
        cache_hit: Optional[bool] = None,
        error: bool = False,
        service: Optional[str] = None,
        scope: Optional[str] = None,
        tool: Optional[str] = None,
        caller: Optional[str] = None,
    ) -> None:
        service = service or self._default_service
        scope = scope or self._default_scope

        if not self._db_path:
            # No db configured — semantic canonicals still work, but no persistence.
            if query:
                self._find_or_create_canonical(query)
            return

        try:
            with closing(stats_db.connect(self._db_path)) as conn:
                stats_db.insert_query(
                    conn,
                    service=service,
                    scope=scope,
                    latency_s=latency_s,
                    cache_hit=cache_hit,
                    error=error,
                    tool=tool,
                    cluster=cluster or None,
                    query=query or None,
                    caller=caller,
                )
        except Exception as e:
            # Never let stats persistence break query serving.
            logger.warning(f"stats_db insert_query failed: {e}")

        # Semantic canonicals run outside any db work
        if query:
            self._find_or_create_canonical(query)

    def record_feedback(
        self,
        query_id: Optional[str],
        cluster: str,
        rating: Optional[int],
        tags: Optional[List[str]],
        cluster_correction: Optional[str],
        comment: Optional[str],
        service: Optional[str] = None,
        scope: Optional[str] = None,
    ) -> None:
        service = service or self._default_service
        scope = scope or self._default_scope

        if not self._db_path:
            return

        try:
            with closing(stats_db.connect(self._db_path)) as conn:
                stats_db.insert_feedback(
                    conn,
                    service=service,
                    scope=scope,
                    query_id=query_id,
                    cluster=cluster or None,
                    rating=rating,
                    tags=tags,
                    cluster_correction=cluster_correction,
                    comment=comment,
                )
        except Exception as e:
            logger.warning(f"stats_db insert_feedback failed: {e}")

    # ── Reporting ─────────────────────────────────────────────────────────

    def get_stats(
        self,
        services: Optional[List[str]] = None,
        scopes: Optional[List[str]] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
    ) -> dict:
        """
        Return dashboard-shaped stats aggregated from the SQLite db, with
        service/scope/date filters. Semantic top-queries are per-process
        (in-memory) and are not filtered by service — they reflect only
        this tracker's own record_query calls since process start.
        """
        uptime_s = (datetime.now(timezone.utc) - self.start_time).total_seconds()

        aggregates = (
            stats_db.query_aggregates(
                self._db_path,
                services=services,
                scopes=scopes,
                since=since,
                until=until,
            )
            if self._db_path
            else _empty_aggregates()
        )
        feedback = (
            stats_db.query_feedback_aggregates(
                self._db_path,
                services=services,
                scopes=scopes,
                since=since,
                until=until,
            )
            if self._db_path
            else _empty_feedback()
        )

        with self._lock:
            top_queries = sorted(
                [(text, count) for text, _, count in self._canonical_queries],
                key=lambda x: x[1],
                reverse=True,
            )[:10]

        return {
            "uptime_seconds": round(uptime_s),
            "since": self.start_time.strftime("%Y-%m-%d %H:%M UTC"),
            "total_queries":       aggregates["total_queries"],
            "cache":               aggregates["cache"],
            "errors":              aggregates["errors"],
            "latency":             aggregates["latency"],
            "queries_by_cluster":  aggregates["queries_by_cluster"],
            "queries_by_service":  aggregates["queries_by_service"],
            "queries_by_tool":     aggregates["queries_by_tool"],
            "queries_by_hour":     aggregates["queries_by_hour"],
            "top_queries":         top_queries,
            "feedback":            feedback,
        }

    def reset(self) -> None:
        """Reset the in-memory canonicals + start_time. Does NOT touch SQLite."""
        with self._lock:
            threshold = self._similarity_threshold
            default_service = self._default_service
            default_scope = self._default_scope
            db_path = self._db_path
            embedding_model = self._embedding_model
            self.__init__(
                similarity_threshold=threshold,
                default_service=default_service,
                default_scope=default_scope,
            )
            self._db_path = db_path
            self._embedding_model = embedding_model


def _empty_aggregates() -> dict:
    return {
        "total_queries": 0,
        "cache": {"hits": 0, "misses": 0, "hit_rate": 0.0},
        "errors": 0,
        "latency": {"avg_s": 0.0, "p50_s": 0.0, "p95_s": 0.0, "p99_s": 0.0, "sample_count": 0},
        "queries_by_cluster": {},
        "queries_by_service": {},
        "queries_by_tool": {},
        "queries_by_hour": {},
    }


def _empty_feedback() -> dict:
    return {"total": 0, "satisfaction": 0.0, "up": 0, "down": 0, "by_cluster": {}}


# Chatbot's tracker singleton. Public docbot → always public scope.
stats_tracker = StatsTracker(default_service="chatbot", default_scope="public")
