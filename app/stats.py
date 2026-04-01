"""
stats.py — In-memory usage statistics tracker for Ada.

Tracks query volume, cache performance, latency, errors, per-cluster
usage, and popular queries. All counters reset on service restart.

Popular queries are semantically collapsed: similar phrasings are grouped
under a single canonical representative using the same sentence-transformers
model as the semantic cache (runs on CPU, no GPU impact).

Access via GET /stats.
"""

import json
import threading
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np


class StatsTracker:
    def __init__(self, similarity_threshold: float = 0.70):
        self._lock = threading.Lock()
        self.start_time = datetime.now(timezone.utc)
        self._similarity_threshold = similarity_threshold

        self.total_queries: int = 0
        self.cache_hits: int = 0
        self.cache_misses: int = 0
        self.errors: int = 0

        self.queries_by_cluster: Dict[str, int] = defaultdict(int)
        self.queries_by_hour: Dict[str, int] = defaultdict(int)

        # Keep last 1000 latency samples for percentile calculation
        self._latencies: List[float] = []

        # Semantic query clustering:
        #   _canonical_queries: list of (canonical_text, embedding, count)
        self._canonical_queries: List[Tuple[str, np.ndarray, int]] = []

        # Embedding model — injected after startup via set_embedding_model()
        self._embedding_model = None

        # Path to append-only JSON-lines log — injected via set_log_path()
        self._log_path: str = ''

    def set_embedding_model(self, model) -> None:
        """Inject the sentence-transformers model after it's loaded at startup."""
        self._embedding_model = model

    def set_log_path(self, path: str) -> None:
        """Inject the stats log file path from settings."""
        self._log_path = path

    # ── Recording ──────────────────────────────────────────────────────────

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
        normalised = query.strip().lower()[:200]
        embedding = self._embed(normalised)

        with self._lock:
            if embedding is not None and self._canonical_queries:
                # Cosine similarities against all canonicals
                canonicals_emb = np.stack([e for _, e, _ in self._canonical_queries])
                sims = canonicals_emb @ embedding
                best_idx = int(np.argmax(sims))
                if sims[best_idx] >= self._similarity_threshold:
                    text, emb, count = self._canonical_queries[best_idx]
                    self._canonical_queries[best_idx] = (text, emb, count + 1)
                    return

            # No match — new canonical
            self._canonical_queries.append((normalised, embedding if embedding is not None else np.array([]), 1))

    def record_query(
        self,
        cluster: str,
        query: str,
        latency_s: float,
        cache_hit: bool,
        error: bool = False,
    ) -> None:
        with self._lock:
            self.total_queries += 1

            if error:
                self.errors += 1
            elif cache_hit:
                self.cache_hits += 1
            else:
                self.cache_misses += 1

            self.queries_by_cluster[cluster] += 1

            # Latency ring-buffer
            self._latencies.append(latency_s)
            if len(self._latencies) > 1000:
                self._latencies.pop(0)

            # Hourly bucket (UTC)
            hour_key = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:00 UTC")
            self.queries_by_hour[hour_key] += 1

            # Append one JSON line to the persistent stats log
            if self._log_path:
                try:
                    record = {
                        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "cluster": cluster,
                        "query": query,
                        "latency_s": round(latency_s, 3),
                        "cache_hit": cache_hit,
                        "error": error,
                    }
                    with open(self._log_path, 'a') as f:
                        f.write(json.dumps(record) + '\n')
                except Exception:
                    pass  # never let logging break query serving

        # Semantic collapsing runs outside the main lock (embedding is slow)
        self._find_or_create_canonical(query)

    # ── Reporting ──────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        with self._lock:
            uptime_s = (datetime.now(timezone.utc) - self.start_time).total_seconds()

            # Latency percentiles
            sorted_lat = sorted(self._latencies)
            n = len(sorted_lat)

            def pct(p: float) -> float:
                if n == 0:
                    return 0.0
                idx = min(int(n * p), n - 1)
                return round(sorted_lat[idx], 3)

            avg_lat = round(sum(sorted_lat) / n, 3) if n > 0 else 0.0

            cache_total = self.cache_hits + self.cache_misses
            hit_rate = round(self.cache_hits / cache_total, 3) if cache_total > 0 else 0.0

            # Last 24 hourly buckets (sorted)
            recent_hours = dict(
                sorted(self.queries_by_hour.items())[-24:]
            )

            # Top 10 canonical queries by count
            top_queries = sorted(
                [(text, count) for text, _, count in self._canonical_queries],
                key=lambda x: x[1],
                reverse=True,
            )[:10]

            return {
                "uptime_seconds": round(uptime_s),
                "since": self.start_time.strftime("%Y-%m-%d %H:%M UTC"),
                "total_queries": self.total_queries,
                "cache": {
                    "hits": self.cache_hits,
                    "misses": self.cache_misses,
                    "hit_rate": hit_rate,
                },
                "errors": self.errors,
                "latency": {
                    "avg_s": avg_lat,
                    "p50_s": pct(0.50),
                    "p95_s": pct(0.95),
                    "p99_s": pct(0.99),
                    "sample_count": n,
                },
                "queries_by_cluster": dict(self.queries_by_cluster),
                "queries_by_hour": recent_hours,
                "top_queries": top_queries,
            }

    def reset(self) -> None:
        """Reset all counters (e.g. for testing)."""
        with self._lock:
            threshold = self._similarity_threshold
            self.__init__(similarity_threshold=threshold)


# Module-level singleton — imported by rag_service and main
stats_tracker = StatsTracker()
