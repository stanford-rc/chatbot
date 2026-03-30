"""
stats.py — In-memory usage statistics tracker for Ada.

Tracks query volume, cache performance, latency, errors, per-cluster
usage, and popular queries. All counters reset on service restart.
Access via GET /stats.
"""

import threading
from collections import defaultdict, Counter
from datetime import datetime, timezone
from typing import Dict, List


class StatsTracker:
    def __init__(self):
        self._lock = threading.Lock()
        self.start_time = datetime.now(timezone.utc)

        self.total_queries: int = 0
        self.cache_hits: int = 0
        self.cache_misses: int = 0
        self.errors: int = 0

        self.queries_by_cluster: Dict[str, int] = defaultdict(int)
        self.queries_by_hour: Dict[str, int] = defaultdict(int)

        # Keep last 1000 latency samples for percentile calculation
        self._latencies: List[float] = []

        # Track top queries (normalised, truncated to 200 chars)
        self._query_counter: Counter = Counter()

    # ── Recording ──────────────────────────────────────────────────────────

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

            # Popular queries
            normalised = query.strip().lower()[:200]
            self._query_counter[normalised] += 1

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
                "top_queries": self._query_counter.most_common(10),
            }

    def reset(self) -> None:
        """Reset all counters (e.g. for testing)."""
        with self._lock:
            self.__init__()


# Module-level singleton — imported by rag_service and main
stats_tracker = StatsTracker()
