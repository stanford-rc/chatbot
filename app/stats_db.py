"""
stats_db.py — Shared SQLite persistence for cross-service query stats.

Written to by three emitters:
  - chatbot's stats_tracker      (service='chatbot',        scope='public')
  - src-mcp's stats_tracker      (service='src-mcp-rag'     scope='internal',
                                  or 'src-mcp-tool'         scope='internal')
  - claude-mcp's _StatsMiddleware (service='claude-mcp',    scope from token)

Read by src-mcp's /stats endpoint, which serves the unified dashboard on
port 8001. WAL mode enables safe concurrent writers.

Public API:
    init_db(db_path)                         -> create schema if missing
    connect(db_path)                         -> sqlite3.Connection
    insert_query(conn, ...)                  -> row_id
    query_aggregates(db_path, **filters)    -> dict shaped for the dashboard

The `query_aggregates` return dict matches the legacy StatsTracker.get_stats()
shape so the dashboard JS doesn't need to change beyond the filter widget.
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from typing import Iterable, Optional


_SCHEMA = """
CREATE TABLE IF NOT EXISTS queries (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ts         TEXT NOT NULL,
    service    TEXT NOT NULL,
    scope      TEXT NOT NULL,
    tool       TEXT,
    cluster    TEXT,
    query      TEXT,
    latency_s  REAL,
    cache_hit  INTEGER,
    error      INTEGER,
    caller     TEXT
);
CREATE INDEX IF NOT EXISTS ix_queries_ts      ON queries(ts);
CREATE INDEX IF NOT EXISTS ix_queries_service ON queries(service);
CREATE INDEX IF NOT EXISTS ix_queries_scope   ON queries(scope);
CREATE INDEX IF NOT EXISTS ix_queries_tool    ON queries(tool);

CREATE TABLE IF NOT EXISTS feedback (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                  TEXT NOT NULL,
    service             TEXT NOT NULL,
    scope               TEXT NOT NULL,
    query_id            TEXT,
    cluster             TEXT,
    rating              INTEGER,
    tags_json           TEXT,
    cluster_correction  TEXT,
    comment             TEXT
);
CREATE INDEX IF NOT EXISTS ix_feedback_ts      ON feedback(ts);
CREATE INDEX IF NOT EXISTS ix_feedback_service ON feedback(service);
"""


def init_db(db_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    with closing(connect(db_path)) as conn:
        conn.executescript(_SCHEMA)
        conn.commit()


def connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def insert_query(
    conn: sqlite3.Connection,
    service: str,
    scope: str,
    latency_s: float,
    cache_hit: Optional[bool] = None,
    error: bool = False,
    tool: Optional[str] = None,
    cluster: Optional[str] = None,
    query: Optional[str] = None,
    caller: Optional[str] = None,
    ts: Optional[str] = None,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO queries
          (ts, service, scope, tool, cluster, query, latency_s, cache_hit, error, caller)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            ts or _now_iso(),
            service,
            scope,
            tool,
            cluster,
            query,
            round(latency_s, 3) if latency_s is not None else None,
            None if cache_hit is None else (1 if cache_hit else 0),
            1 if error else 0,
            caller,
        ),
    )
    conn.commit()
    return cur.lastrowid


def insert_feedback(
    conn: sqlite3.Connection,
    service: str,
    scope: str,
    query_id: Optional[str],
    cluster: Optional[str],
    rating: Optional[int],
    tags: Optional[list],
    cluster_correction: Optional[str],
    comment: Optional[str],
) -> int:
    cur = conn.execute(
        """
        INSERT INTO feedback
          (ts, service, scope, query_id, cluster, rating, tags_json,
           cluster_correction, comment)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            _now_iso(),
            service,
            scope,
            query_id,
            cluster,
            rating,
            json.dumps(tags or []),
            cluster_correction,
            comment,
        ),
    )
    conn.commit()
    return cur.lastrowid


# ---------------------------------------------------------------------------
# Aggregation for the dashboard
# ---------------------------------------------------------------------------

def _where_clause(
    services: Optional[list[str]],
    scopes: Optional[list[str]],
    since: Optional[str],
    until: Optional[str],
) -> tuple[str, list]:
    frags: list[str] = []
    params: list = []
    if services:
        frags.append("service IN (" + ",".join("?" * len(services)) + ")")
        params.extend(services)
    if scopes:
        frags.append("scope IN (" + ",".join("?" * len(scopes)) + ")")
        params.extend(scopes)
    if since:
        frags.append("ts >= ?")
        params.append(since)
    if until:
        frags.append("ts <= ?")
        params.append(until)
    where = ("WHERE " + " AND ".join(frags)) if frags else ""
    return (where, params)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, int(round(pct / 100.0 * (len(ordered) - 1)))))
    return round(ordered[k], 3)


def query_aggregates(
    db_path: str,
    services: Optional[list[str]] = None,
    scopes: Optional[list[str]] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
    latency_sample_cap: int = 5000,
) -> dict:
    """
    Aggregate stats for the dashboard. Return shape matches the legacy
    StatsTracker.get_stats() output so the JS renderer doesn't have to change.
    top_queries and semantic-clustering fields are NOT included here — the
    live tracker in the serving process fills those in per-service.
    """
    where, params = _where_clause(services, scopes, since, until)

    with closing(connect(db_path)) as conn:
        total = conn.execute(
            f"SELECT COUNT(*) AS c FROM queries {where}", params
        ).fetchone()["c"]

        cache_hits = conn.execute(
            f"SELECT COUNT(*) AS c FROM queries {where} "
            + ("AND" if where else "WHERE") + " cache_hit = 1",
            params,
        ).fetchone()["c"]
        cache_misses = conn.execute(
            f"SELECT COUNT(*) AS c FROM queries {where} "
            + ("AND" if where else "WHERE") + " cache_hit = 0",
            params,
        ).fetchone()["c"]
        errors = conn.execute(
            f"SELECT COUNT(*) AS c FROM queries {where} "
            + ("AND" if where else "WHERE") + " error = 1",
            params,
        ).fetchone()["c"]

        # Latency sample — bounded by cap; percentiles computed in Python.
        lat_rows = conn.execute(
            f"SELECT latency_s FROM queries {where} "
            + ("AND" if where else "WHERE") + " latency_s IS NOT NULL "
            "ORDER BY id DESC LIMIT ?",
            params + [latency_sample_cap],
        ).fetchall()
        latencies = [r["latency_s"] for r in lat_rows]

        by_cluster_rows = conn.execute(
            f"SELECT COALESCE(NULLIF(cluster, ''), 'unknown') AS c, COUNT(*) AS n "
            f"FROM queries {where} GROUP BY c ORDER BY n DESC",
            params,
        ).fetchall()

        by_service_rows = conn.execute(
            f"SELECT service, COUNT(*) AS n FROM queries {where} "
            f"GROUP BY service ORDER BY n DESC",
            params,
        ).fetchall()

        by_tool_rows = conn.execute(
            f"SELECT COALESCE(tool, '(non-tool)') AS t, COUNT(*) AS n "
            f"FROM queries {where} GROUP BY t ORDER BY n DESC LIMIT 20",
            params,
        ).fetchall()

        # Last 24 hourly buckets, UTC — matches legacy StatsTracker format
        # (`"YYYY-MM-DD HH:00 UTC"`) so the existing dashboard renders unchanged.
        by_hour_rows = conn.execute(
            f"SELECT strftime('%Y-%m-%d %H:00 UTC', ts) AS h, COUNT(*) AS n "
            f"FROM queries {where} GROUP BY h ORDER BY h DESC LIMIT 24",
            params,
        ).fetchall()

    cache_total = cache_hits + cache_misses
    hit_rate = round(cache_hits / cache_total, 3) if cache_total > 0 else 0.0
    avg_lat = round(sum(latencies) / len(latencies), 3) if latencies else 0.0

    return {
        "total_queries": total,
        "cache": {
            "hits":     cache_hits,
            "misses":   cache_misses,
            "hit_rate": hit_rate,
        },
        "errors": errors,
        "latency": {
            "avg_s":         avg_lat,
            "p50_s":         _percentile(latencies, 50),
            "p95_s":         _percentile(latencies, 95),
            "p99_s":         _percentile(latencies, 99),
            "sample_count":  len(latencies),
        },
        "queries_by_cluster": {r["c"]: r["n"] for r in by_cluster_rows},
        "queries_by_service": {r["service"]: r["n"] for r in by_service_rows},
        "queries_by_tool":    {r["t"]: r["n"] for r in by_tool_rows},
        "queries_by_hour":    {r["h"]: r["n"] for r in reversed(by_hour_rows)},
    }


def query_feedback_aggregates(
    db_path: str,
    services: Optional[list[str]] = None,
    scopes: Optional[list[str]] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
) -> dict:
    """Feedback aggregates matching legacy shape."""
    where, params = _where_clause(services, scopes, since, until)

    with closing(connect(db_path)) as conn:
        total = conn.execute(
            f"SELECT COUNT(*) AS c FROM feedback {where}", params
        ).fetchone()["c"]
        up = conn.execute(
            f"SELECT COUNT(*) AS c FROM feedback {where} "
            + ("AND" if where else "WHERE") + " rating > 0",
            params,
        ).fetchone()["c"]
        down = conn.execute(
            f"SELECT COUNT(*) AS c FROM feedback {where} "
            + ("AND" if where else "WHERE") + " rating < 0",
            params,
        ).fetchone()["c"]
        by_cluster = conn.execute(
            f"SELECT COALESCE(NULLIF(cluster, ''), 'unknown') AS c, "
            f"       SUM(CASE WHEN rating > 0 THEN 1 ELSE 0 END) AS up, "
            f"       SUM(CASE WHEN rating < 0 THEN 1 ELSE 0 END) AS down "
            f"FROM feedback {where} GROUP BY c",
            params,
        ).fetchall()

    return {
        "total":        total,
        "satisfaction": round(up / max(total, 1), 3),
        "up":           up,
        "down":         down,
        "by_cluster":   {r["c"]: {"up": r["up"], "down": r["down"]} for r in by_cluster},
    }
