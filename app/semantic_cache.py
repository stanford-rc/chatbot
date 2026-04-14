"""
Semantic Response Cache for RAG Chatbot

Caches full query responses with semantic similarity matching.
Uses sentence embeddings to match similar questions.

Cache invalidation:
    Each cached response stores the source doc filenames that were retrieved
    to answer the query.  On startup, the RAG service compares the current
    content manifest (written by the scraper) against the previous one.
    Any source files whose content hash changed are passed to
    invalidate_by_sources(), which deletes cache entries that depended on
    those files.  Stable content stays cached; only stale answers are evicted.
"""
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Optional, Dict, Any, List, Set
import numpy as np
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)


class SemanticResponseCache:
    """
    Full response cache with semantic similarity matching.

    Caches entire QueryResponse objects and matches similar questions
    using cosine similarity of sentence embeddings.
    """

    def __init__(
        self,
        db_path: str = ".response_cache.db",
        model_name: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.95
    ):
        """
        Initialize semantic cache.

        Args:
            db_path: SQLite database path
            model_name: Sentence transformer model name
            similarity_threshold: Cosine similarity threshold (0-1)
        """
        self.db_path = Path(db_path).resolve()  # absolute so CWD changes can't misdirect connections
        self.similarity_threshold = similarity_threshold

        # Load embedding model on CPU — GPU is reserved entirely for the LLM.
        logger.info(f"Loading embedding model (CPU): {model_name}")
        self.model = SentenceTransformer(model_name, device='cpu', trust_remote_code=True)
        logger.info("✓ Embedding model loaded on CPU")

        # Initialize database
        self._init_db()

    def _init_db(self):
        """Create cache table if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # WAL mode allows concurrent reads from multiple workers without locking
        cursor.execute("PRAGMA journal_mode=WAL;")
        cursor.execute("PRAGMA busy_timeout=5000;")  # Wait up to 5s on write contention

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS response_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_text TEXT NOT NULL,
                cluster TEXT NOT NULL,
                embedding BLOB NOT NULL,
                response_json TEXT NOT NULL,
                source_files TEXT NOT NULL DEFAULT '[]',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Index for cluster-based filtering
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_cluster
            ON response_cache(cluster)
        """)

        # Migrate: add source_files column if missing (existing DBs)
        cursor.execute("PRAGMA table_info(response_cache)")
        columns = {row[1] for row in cursor.fetchall()}
        if 'source_files' not in columns:
            cursor.execute("""
                ALTER TABLE response_cache ADD COLUMN source_files TEXT NOT NULL DEFAULT '[]'
            """)
            logger.info("Migrated response_cache: added source_files column")

        conn.commit()
        conn.close()
        logger.info(f"✓ Cache database initialized: {self.db_path}")

    def _get_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text."""
        return self.model.encode(text, normalize_embeddings=True)

    def _cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings."""
        return float(np.dot(emb1, emb2))

    def _ensure_table(self, conn: sqlite3.Connection) -> None:
        """Create the cache table if it doesn't exist (fail-safe for missing schema)."""
        conn.execute("""
            CREATE TABLE IF NOT EXISTS response_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_text TEXT NOT NULL,
                cluster TEXT NOT NULL,
                embedding BLOB NOT NULL,
                response_json TEXT NOT NULL,
                source_files TEXT NOT NULL DEFAULT '[]',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_cluster ON response_cache(cluster)
        """)
        conn.commit()

    def get(self, query: str, cluster: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached response for similar query.

        Args:
            query: User query text
            cluster: Cluster name

        Returns:
            Cached response dict or None if no match
        """
        query_embedding = self._get_embedding(query)

        conn = sqlite3.connect(self.db_path)
        self._ensure_table(conn)
        cursor = conn.cursor()

        # Get all cached queries for this cluster
        cursor.execute(
            "SELECT id, query_text, embedding, response_json FROM response_cache WHERE cluster = ?",
            (cluster,)
        )

        best_match = None
        best_similarity = 0.0

        for row_id, cached_query, embedding_blob, response_json in cursor.fetchall():
            # Deserialize embedding
            cached_embedding = np.frombuffer(embedding_blob, dtype=np.float32)

            # Calculate similarity
            similarity = self._cosine_similarity(query_embedding, cached_embedding)

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = (row_id, cached_query, response_json)

        conn.close()

        # Check if best match meets threshold
        if best_match and best_similarity >= self.similarity_threshold:
            row_id, cached_query, response_json = best_match
            logger.info(
                f"✓ Cache HIT! Similarity: {best_similarity:.3f} | "
                f"Query: '{query[:50]}...' → Cached: '{cached_query[:50]}...'"
            )
            return json.loads(response_json)
        else:
            if best_match:
                _, cached_query, _ = best_match
                logger.info(
                    f"✗ Cache MISS (low similarity: {best_similarity:.3f}) | "
                    f"Query: '{query[:50]}...' vs '{cached_query[:50]}...'"
                )
            else:
                logger.info(f"✗ Cache MISS (no entries for cluster '{cluster}')")
            return None

    def set(self, query: str, cluster: str, response: Dict[str, Any],
            source_files: Optional[List[str]] = None):
        """
        Cache a response.

        Args:
            query: User query text
            cluster: Cluster name
            response: Response dict to cache
            source_files: List of source doc filenames that contributed to this answer
        """
        query_embedding = self._get_embedding(query)
        sources_json = json.dumps(source_files or [])

        conn = sqlite3.connect(self.db_path)
        self._ensure_table(conn)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO response_cache (query_text, cluster, embedding, response_json, source_files)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                query,
                cluster,
                query_embedding.tobytes(),
                json.dumps(response),
                sources_json,
            )
        )

        conn.commit()
        conn.close()

        logger.info(f"✓ Cached response for: '{query[:60]}...' (sources: {len(source_files or [])} docs)")

    def invalidate_by_sources(self, changed_files: Set[str]) -> int:
        """Delete cache entries that depend on any of the changed source files.

        Args:
            changed_files: Set of source filenames whose content changed.

        Returns:
            Number of cache entries invalidated.
        """
        if not changed_files:
            return 0

        conn = sqlite3.connect(self.db_path)
        self._ensure_table(conn)
        cursor = conn.cursor()

        cursor.execute("SELECT id, source_files FROM response_cache")
        ids_to_delete = []
        for row_id, sources_json in cursor.fetchall():
            try:
                sources = set(json.loads(sources_json))
            except (json.JSONDecodeError, TypeError):
                sources = set()
            if sources & changed_files:
                ids_to_delete.append(row_id)

        if ids_to_delete:
            placeholders = ",".join("?" * len(ids_to_delete))
            cursor.execute(
                f"DELETE FROM response_cache WHERE id IN ({placeholders})",
                ids_to_delete,
            )
            conn.commit()
            logger.info(
                f"✓ Invalidated {len(ids_to_delete)} cache entries for changed sources: "
                f"{sorted(changed_files)[:5]}{'...' if len(changed_files) > 5 else ''}"
            )
        else:
            logger.info(f"No cache entries to invalidate for {len(changed_files)} changed files")

        conn.close()
        return len(ids_to_delete)

    def clear(self, cluster: Optional[str] = None):
        """Clear cache for a cluster or entire cache."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if cluster:
            cursor.execute("DELETE FROM response_cache WHERE cluster = ?", (cluster,))
            logger.info(f"✓ Cleared cache for cluster: {cluster}")
        else:
            cursor.execute("DELETE FROM response_cache")
            logger.info("✓ Cleared entire cache")

        conn.commit()
        conn.close()

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM response_cache")
        total = cursor.fetchone()[0]

        cursor.execute(
            "SELECT cluster, COUNT(*) FROM response_cache GROUP BY cluster"
        )
        by_cluster = dict(cursor.fetchall())

        conn.close()

        return {
            "total_entries": total,
            "by_cluster": by_cluster,
            "similarity_threshold": self.similarity_threshold
        }
