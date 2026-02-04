"""
Semantic Response Cache for RAG Chatbot

Caches full query responses with semantic similarity matching.
Uses sentence embeddings to match similar questions.
"""
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Optional, Dict, Any
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
        self.db_path = Path(db_path)
        self.similarity_threshold = similarity_threshold
        
        # Load embedding model (80MB, runs on CPU)
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        logger.info("✓ Embedding model loaded")
        
        # Initialize database
        self._init_db()
    
    def _init_db(self):
        """Create cache table if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS response_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_text TEXT NOT NULL,
                cluster TEXT NOT NULL,
                embedding BLOB NOT NULL,
                response_json TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Index for cluster-based filtering
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_cluster 
            ON response_cache(cluster)
        """)
        
        conn.commit()
        conn.close()
        logger.info(f"✓ Cache database initialized: {self.db_path}")
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text."""
        return self.model.encode(text, normalize_embeddings=True)
    
    def _cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings."""
        return float(np.dot(emb1, emb2))
    
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
    
    def set(self, query: str, cluster: str, response: Dict[str, Any]):
        """
        Cache a response.
        
        Args:
            query: User query text
            cluster: Cluster name
            response: Response dict to cache
        """
        query_embedding = self._get_embedding(query)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            """
            INSERT INTO response_cache (query_text, cluster, embedding, response_json)
            VALUES (?, ?, ?, ?)
            """,
            (
                query,
                cluster,
                query_embedding.tobytes(),
                json.dumps(response)
            )
        )
        
        conn.commit()
        conn.close()
        
        logger.info(f"✓ Cached response for: '{query[:60]}...'")
    
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
