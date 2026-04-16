import hashlib
import json
import logging
import os
import pickle
import re
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import frontmatter
import numpy as np
from fastapi import HTTPException
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_community.cache import SQLiteCache
from langchain_community.retrievers import BM25Retriever
from langchain_core.globals import set_llm_cache
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from vllm import LLM, SamplingParams

from app.config import Settings
from app.models import QueryRequest, QueryResponse, Source
from app.prompts import get_prompt_template
from app.stats import stats_tracker

# Semantic caching - optional dependency
try:
    from app.semantic_cache import SemanticResponseCache
    SEMANTIC_CACHE_AVAILABLE = True
except ImportError:
    SEMANTIC_CACHE_AVAILABLE = False

# FAISS vector search - optional dependency
try:
    import faiss
    from sentence_transformers import SentenceTransformer
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

logger = logging.getLogger(__name__)


class RAGService:
    """
    Retrieval-Augmented Generation service for querying documentation.
    Handles model loading, document retrieval, and response generation.
    """
    
    def __init__(self, config: Settings):
        self.settings = config
        self.llm = None
        self.model = None
        self.tokenizer = None
        self.retrievers: Dict[str, BM25Retriever] = {}
        self.vector_stores: Dict[str, dict] = {}  # cluster -> {index, docs, model}
        self.chain = None
        self.semantic_cache = None
        self.embedding_model = None

    def _ingest_markdown_files(self, corpus_dir: str) -> List[Document]:
        """
        Load markdown files from directory with frontmatter metadata.
        
        Args:
            corpus_dir: Path to directory containing .md files
            
        Returns:
            List of Document objects with content and metadata
        """
        documents = []
        for filename in os.listdir(corpus_dir):
            if filename.endswith(".md"):
                file_path = os.path.join(corpus_dir, filename)
                try:
                    post = frontmatter.load(file_path)
                    metadata = post.metadata
                    metadata['source'] = filename
                    # Prepend title to body so chunks inherit it for BM25/FAISS.
                    # Without this, a doc titled "Classes and Workshops" whose body
                    # says "Upcoming Classes" is invisible to "workshop" queries.
                    title = metadata.get('title', '')
                    body = f"# {title}\n\n{post.content}" if title else post.content
                    doc = Document(page_content=body, metadata=metadata)
                    documents.append(doc)
                    logger.info(f"Loaded document from {file_path}")
                except Exception as e:
                    logger.warning(f"Could not read or parse front matter from file {file_path}: {e}")
        return documents

    def _split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents using header-aware strategy.

        1. Small docs (≤ chunk_size): kept whole.
        2. Docs with ## headers: split on ## boundaries so each section is a
           self-contained chunk.  The doc title prefix (# Title) is prepended
           to every section chunk so BM25/FAISS can match on it.
        3. Docs without ## headers: fall back to character splitting.
        4. Any single section that exceeds chunk_size is sub-split with
           RecursiveCharacterTextSplitter.
        """
        chunk_size = self.settings.CHUNK_SIZE
        chunk_overlap = self.settings.CHUNK_OVERLAP

        md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("##", "section")],
            strip_headers=False,
        )
        char_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        all_chunks: List[Document] = []
        for doc in documents:
            content = doc.page_content

            # Small docs: no splitting needed
            if len(content) <= chunk_size:
                all_chunks.append(doc)
                continue

            # Try header-based splitting
            if "\n## " in content:
                # Extract the title prefix (everything before first ##)
                first_h2 = content.find("\n## ")
                title_prefix = content[:first_h2].strip() + "\n\n" if first_h2 > 0 else ""

                md_chunks = md_splitter.split_text(content)

                for md_chunk in md_chunks:
                    section_header = md_chunk.metadata.get("section", "")
                    # The splitter rolls # Title into the first ## chunk.
                    # Only prepend title_prefix to subsequent chunks that
                    # don't already contain it.
                    chunk_text = md_chunk.page_content
                    if title_prefix and not chunk_text.startswith(title_prefix.strip()):
                        chunk_text = title_prefix + chunk_text

                    chunk_metadata = {**doc.metadata, "section_header": section_header}

                    if len(chunk_text) <= chunk_size:
                        all_chunks.append(Document(
                            page_content=chunk_text,
                            metadata=chunk_metadata,
                        ))
                    else:
                        # Oversized section: sub-split with character splitter
                        sub_doc = Document(page_content=chunk_text, metadata=chunk_metadata)
                        all_chunks.extend(char_splitter.split_documents([sub_doc]))
            else:
                # No ## headers: fall back to character splitting
                all_chunks.extend(char_splitter.split_documents([doc]))

        return all_chunks

    def _setup_caching(self):
        """Initialize LangChain and semantic caching"""
        try:
            set_llm_cache(SQLiteCache(database_path=self.settings.LANGCHAIN_CACHE_DB))
            logger.info(f"LangChain LLM cache enabled: {self.settings.LANGCHAIN_CACHE_DB}")
        except Exception as e:
            logger.error(f"Could not set up LLM cache: {e}")
        
        # Initialize semantic response cache
        if SEMANTIC_CACHE_AVAILABLE and self.settings.SEMANTIC_CACHE_ENABLED:
            try:
                self.semantic_cache = SemanticResponseCache(
                    db_path=self.settings.SEMANTIC_CACHE_DB,
                    model_name=self.settings.VECTOR_MODEL,
                    similarity_threshold=self.settings.SEMANTIC_CACHE_THRESHOLD
                )
                if self.settings.SEMANTIC_CACHE_CLEAR_ON_STARTUP:
                    self.semantic_cache.clear()
                    logger.info("Semantic cache cleared on startup (SEMANTIC_CACHE_CLEAR_ON_STARTUP=true)")
                else:
                    # Selective invalidation: evict only cache entries whose
                    # source docs changed since the last scrape.
                    self._invalidate_stale_cache_entries()
                logger.info(f"✓ Semantic cache initialized (threshold: {self.settings.SEMANTIC_CACHE_THRESHOLD})")
            except Exception as e:
                logger.error(f"Failed to initialize semantic cache: {e}")
                self.semantic_cache = None
        else:
            logger.warning("Semantic cache disabled")
            self.semantic_cache = None

    def _detect_changed_sources(self) -> Set[str]:
        """Compare content manifests to find source files that changed.

        Each docs directory may contain a .content_manifest.json written by the
        scraper.  A separate .content_manifest.prev.json stores the manifest
        from the previous startup.  Files whose hash differs (or was added/
        removed) are returned as changed.
        """
        changed: Set[str] = set()

        # Collect all doc directories (shared + per-cluster)
        doc_dirs: List[Path] = []
        if self.settings.SHARED_DOCS_PATH:
            doc_dirs.append(Path(self.settings.SHARED_DOCS_PATH))
        for _cluster, path in self.settings.CLUSTERS.items():
            doc_dirs.append(Path(path))

        for doc_dir in doc_dirs:
            manifest_path = doc_dir / ".content_manifest.json"
            prev_path = doc_dir / ".content_manifest.prev.json"

            if not manifest_path.exists():
                continue

            # Load current manifest
            try:
                current = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Could not read manifest {manifest_path}: {e}")
                continue

            # Load previous manifest
            previous: dict = {}
            if prev_path.exists():
                try:
                    previous = json.loads(prev_path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    pass  # treat as empty — all files are "new"

            # Detect changes: new, modified, or deleted files
            all_files = set(current.keys()) | set(previous.keys())
            for filename in all_files:
                if current.get(filename) != previous.get(filename):
                    changed.add(filename)

            # Rotate: current manifest becomes prev for next startup
            try:
                prev_path.write_text(
                    json.dumps(current, indent=2), encoding="utf-8"
                )
            except OSError as e:
                logger.warning(f"Could not write previous manifest {prev_path}: {e}")

        return changed

    def _invalidate_stale_cache_entries(self):
        """Detect changed source docs and evict their cached answers."""
        if not self.semantic_cache:
            return

        try:
            changed = self._detect_changed_sources()
            if changed:
                count = self.semantic_cache.invalidate_by_sources(changed)
                logger.info(
                    f"Content-aware cache invalidation: {len(changed)} source(s) changed, "
                    f"{count} cache entry/entries evicted"
                )
            else:
                logger.info("Content-aware cache invalidation: no source changes detected")
        except Exception as e:
            logger.error(f"Cache invalidation failed (non-fatal): {e}")

    def _load_model(self):
        """Load the language model via vLLM with tensor parallelism across both GPUs."""
        logger.info(f"Loading model with vLLM from {self.settings.MODEL_PATH}...")

        # Tensor parallel across both L4 GPUs.  Both must be visible to vLLM.
        # Override any per-worker CUDA_VISIBLE_DEVICES restriction — TP=2 requires
        # both GPUs in a single process; the old two-worker nginx architecture is
        # replaced by vLLM's native async scheduler and continuous batching.
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
        logger.info("Tensor parallel mode: exposing both GPUs (CUDA_VISIBLE_DEVICES=0,1)")

        # vLLM's internal RPC timeout between EngineCore and GPU WorkerProcs.
        # setdefault so an explicit env override from the launch environment wins.
        os.environ.setdefault('VLLM_RPC_TIMEOUT', '300000')          # 5 min (ms)
        os.environ.setdefault('VLLM_WORKER_MULTIPROC_TIMEOUT', '600') # 10 min (s)

        # ── NCCL transport override ──────────────────────────────────────────
        # L4 GPUs are PCIe-only (no NVLink).  Inside Apptainer, NCCL's normal
        # transport selection hangs:
        #   1. P2P (PCIe peer-mem)  — blocked by IOMMU / cgroup inside container
        #   2. SHM (shared-memory)  — also hangs when NCCL's shm region name
        #                            collides with the container's mount namespace
        # Disabling both forces NCCL to use pure TCP socket on the loopback
        # interface.  Loopback bandwidth ≈ 10-20 GB/s; the TP all-reduce for
        # Qwen 32B at seq_len=512 is ~5 MB/layer × 64 layers ≈ ~320 MB total
        # → ~32 ms extra latency per forward pass.  Acceptable for a chatbot.
        # Use direct assignment (not setdefault) so these always win over any
        # stale env var inherited from the shell.
        os.environ['NCCL_P2P_DISABLE'] = '1'     # disable PCIe peer-mem
        os.environ['NCCL_SHM_DISABLE'] = '1'     # disable shared-mem transport
        os.environ['NCCL_SOCKET_IFNAME'] = 'lo'  # force loopback socket
        os.environ['NCCL_DEBUG'] = 'WARN'        # surface NCCL errors in log

        self.model = LLM(
            model=self.settings.MODEL_PATH,
            # Use AWQ kernels directly (no online AWQ→Marlin conversion).
            quantization="awq",
            dtype=self.settings.MODEL_DTYPE,
            # Tensor parallel across 2× NVIDIA L4 (22.5 GiB each).
            # Model shard per GPU: ~9 GiB.  KV budget: ~9.6 GiB/GPU (19x concurrency).
            tensor_parallel_size=2,
            gpu_memory_utilization=0.90,
            max_model_len=4096,
            # ── Critical Apptainer workarounds ──────────────────────────────
            # 1. enforce_eager: disable CUDA graph capture to prevent
            #    EngineCore→WorkerProc RPC timeout during graph compilation.
            enforce_eager=True,
            # 2. disable_custom_all_reduce: vLLM's fast custom all-reduce has
            #    two implementations:
            #      a. SymmMemCommunicator (symmetric memory) — NOT supported on
            #         sm_89 (L4); vLLM logs "Device capability 8.9 not supported".
            #      b. CustomAllreduce (legacy) — uses CUDA IPC memory handles to
            #         share GPU buffers across processes.  CUDA IPC is blocked by
            #         Apptainer's container security model, causing the post-init
            #         warmup forward pass to hang indefinitely (visible as repeated
            #         "No available shared memory broadcast block" log lines).
            #    Setting True forces vLLM to use standard NCCL all-reduce instead,
            #    which uses socket transport (NCCL_P2P_DISABLE + NCCL_SHM_DISABLE
            #    set above) and works correctly inside Apptainer.
            disable_custom_all_reduce=True,
            disable_log_stats=True,
        )
        self.tokenizer = self.model.get_tokenizer()
        self.llm = RunnableLambda(self._generate_response)
        logger.info("Model loaded successfully via vLLM.")

    # ── Index disk cache ──────────────────────────────────────────────────────

    def _index_cache_dir(self) -> Path:
        """Derive cache dir from SEMANTIC_CACHE_DB location — no extra config needed."""
        d = Path(self.settings.SEMANTIC_CACHE_DB).parent / 'indices'
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _index_fingerprint(self, cluster_path: str, shared_docs_path: str) -> str:
        """SHA-256 over all source .md content + chunking + embedding model.

        Any change to docs, chunk size, or model invalidates the cache for
        that cluster.
        """
        h = hashlib.sha256()
        for base in sorted([p for p in [cluster_path, shared_docs_path] if p and os.path.isdir(p)]):
            for md in sorted(Path(base).glob('**/*.md')):
                h.update(str(md).encode())
                h.update(md.read_bytes())
        h.update(str(self.settings.CHUNK_SIZE).encode())
        h.update(str(self.settings.CHUNK_OVERLAP).encode())
        h.update(str(self.settings.VECTOR_MODEL).encode())
        return h.hexdigest()

    def _try_load_index_cache(
        self, cluster_name: str, fingerprint: str
    ) -> Optional[tuple]:
        """Return (bm25, faiss_index_or_None, split_docs) if cache is valid, else None."""
        d = self._index_cache_dir()
        meta_path = d / f'{cluster_name}.meta.json'
        bm25_path = d / f'{cluster_name}.bm25.pkl'
        docs_path = d / f'{cluster_name}.docs.pkl'
        faiss_path = d / f'{cluster_name}.faiss'

        if not (meta_path.exists() and bm25_path.exists() and docs_path.exists()):
            return None
        meta = json.loads(meta_path.read_text())
        if meta.get('fingerprint') != fingerprint:
            logger.info(f"Index cache stale for '{cluster_name}' — rebuilding.")
            return None

        try:
            split_docs  = pickle.loads(docs_path.read_bytes())
            bm25        = pickle.loads(bm25_path.read_bytes())
            faiss_index = None
            if FAISS_AVAILABLE and self.embedding_model is not None and faiss_path.exists():
                faiss_index = faiss.read_index(str(faiss_path))
            logger.info(f"Loaded '{cluster_name}' index from disk cache ({len(split_docs)} chunks).")
            return bm25, faiss_index, split_docs
        except Exception as e:
            logger.warning(f"Failed to load index cache for '{cluster_name}' ({e}) — rebuilding.")
            return None

    def _save_index_cache(
        self, cluster_name: str, fingerprint: str,
        bm25, faiss_index, split_docs: list
    ) -> None:
        """Persist BM25 + FAISS index to disk for fast reloads."""
        d = self._index_cache_dir()
        (d / f'{cluster_name}.docs.pkl').write_bytes(pickle.dumps(split_docs))
        (d / f'{cluster_name}.bm25.pkl').write_bytes(pickle.dumps(bm25))
        if faiss_index is not None and FAISS_AVAILABLE:
            faiss.write_index(faiss_index, str(d / f'{cluster_name}.faiss'))
        (d / f'{cluster_name}.meta.json').write_text(json.dumps({
            'fingerprint': fingerprint,
            'cluster':     cluster_name,
            'num_chunks':  len(split_docs),
            'created':     time.time(),
        }))
        logger.info(f"Saved '{cluster_name}' index to disk cache.")

    def _load_retrievers(self):
        """Load BM25 and (optionally) FAISS retrievers for each cluster"""

        # Load embedding model once if hybrid retrieval is enabled
        if self.settings.HYBRID_ENABLED and FAISS_AVAILABLE:
            logger.info(f"Loading embedding model for hybrid retrieval (CPU): {self.settings.VECTOR_MODEL}")
            self.embedding_model = SentenceTransformer(self.settings.VECTOR_MODEL, device='cpu', trust_remote_code=True)
            logger.info("Embedding model loaded on CPU.")
            stats_tracker.set_embedding_model(self.embedding_model)
        elif self.settings.HYBRID_ENABLED and not FAISS_AVAILABLE:
            logger.warning("Hybrid retrieval enabled but faiss-cpu not installed. Falling back to BM25 only.")

        # Load shared SRCC web docs once — merged into every cluster's retriever
        shared_docs = []
        if self.settings.SHARED_DOCS_PATH and os.path.isdir(self.settings.SHARED_DOCS_PATH):
            shared_docs = self._ingest_markdown_files(self.settings.SHARED_DOCS_PATH)
            logger.info(f"Loaded {len(shared_docs)} shared SRCC web documents from {self.settings.SHARED_DOCS_PATH}")
        elif self.settings.SHARED_DOCS_PATH:
            logger.warning(f"Shared docs path not found: {self.settings.SHARED_DOCS_PATH}. Skipping.")

        for cluster_name, path in self.settings.CLUSTERS.items():
            if not os.path.isdir(path):
                logger.warning(f"Directory not found for cluster '{cluster_name}': {path}. Skipping.")
                continue

            # ── Try disk cache first ──────────────────────────────────────────
            fingerprint = self._index_fingerprint(path, self.settings.SHARED_DOCS_PATH)
            cached = self._try_load_index_cache(cluster_name, fingerprint)
            if cached:
                bm25, faiss_index, split_docs = cached
                self.retrievers[cluster_name] = bm25
                if faiss_index is not None:
                    self.vector_stores[cluster_name] = {"index": faiss_index, "docs": split_docs}
                continue

            # ── Cache miss: build from scratch ───────────────────────────────
            logger.info(f"Ingesting documents for cluster: {cluster_name}")
            documents = self._ingest_markdown_files(path)
            if not documents:
                logger.warning(f"No documents found for cluster '{cluster_name}'.")
                continue

            # Merge cluster-specific docs with shared SRCC web content.
            # Relevancy among shared docs is handled naturally by BM25/FAISS
            # scoring — a Sherlock query will rank Sherlock-relevant shared docs
            # higher without any explicit filtering needed.
            all_documents = documents + shared_docs
            split_docs = self._split_documents(all_documents)
            bm25 = BM25Retriever.from_documents(split_docs)
            self.retrievers[cluster_name] = bm25
            logger.info(f"Created BM25 retriever for '{cluster_name}' ({len(split_docs)} chunks).")

            # Build FAISS index for this cluster
            faiss_index = None
            if self.embedding_model is not None:
                embeddings = self.embedding_model.encode(
                    [doc.page_content for doc in split_docs],
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                dimension = embeddings.shape[1]
                faiss_index = faiss.IndexFlatIP(dimension)  # Inner product on normalized vectors = cosine similarity
                faiss_index.add(embeddings.astype(np.float32))
                self.vector_stores[cluster_name] = {
                    "index": faiss_index,
                    "docs": split_docs,
                }
                logger.info(f"Created FAISS index for '{cluster_name}' ({len(split_docs)} vectors, dim={dimension}).")

            self._save_index_cache(cluster_name, fingerprint, bm25, faiss_index, split_docs)

    def _retrieve_bm25_with_scores(self, query: str, cluster: str) -> List[Tuple[Document, float]]:
        """Retrieve documents via BM25 with relevance scores."""
        retriever = self.retrievers[cluster]
        # Access the underlying BM25 vectorizer to get scores
        tokenized_query = retriever.preprocess_func(query)
        bm25_scores = retriever.vectorizer.get_scores(tokenized_query)

        # Pair each doc with its score and sort descending
        scored_docs = list(zip(retriever.docs, bm25_scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Log top scores for tuning visibility
        top_scores = scored_docs[:self.settings.MAX_RETRIEVED_DOCS]
        for doc, score in top_scores:
            title = doc.metadata.get('title', doc.metadata.get('source', 'Unknown'))
            logger.info(f"BM25 score={score:.2f} for '{title}'")

        # Filter by minimum score threshold
        min_score = self.settings.MIN_BM25_SCORE
        filtered = [(doc, score) for doc, score in scored_docs if score >= min_score]
        logger.info(f"BM25 filtering: {len(filtered)} of {len(scored_docs)} docs above threshold {min_score}")
        return filtered[:self.settings.MAX_RETRIEVED_DOCS]

    def _retrieve_faiss(self, query: str, cluster: str) -> List[Tuple[Document, float]]:
        """Retrieve documents via FAISS vector similarity."""
        store = self.vector_stores.get(cluster)
        if not store or self.embedding_model is None:
            return []

        query_embedding = self.embedding_model.encode(
            [query], normalize_embeddings=True
        ).astype(np.float32)

        k = min(self.settings.MAX_RETRIEVED_DOCS * 3, store["index"].ntotal)
        scores, indices = store["index"].search(query_embedding, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:  # FAISS returns -1 for missing results
                results.append((store["docs"][idx], float(score)))

        # Log top FAISS results (matching BM25 logging pattern)
        for doc, score in results[:self.settings.MAX_RETRIEVED_DOCS]:
            title = doc.metadata.get('title', doc.metadata.get('source', 'Unknown'))
            logger.info(f"FAISS score={score:.4f} for '{title}'")

        return results

    def _reciprocal_rank_fusion(
        self,
        bm25_results: List[Tuple[Document, float]],
        faiss_results: List[Tuple[Document, float]],
    ) -> List[Document]:
        """Merge two ranked lists using reciprocal rank fusion (RRF).

        FAISS (semantic) results are weighted by FAISS_RRF_WEIGHT so that
        semantic similarity has more influence than raw keyword frequency.
        """
        k = self.settings.RRF_K
        faiss_weight = self.settings.FAISS_RRF_WEIGHT
        # Use page_content as dedup key since the same chunk can appear in both lists
        rrf_scores: Dict[str, float] = defaultdict(float)
        doc_map: Dict[str, Document] = {}

        for rank, (doc, _score) in enumerate(bm25_results):
            key = doc.page_content
            rrf_scores[key] += 1.0 / (k + rank + 1)
            doc_map[key] = doc

        for rank, (doc, _score) in enumerate(faiss_results):
            key = doc.page_content
            rrf_scores[key] += faiss_weight / (k + rank + 1)
            doc_map[key] = doc

        # Sort by fused score descending
        sorted_keys = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        return [doc_map[key] for key in sorted_keys[:self.settings.MAX_RETRIEVED_DOCS]]

    def _build_chain(self):
        """Build the RAG chain with retriever, prompt, and LLM"""
        prompt_template = get_prompt_template(self.settings.MODEL_TYPE)

        def retrieve_and_format_context(inputs: Dict) -> str:
            query, cluster = inputs['query'], inputs['cluster']

            # Prepend the cluster name to the BM25 query so keyword scoring
            # naturally boosts cluster-relevant docs. FAISS uses the raw query
            # so that semantic search isn't biased toward cluster-specific content
            # when the user is asking about cross-cluster topics (e.g. workshops).
            bm25_query = f"{cluster} {query}"

            # BM25 retrieval with score filtering
            bm25_results = self._retrieve_bm25_with_scores(bm25_query, cluster)
            logger.info(f"BM25 returned {len(bm25_results)} docs above threshold")

            # Hybrid: merge BM25 + FAISS via RRF
            if self.settings.HYBRID_ENABLED and cluster in self.vector_stores:
                faiss_results = self._retrieve_faiss(query, cluster)
                logger.info(f"FAISS returned {len(faiss_results)} docs")
                retrieved_docs = self._reciprocal_rank_fusion(bm25_results, faiss_results)
            else:
                retrieved_docs = [doc for doc, _score in bm25_results]

            inputs['retrieved_docs'] = retrieved_docs

            if not retrieved_docs:
                return "No relevant documents were found for this query."

            def _clean_content(text: str) -> str:
                # Sherlock/Farmshare docs use Markdown reference variables such as
                # [Scheduling on Sherlock][url_scheduling] and [url_xxx]: https://...
                # Strip them so the model sees plain prose and cannot copy anchor
                # text as a citation title — it must use the document header instead.
                text = re.sub(r'\[([^\]]+)\]\[[^\]]*\]', r'\1', text)      # [text][ref] → text
                text = re.sub(r'\[url_[^\]]+\](?::[^\n]*)?\n?', '', text)  # [url_xxx]: lines
                return text

            titles = [
                doc.metadata.get('title', doc.metadata.get('source', 'Unknown'))
                for doc in retrieved_docs
            ]

            # Build document labels: include section header when present
            # so the LLM sees "Classes and Workshops > Upcoming Classes"
            def _doc_label(doc, title):
                section = doc.metadata.get('section_header', '')
                return f"{title} > {section}" if section else title

            # Lead with an explicit title list so the model knows the exact
            # strings it must use when citing — prevents invented titles like
            # "Scheduling on Sherlock" that don't match any retrieved document.
            # Citation list uses bare titles only (not section labels).
            title_list = "\n".join(f"  - {t}" for t in titles)
            docs_text = "\n\n".join(
                f"--- Document: {_doc_label(doc, title)} ---\n{_clean_content(doc.page_content)}"
                for title, doc in zip(titles, retrieved_docs)
            )
            return (
                f"AVAILABLE SOURCES — cite using [Title] with ONLY these exact titles:\n"
                f"{title_list}\n\n"
                f"{docs_text}"
            )

        self.chain = (
            RunnablePassthrough()
            | RunnablePassthrough.assign(context=RunnableLambda(retrieve_and_format_context))
            | prompt_template
            | self.llm
            | StrOutputParser()
        )

    def initialize(self):
        """Initialize the RAG service - load models, retrievers, and build chain"""
        logger.info("Initializing RAG Service...")
        
        try:
            self._setup_caching()
            self._load_model()
            self._load_retrievers()
            self._build_chain()
            logger.info("RAG service initialization complete.")
        except Exception as e:
            logger.critical(f"FATAL: Could not initialize RAG service. Error: {e}")
            raise

    def _generate_response(self, inputs: Dict) -> str:
        """
        Generate a response using vLLM.

        Args:
            inputs: Dictionary containing the prompt/query from LangChain

        Returns:
            Generated text response
        """
        try:
            # Build a properly-roled message list for vLLM.chat().
            # LangChain message types: SystemMessage → "system",
            # HumanMessage → "user", AIMessage → "assistant".
            # Passing a single merged user-message caused Qwen to ignore the
            # system-level citation instructions; proper roles fix that.
            _ROLE_MAP = {"system": "system", "human": "user", "ai": "assistant"}
            if isinstance(inputs, dict) and 'messages' in inputs:
                messages = [
                    {"role": _ROLE_MAP.get(getattr(m, "type", "human"), "user"),
                     "content": m.content}
                    for m in inputs['messages']
                ]
            else:
                messages = [{"role": "user", "content": str(inputs)}]

            sampling_params = SamplingParams(
                max_tokens=self.settings.MAX_NEW_TOKENS,
                temperature=0.0,  # greedy decoding
            )

            # vLLM's chat() applies the model's own chat template automatically.
            # enable_thinking=False disables Qwen3's chain-of-thought mode, which
            # is on by default and would wrap every response in <think>...</think>.
            logger.info("Starting vLLM generation...")
            outputs = self.model.chat(
                messages=messages,
                sampling_params=sampling_params,
                use_tqdm=False,
                chat_template_kwargs={"enable_thinking": False},
            )
            response = outputs[0].outputs[0].text
            logger.info(f"Generation complete. Tokens generated: {len(outputs[0].outputs[0].token_ids)}")
            return response

        except Exception as e:
            logger.error(f"Error during vLLM generation: {e}")
            raise ValueError(f"LLM generation error: {e}")


    def _format_sources(self, documents: List[Document]) -> List[Source]:
        """
        Format documents into Source objects with titles and URLs.
        
        Args:
            documents: List of retrieved documents
            
        Returns:
            List of Source objects with deduplicated titles
        """
        sources = []
        seen_titles = set()
        
        for doc in documents:
            title = doc.metadata.get('title', 'Unknown')
            url = doc.metadata.get('url', None)
            
            # Avoid duplicate sources with same title
            if title not in seen_titles:
                sources.append(Source(title=title, url=url))
                seen_titles.add(title)
        
        return sources

    # Short conversational/identity responses that should never trigger grounding
    _CONVERSATIONAL_PATTERNS = re.compile(
        r'\b(my name is|i\'m ada|i am ada|you\'re welcome|you are welcome'
        r'|happy to help|how can i (assist|help)|glad to help'
        r'|let me know if you|is there anything else|feel free to ask)\b',
        re.IGNORECASE,
    )

    # Keywords that suggest the answer is about cluster-specific configuration
    # (Slurm commands, partitions, storage paths, etc.) where ungrounded answers
    # could be harmful.  General SRC/organizational terms (workshops, classes,
    # events, people) should NOT trigger the disclaimer.
    _CLUSTER_SPECIFIC_PATTERNS = re.compile(
        r'\b(?:partition|sbatch|srun|squeue|scancel|salloc|sinfo|scontrol'
        r'|module\s+load|module\s+avail|scratch|oak'
        r'|slurm|quota|storage\s+(?:limit|quota|policy)|gpu\s+partition|memory\s+limit'
        r'|job\s+submit|batch\s+script|queue)\b',
        re.IGNORECASE,
    )

    def _check_grounding(self, answer: str, cited_titles: set, retrieved_titles: set) -> str:
        """
        If the answer discusses cluster-specific topics but cites no retrieved
        documents, append a disclaimer.
        """
        if not self.settings.GROUNDING_CHECK_ENABLED:
            return answer

        # If the model cited at least one retrieved doc, it's grounded
        if cited_titles & retrieved_titles:
            return answer

        # Skip grounding check for short conversational/identity responses
        if self._CONVERSATIONAL_PATTERNS.search(answer):
            return answer

        # Skip if answer already contains the support email — it's already a
        # handled refusal or redirect and doesn't need a second disclaimer
        if 'srcc-support@stanford.edu' in answer:
            return answer

        # Check whether the answer touches cluster-specific topics
        if self._CLUSTER_SPECIFIC_PATTERNS.search(answer):
            logger.warning("Grounding check: answer discusses cluster topics but cites no retrieved docs")
            return answer + f"\n\n*{self.settings.REFUSAL_DISCLAIMER}*"

        return answer

    def _process_llm_answer(self, llm_answer: str, retrieved_docs: List[Document]) -> tuple[str, List[Source]]:
        """
        Process LLM answer to extract citations and format sources.

        Args:
            llm_answer: Raw answer from LLM
            retrieved_docs: Documents that were retrieved for context

        Returns:
            Tuple of (formatted_answer, source_objects)
        """
        # Safety net: strip Qwen3 thinking blocks if enable_thinking=False was
        # ignored or if an older vLLM version doesn't support chat_template_kwargs.
        llm_answer = re.sub(r'<think>.*?</think>\s*', '', llm_answer, flags=re.DOTALL).strip()
        # Some models output the literal two-character sequence \n instead of
        # a real newline. Normalize those to actual newlines regardless of model.
        llm_answer = llm_answer.replace('\\n', '\n')
        # Strip code block fences the model sometimes wraps its entire response in
        llm_answer = re.sub(r'^[\s]*```[^\n]*\n', '', llm_answer)
        llm_answer = re.sub(r'\n```[\s]*$', '', llm_answer)
        llm_answer = llm_answer.strip()

        # ── Normalise citation syntax ────────────────────────────────────────
        # Sherlock/Farmshare docs use Markdown reference-link variables such as
        #   [url_scheduling], [url_storage], etc.
        # The model sometimes inherits this style, writing:
        #   "refer to [Scheduling on Sherlock][url_scheduling]"
        # Step 1: collapse reference notation [text][ref_id] → [text]
        llm_answer = re.sub(r'\[([^\]]+)\]\[[^\]]*\]', r'[\1]', llm_answer)
        # Step 2: strip bare url-variable references left over: [url_xxx]
        llm_answer = re.sub(r'\[url_[^\]]+\]', '', llm_answer)

        # Extract citations (format: [Title])
        cited_titles = set(re.findall(r'\[([^\]]+)\]', llm_answer))
        logger.info(f"LLM cited {len(cited_titles)} sources: {cited_titles}")

        # Build title → URL + doc mappings (case-insensitive key for matching)
        title_to_url: dict = {}
        title_to_doc: dict = {}
        title_lower_map: dict = {}   # lowercase canonical title → original title
        for doc in retrieved_docs:
            title = doc.metadata.get('title', 'Unknown')
            url = doc.metadata.get('url', None)
            if url:
                url = ''.join(url.split())  # strip embedded whitespace/newlines
            title_to_url[title] = url
            title_to_doc[title] = doc
            title_lower_map[title.lower()] = title

        def _resolve_title(cited: str):
            """Return the canonical doc title for a citation, or None."""
            if cited in title_to_doc:
                return cited
            return title_lower_map.get(cited.lower())

        # Convert [Title] citations to inline markdown links [Title](URL)
        final_answer = llm_answer
        for cited in cited_titles:
            canonical = _resolve_title(cited)
            if canonical and title_to_url.get(canonical):
                final_answer = final_answer.replace(
                    f'[{cited}]',
                    f'[{cited}]({title_to_url[canonical]})'
                )

        # Build sources list from matched cited documents
        cited_docs = [
            title_to_doc[_resolve_title(t)]
            for t in cited_titles
            if _resolve_title(t) is not None
        ]
        # Show all retrieved docs as sources — they all contributed to the answer.
        # cited_docs (inline citations) is a subset; surfacing only them hides relevant docs.
        source_objects = self._format_sources(retrieved_docs)

        # Grounding safety net: flag ungrounded cluster-specific answers.
        # Only fire if retrieval came up empty — if docs were retrieved, the
        # answer is grounded in those docs regardless of whether the model
        # remembered to include inline citation links.
        retrieved_titles = set(title_to_doc.keys())
        genuine_citations = {t for t in cited_titles if len(t.split()) > 1
                             and not t.startswith('url_')}
        if retrieved_docs:
            pass  # docs were retrieved — answer is grounded, skip disclaimer
        elif genuine_citations and (genuine_citations & retrieved_titles
                                    or any(_resolve_title(t) for t in genuine_citations)):
            pass  # model cited a resolvable source — consider it grounded
        else:
            final_answer = self._check_grounding(final_answer, cited_titles, retrieved_titles)

        # Append sources as a markdown list
        if source_objects:
            source_lines = "\n".join(
                f"- [{src.title}]({src.url})" if src.url else f"- {src.title}"
                for src in source_objects
            )
            final_answer = final_answer + f"\n\n**Sources:**\n{source_lines}"

        return final_answer, source_objects

    def query(self, request: QueryRequest) -> QueryResponse:
        """
        Process a user query and return answer with sources.
        
        Args:
            request: QueryRequest with query text and optional cluster
            
        Returns:
            QueryResponse with answer, cluster, and sources
        """
        if not self.chain or not self.retrievers:
            raise HTTPException(
                status_code=503,
                detail="Service Unavailable: RAG service is not initialized."
            )

        # Determine cluster — set by the upstream routing layer in production.
        # Falls back to the first loaded cluster for internal testing where no
        # cluster is passed.  TODO(production): remove fallback once cluster is
        # required in QueryRequest.
        if request.cluster and request.cluster in self.retrievers:
            cluster = request.cluster
        else:
            cluster = next(iter(self.retrievers))

        if cluster not in self.retrievers:
            cluster_list_str = ", ".join(self.retrievers.keys())
            return QueryResponse(
                query_id=str(uuid.uuid4()),
                answer=f"Unknown cluster '{cluster}'. Available: {cluster_list_str}.",
                cluster="unknown",
                sources=[]
            )

        # Every query invocation gets a fresh UUID — passed back to the client
        # so they can submit /feedback referencing this specific response.
        query_id = str(uuid.uuid4())

        # Check semantic cache first
        t_start = time.monotonic()
        if self.semantic_cache:
            cached_response = self.semantic_cache.get(request.query, cluster)
            if cached_response:
                logger.info("Cache hit - returning cached response")
                stats_tracker.record_query(
                    cluster=cluster,
                    query=request.query,
                    latency_s=time.monotonic() - t_start,
                    cache_hit=True,
                    query_id=query_id,
                )
                # Override query_id so the client gets a fresh feedbackable ID
                cached_response["query_id"] = query_id
                return QueryResponse(**cached_response)

        logger.info(f"Processing query for cluster '{cluster}': '{request.query[:100]}...'")

        # Execute RAG chain
        chain_input = {"query": request.query, "cluster": cluster, "retrieved_docs": []}
        try:
            llm_answer = self.chain.invoke(chain_input)
            retrieved_docs = chain_input['retrieved_docs']
            logger.info(f"Retrieved {len(retrieved_docs)} documents")
        except Exception as e:
            logger.error(f"Error during RAG chain invocation: {e}", exc_info=True)
            stats_tracker.record_query(
                cluster=cluster,
                query=request.query,
                latency_s=time.monotonic() - t_start,
                cache_hit=False,
                error=True,
                query_id=query_id,
            )
            raise HTTPException(status_code=500, detail="Failed to generate a response from the model.")

        # Process answer and extract sources
        final_answer, source_objects = self._process_llm_answer(llm_answer, retrieved_docs)

        response = QueryResponse(
            query_id=query_id,
            answer=final_answer,
            cluster=cluster,
            sources=source_objects
        )

        stats_tracker.record_query(
            cluster=cluster,
            query=request.query,
            latency_s=time.monotonic() - t_start,
            cache_hit=False,
            query_id=query_id,
        )

        # Cache the response for future similar queries.
        # Track which source doc files contributed so stale entries can be
        # selectively invalidated when the scraper detects content changes.
        if self.semantic_cache:
            try:
                source_files = list({
                    doc.metadata.get('source', '')
                    for doc in retrieved_docs
                    if doc.metadata.get('source')
                })
                self.semantic_cache.set(
                    request.query,
                    cluster,
                    response.model_dump(),
                    source_files=source_files,
                )
                logger.info(f"Response cached (source files: {source_files})")
            except Exception as e:
                logger.error(f"Failed to cache response: {e}")

        return response
