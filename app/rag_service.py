import logging
import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple

import frontmatter
import numpy as np
from fastapi import HTTPException
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.cache import SQLiteCache
from langchain_community.retrievers import BM25Retriever
from langchain_core.globals import set_llm_cache
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from vllm import LLM, SamplingParams

from app.config import Settings
from app.models import QueryRequest, QueryResponse, Source
from app.prompts import get_prompt_template

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
                    doc = Document(page_content=post.content, metadata=metadata)
                    documents.append(doc)
                    logger.info(f"Loaded document from {file_path}")
                except Exception as e:
                    logger.warning(f"Could not read or parse front matter from file {file_path}: {e}")
        return documents

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
                    similarity_threshold=self.settings.SEMANTIC_CACHE_THRESHOLD
                )
                logger.info(f"✓ Semantic cache initialized (threshold: {self.settings.SEMANTIC_CACHE_THRESHOLD})")
            except Exception as e:
                logger.error(f"Failed to initialize semantic cache: {e}")
                self.semantic_cache = None
        else:
            logger.warning("Semantic cache disabled")
            self.semantic_cache = None

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
            dtype="half",
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

    def _load_retrievers(self):
        """Load BM25 and (optionally) FAISS retrievers for each cluster"""
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

        # Load embedding model once if hybrid retrieval is enabled
        if self.settings.HYBRID_ENABLED and FAISS_AVAILABLE:
            logger.info(f"Loading embedding model for hybrid retrieval (CPU): {self.settings.VECTOR_MODEL}")
            self.embedding_model = SentenceTransformer(self.settings.VECTOR_MODEL, device='cpu')
            logger.info("Embedding model loaded on CPU.")
        elif self.settings.HYBRID_ENABLED and not FAISS_AVAILABLE:
            logger.warning("Hybrid retrieval enabled but faiss-cpu not installed. Falling back to BM25 only.")

        for cluster_name, path in self.settings.CLUSTERS.items():
            if not os.path.isdir(path):
                logger.warning(f"Directory not found for cluster '{cluster_name}': {path}. Skipping.")
                continue

            logger.info(f"Ingesting documents for cluster: {cluster_name}")
            documents = self._ingest_markdown_files(path)
            if not documents:
                logger.warning(f"No documents found for cluster '{cluster_name}'.")
                continue

            split_docs = text_splitter.split_documents(documents)
            self.retrievers[cluster_name] = BM25Retriever.from_documents(split_docs)
            logger.info(f"Created BM25 retriever for '{cluster_name}' ({len(split_docs)} chunks).")

            # Build FAISS index for this cluster
            if self.embedding_model is not None:
                embeddings = self.embedding_model.encode(
                    [doc.page_content for doc in split_docs],
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                dimension = embeddings.shape[1]
                index = faiss.IndexFlatIP(dimension)  # Inner product on normalized vectors = cosine similarity
                index.add(embeddings.astype(np.float32))
                self.vector_stores[cluster_name] = {
                    "index": index,
                    "docs": split_docs,
                }
                logger.info(f"Created FAISS index for '{cluster_name}' ({len(split_docs)} vectors, dim={dimension}).")

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

        k = min(self.settings.MAX_RETRIEVED_DOCS * 2, store["index"].ntotal)
        scores, indices = store["index"].search(query_embedding, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:  # FAISS returns -1 for missing results
                results.append((store["docs"][idx], float(score)))
        return results

    def _reciprocal_rank_fusion(
        self,
        bm25_results: List[Tuple[Document, float]],
        faiss_results: List[Tuple[Document, float]],
    ) -> List[Document]:
        """Merge two ranked lists using reciprocal rank fusion (RRF)."""
        k = self.settings.RRF_K
        # Use page_content as dedup key since the same chunk can appear in both lists
        rrf_scores: Dict[str, float] = defaultdict(float)
        doc_map: Dict[str, Document] = {}

        for rank, (doc, _score) in enumerate(bm25_results):
            key = doc.page_content
            rrf_scores[key] += 1.0 / (k + rank + 1)
            doc_map[key] = doc

        for rank, (doc, _score) in enumerate(faiss_results):
            key = doc.page_content
            rrf_scores[key] += 1.0 / (k + rank + 1)
            doc_map[key] = doc

        # Sort by fused score descending
        sorted_keys = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        return [doc_map[key] for key in sorted_keys[:self.settings.MAX_RETRIEVED_DOCS]]

    def _build_chain(self):
        """Build the RAG chain with retriever, prompt, and LLM"""
        prompt_template = get_prompt_template(self.settings.MODEL_TYPE)

        def retrieve_and_format_context(inputs: Dict) -> str:
            query, cluster = inputs['query'], inputs['cluster']

            # BM25 retrieval with score filtering
            bm25_results = self._retrieve_bm25_with_scores(query, cluster)
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

            # Lead with an explicit title list so the model knows the exact
            # strings it must use when citing — prevents invented titles like
            # "Scheduling on Sherlock" that don't match any retrieved document.
            title_list = "\n".join(f"  - {t}" for t in titles)
            docs_text = "\n\n".join(
                f"--- Document: {title} ---\n{_clean_content(doc.page_content)}"
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

    # Keywords that suggest the answer is about cluster-specific configuration
    _CLUSTER_SPECIFIC_PATTERNS = re.compile(
        r'\b(?:partition|sbatch|srun|squeue|scancel|salloc|sinfo|scontrol'
        r'|module\s+load|module\s+avail|scratch|oak|sherlock|farmshare|elm'
        r'|slurm|quota|storage|node|gpu\s+partition|memory\s+limit'
        r'|job\s+submit|batch\s+script|queue'
        # Also catch refusal messages that mention SRCC/HPC context — these fire
        # when the model correctly declines off-topic questions but the grounding
        # disclaimer should still be appended so users know where to ask instead.
        r'|srcc|stanford\s+research|hpc\s+cluster|computing\s+center)\b',
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
        source_objects = self._format_sources(cited_docs) if cited_docs else []

        # Grounding safety net: flag ungrounded cluster-specific answers.
        # Skip if the model made genuine multi-word citation attempts — those
        # indicate it's trying to ground its answer even if the title didn't
        # match exactly (avoids false-positive disclaimers on cited answers).
        retrieved_titles = set(title_to_doc.keys())
        genuine_citations = {t for t in cited_titles if len(t.split()) > 1
                             and not t.startswith('url_')}
        # Grounding check — only skip if a genuine citation resolved correctly.
        if genuine_citations and (genuine_citations & retrieved_titles
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
                answer=f"Unknown cluster '{cluster}'. Available: {cluster_list_str}.",
                cluster="unknown",
                sources=[]
            )
        
        # Check semantic cache first
        if self.semantic_cache:
            cached_response = self.semantic_cache.get(request.query, cluster)
            if cached_response:
                logger.info("Cache hit - returning cached response")
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
            raise HTTPException(status_code=500, detail="Failed to generate a response from the model.")

        # Process answer and extract sources
        final_answer, source_objects = self._process_llm_answer(llm_answer, retrieved_docs)

        response = QueryResponse(
            answer=final_answer,
            cluster=cluster,
            sources=source_objects
        )
        
        # Cache the response for future similar queries
        if self.semantic_cache:
            try:
                self.semantic_cache.set(
                    request.query,
                    cluster,
                    response.model_dump()
                )
                logger.info("Response cached")
            except Exception as e:
                logger.error(f"Failed to cache response: {e}")
        
        return response
