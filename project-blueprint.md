# Project Blueprint: Ada Chatbot

*Generated for account migration — exhaustive context for onboarding a new Claude instance.*

---

## 1. Project Name

**Ada** — RAG-based documentation chatbot for Stanford Research Computing (SRC) HPC clusters.

---

## 2. Custom Project Instructions (Verbatim)

> You are working on the Ada chatbot project for Stanford Research Computing. This is a production RAG chatbot serving documentation for Stanford's HPC clusters (Sherlock, Farmshare, Oak, Elm, Carina, Nero). The backend is a FastAPI app running inside an Apptainer container on `ada-lovelace.stanford.edu`, using vLLM for inference with Qwen3-32B-AWQ on 2× NVIDIA L4 GPUs. The frontend is a Jekyll-based static site in `chatbot-client-2026/`. There is also a separate MCP server project at `/srv/scratch/bcritt/src-mcp/`.
>
> Key constraints:
> - Never commit or push git changes without explicit user instruction
> - The server is ARM64 (sbsa) — CUDA kernel compatibility matters
> - Config-driven: all runtime settings live in `config.yaml`, not hardcoded
> - Container is portable: no hardcoded paths baked into `chatbot.def` except `requirements.txt`
> - Apptainer bind mounts: `APP_DIR` → `/workspace`; all other absolute paths bind as themselves
> - `CONTAINER_ADA_CONFIG` in `main.sh` translates host config path to in-container path

---

## 3. Knowledge Files

These files are part of the project codebase and serve as the primary reference material:

| File | Description |
|------|-------------|
| `CONTEXT.md` | Session context doc — server environment, model rationale, outstanding issues, diagnostic commands |
| `config.yaml` | All runtime configuration: model path, retrieval settings, caching, cluster paths, workers |
| `app/rag_service.py` | Core RAG logic: document loading, BM25+FAISS retrieval, RRF fusion, vLLM generation, citation parsing, source building, semantic caching, disk index caching |
| `app/prompts.py` | System prompt / Ada's instructions + LangChain chat templates for Llama vs. Qwen |
| `app/config.py` | Pydantic settings class — reads `config.yaml` and env vars |
| `app/semantic_cache.py` | SQLite-backed semantic response cache with sentence-transformer similarity |
| `app/main.py` | FastAPI app entrypoint, health check, `/query/` and `/feedback/` endpoints |
| `app/models.py` | Pydantic request/response models |
| `app/load_balancer.py` | Round-robin request router with async lock for multi-GPU workers |
| `main.sh` | Primary launcher — single-GPU dev mode and production mode; bind mount logic; `CONTAINER_ADA_CONFIG` translation |
| `chatbot.def` | Apptainer container definition — installs Python env with `uv`, vLLM wheel, requirements |
| `requirements.txt` | Python dependencies (does NOT include vllm — installed separately in chatbot.def) |
| `setup_upgrade.sh` | Interactive tool: container rebuild, model download, BM25 score tuning |
| `start_multi_gpu.sh` | Multi-GPU launcher (2 workers + load balancer) |
| `stop_multi_gpu.sh` | Teardown script |
| `file_magic.py` | Clones private GitHub repos (per-cluster doc repos) using PAT, converts to markdown |
| `README.md` | Deployment-specific README for the SRC instance |
| `README.generic.md` | Generic/portable README for adapting Ada to a new deployment |
| `systemd/` | Systemd unit files for production service management |
| `docs/sherlock/` | Sherlock cluster documentation (markdown) |
| `docs/farmshare/` | Farmshare cluster documentation |
| `docs/oak/` | Oak storage documentation |
| `docs/elm/` | Elm cluster documentation |
| `docs/carina/` | Carina cluster documentation |
| `docs/nero/` | Nero cluster documentation |
| `docs/srcc/` | Shared SRCC web content — merged into every cluster's retriever |

**Frontend (separate repo: `chatbot-client-2026/`):**

| File | Description |
|------|-------------|
| `docs/assets/js/chatbot.js` | Main frontend JS: chat UI, history, feedback system, API integration |
| `docs/assets/css/chatbot.css` | All chatbot widget styles including feedback UI |
| `docs/Gemfile` | Ruby deps — pinned to liquid 4.0.4 and webrick for rbenv 3.3.6 compatibility |
| `docs/package.json` | Node build scripts (proxy server) |

---

## 4. Project Summary

### What It Is

Ada is a production RAG chatbot that answers questions about Stanford Research Computing's HPC clusters. Users interact through a chat widget embedded in the SRC documentation site. The backend retrieves relevant documentation chunks and passes them to a local LLM for grounded answers with inline citations.

### Architecture

```
Browser (Jekyll site)
    └── chatbot.js widget → POST /query/ → FastAPI load balancer (ada-lovelace:8000)
                                                ├── Worker 1 (port 8001, GPU 0)
                                                └── Worker 2 (port 8002, GPU 1)
                                                        └── vLLM (Qwen3-32B-AWQ)
                                                        └── BM25 + FAISS retrievers
                                                        └── Semantic cache (SQLite)
                                                        └── Disk index cache (FAISS/BM25)
```

All running inside an Apptainer container (`chatbot.sif`) on `ada-lovelace.stanford.edu`.

### Server Environment

- **Host:** `ada-lovelace.stanford.edu`
- **Working dir:** `/srv/scratch/bcritt/chatbot`
- **Container:** `chatbot.sif` (Apptainer, built from `chatbot.def`)
- **Python env inside container:** `/opt/chatbot-env` (managed by `uv`)
- **GPUs:** 2× NVIDIA L4 (24GB each) — **ARM64/sbsa architecture**
- **CUDA:** 13.0 (`cu130`)
- **Model:** `Qwen3-32B-AWQ` at `/home/users/bcritt/apichatbot/models/Qwen3-32B-AWQ`
- **Embedding model:** `gte-large-en-v1.5` at `/home/users/bcritt/apichatbot/models/gte-large-en-v1.5`
- **HF cache:** `/srv/scratch/bcritt/chatbot/.hf_cache` (bind-mounted as `/workspace/.hf_cache`)
- **Index cache:** `/srv/scratch/bcritt/chatbot/indices/` (FAISS/BM25 disk cache)
- **Semantic cache DB:** `/srv/scratch/bcritt/chatbot/.response_cache.db`

### Model History & Rationale

| Model | Status | Reason |
|-------|--------|--------|
| Gemma 2 9B IT | Replaced | Sara's model, generic knowledge |
| Llama 3.3 70B AWQ | Rejected | 35GB, doesn't fit on single L4 |
| Qwen 2.5 32B AWQ | Replaced by Qwen3 | Good, but newer model available |
| **Qwen3-32B-AWQ** | **Current** | Best instruction following, fits on L4, AWQ INT4 |

ARM64 constraint eliminates ExLlama v2 kernels → vLLM is the only viable inference backend.

### Retrieval Pipeline

1. **BM25** (keyword): query prepended with cluster name, filtered by `MIN_BM25_SCORE: 1.0`
2. **FAISS** (semantic): cosine similarity via `gte-large-en-v1.5` embeddings, `MAX_RETRIEVED_DOCS: 5`
3. **RRF fusion**: Reciprocal Rank Fusion combines BM25 + FAISS results (`RRF_K: 60`, `FAISS_RRF_WEIGHT: 2.0`)
4. **RRF scores** stored in `doc.metadata['rrf_score']` for downstream use
5. **Context formatting**: `--- Document: Title > Section ---` headers passed to LLM
6. **Semantic cache**: similarity threshold 0.85 (was 0.70 — caused false positives; 0.90 was too strict)
7. **Disk index cache**: SHA-256 fingerprint of all `.md` content + chunking params + model path; saves FAISS index + BM25 pickle to `indices/`. Makes restarts instant (~1s) vs. 30-min rebuild.

### Citation System

The LLM is instructed to cite every document it uses with `[Title]` inline, using the exact string from `--- Document: Title ---` headers. The backend:
1. Extracts `[Title]` patterns via regex
2. Resolves them to docs via `_resolve_title()` — handles exact match, case-insensitive, and `Title > Section` suffix stripping (model sometimes cites section-qualified titles)
3. Converts inline citations to markdown links `[Title](URL)`
4. Builds Sources list from cited docs only (works-cited style, not bibliography)

### Feedback System

Users see faint 👍/👎 icons on Ada's responses (opacity 0.2, visible on hover). Clicking 👎 expands a panel with:
- "What was wrong?" heading
- 2×2 grid of tag pills: Wrong answer / Incomplete / Outdated / Wrong cluster
- Submit (Stanford red #B1040E) / Cancel buttons

Each response gets a UUID `query_id`. Feedback POSTs to `/feedback/` endpoint. Submitted state persisted in `sessionStorage` so icons don't re-prompt.

### Key Decisions Made

| Decision | Rationale |
|----------|-----------|
| vLLM over transformers | ARM64 — no ExLlama v2 CUDA kernels |
| VLLM_USE_V1 removed | vLLM 0.18.0 doesn't have v1 engine toggle |
| Config-driven everything | Admin portability — one file to edit per deployment |
| Bind mount config, not copy | Container stays portable; config lives on host |
| `CONTAINER_ADA_CONFIG` in main.sh | Translates host path to `/workspace/` prefix when config is inside APP_DIR |
| Disk index cache | 30-min FAISS rebuild → ~1s load on restart |
| SHA-256 fingerprint invalidation | Auto-detects doc changes, chunk size changes, model changes |
| Sources = cited docs only | Works-cited style; prompt instructs model to cite every doc it uses |
| `_resolve_title` section stripping | Model cites `Storage > Where should I store my files?`, backend resolves to `Storage` |
| Semantic cache threshold 0.85 | 0.70 matched GPU questions to Ollama questions; 0.90 too strict |
| Feedback icons unobtrusive | Opacity 0.2 at rest, slightly visible on hover — no "Was this helpful?" nag |
| Removed %files from chatbot.def | Container was baking in docs/app — broke builds from other locations |
| FAISS_RRF_WEIGHT: 2.0 | Prefer semantic search over BM25 keyword matching |
| Cluster name prepended to BM25 query | Naturally boosts cluster-relevant docs without biasing FAISS |

### Current State (as of last session)

- ✅ Backend running on ada-lovelace with Qwen3-32B-AWQ
- ✅ Disk index cache working (all 6 clusters load in <1s)
- ✅ Semantic cache working (threshold 0.85)
- ✅ Inline citations resolved and linked
- ✅ Sources section populated from cited docs
- ✅ Feedback UI integrated in frontend
- ✅ Grounding disclaimer fires only when retrieval comes up empty
- ✅ Out-of-scope detection working
- ✅ Ingest pipeline: RSS + Slack + Freshdesk running on APScheduler
- ⚠️ BM25 noisy — often returns researcher profiles and irrelevant docs in top-5; FAISS compensates via RRF but BM25 slots are wasted
- ⚠️ MIN_BM25_SCORE: 1.0 is effectively no filtering (976/1093 chunks pass for typical queries)

### Outstanding Work

- **BM25 filtering**: Score distribution is too flat — relevant and irrelevant docs score within 1 point of each other. Consider disabling BM25 filtering entirely and relying on RRF, or moving to a TF-IDF variant.
- **Feedback button vertical spacing**: Was iterating on CSS padding when context ended — flexbox gap approach working, may need final tweak.
- **src-mcp**: Separate MCP server project at `/srv/scratch/bcritt/src-mcp/` — was testing/running when context ended.
- **Jekyll site deployment**: `chatbot-client-2026/` changes (feedback UI) need to be deployed.

### User Preferences for This Project

- **No auto-commit/push** — never run `git commit` or `git push` without explicit instruction
- **No filler docs** — don't create README or markdown files unless asked
- **Read before edit** — always read a file before editing it
- **Terse** — code-first, no padding
- **Check server logs directly** — user tails logs on server; don't guess at server state
- **Don't suggest SSH tunneling** — user has `LocalForward 8000 localhost:8000` in `~/.ssh/config` already
- **Config over code** — prefer adding config keys to `config.yaml` over hardcoding values

### Key Paths Quick Reference

```
Server working dir:     /srv/scratch/bcritt/chatbot/
Model:                  /home/users/bcritt/apichatbot/models/Qwen3-32B-AWQ
Embedding model:        /home/users/bcritt/apichatbot/models/gte-large-en-v1.5
HF cache:               /srv/scratch/bcritt/chatbot/.hf_cache
Index cache:            /srv/scratch/bcritt/chatbot/indices/
Semantic cache DB:      /srv/scratch/bcritt/chatbot/.response_cache.db
Logs:                   /srv/scratch/bcritt/chatbot/logs/
Config:                 /srv/scratch/bcritt/chatbot/config.yaml
Container:              /srv/scratch/bcritt/chatbot/chatbot.sif
MCP server:             /srv/scratch/bcritt/src-mcp/
Local project:          /Users/bcritt/Documents/StanfordProjects/claude/chatbot/
Local frontend:         /Users/bcritt/Documents/StanfordProjects/claude/chatbot-client-2026/
```

### Diagnostic Commands

```bash
# Check service is up
curl -s http://localhost:8000/health | python3 -m json.tool

# Test a query
curl -s -X POST http://localhost:8000/query/ \
  -H "Content-Type: application/json" \
  -d '{"query":"How do I submit a batch job on Sherlock?","cluster":"sherlock"}' | python3 -m json.tool

# Watch logs
tail -f /srv/scratch/bcritt/chatbot/logs/myapp.log

# Clear semantic cache
rm /srv/scratch/bcritt/chatbot/.response_cache.db
touch /srv/scratch/bcritt/chatbot/.response_cache.db
chmod 666 /srv/scratch/bcritt/chatbot/.response_cache.db

# Check index cache
ls -lh /srv/scratch/bcritt/chatbot/indices/

# Restart service
cd /srv/scratch/bcritt/chatbot && ./main.sh prod
```
