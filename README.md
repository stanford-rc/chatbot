# SRCC Chatbot

A RAG (Retrieval-Augmented Generation) API for Stanford Research Computing Center (SRCC) documentation. Answers questions about the Sherlock, Farmshare, Oak, Elm, Carina, and Nero clusters using documentation scraped from their GitHub repos and websites.

Runs on `ada-lovelace.stanford.edu` — 96 Ampere-1a cores, 378 GB RAM, 2x NVIDIA L4 GPUs (22.5 GiB each).

---

## Architecture

```
Client -> FastAPI (app/main.py)
            +-- RAGService
                 |-- vLLM (Qwen3-32B-AWQ, tensor_parallel_size=2)
                 |-- BM25 retriever (per cluster)
                 |-- FAISS vector index (per cluster, hybrid mode)
                 |   +-- gte-large-en-v1.5 embeddings (CPU)
                 +-- Semantic response cache (SQLite + sentence-transformers)
                      +-- Content-aware invalidation via scraper manifests
```

The model runs with `tensor_parallel_size=2`, occupying both L4 GPUs in a single vLLM worker. vLLM's async continuous-batching engine handles all concurrent requests natively — no separate load balancer is needed.

**Retrieval pipeline:** Each query is run through both BM25 (keyword) and FAISS (semantic) retrievers. Results are merged via Reciprocal Rank Fusion (RRF), with FAISS weighted higher by default. Documents are split on `##` header boundaries to preserve section semantics (e.g., "Upcoming Classes" vs "Recent Classes").

**Cache invalidation:** When the scrapers run, they produce content manifests (SHA-256 hashes per file). On the next service restart, the RAG service compares the current manifest against the previous one and evicts only the cache entries whose source documents changed. Stable content stays cached.

---

## Prerequisites

- Apptainer/Singularity
- Python 3.x with PyYAML on the host (for reading `config.yaml` in shell scripts)
- 2x NVIDIA L4 GPUs (or equivalent with 22+ GiB VRAM each)
- Model files downloaded to the paths set in `config.yaml` (see `setup_upgrade.sh`)

---

## Quick Start

```bash
# 1. Download the model and rebuild the container (first time only)
./setup_upgrade.sh

# 2. Fetch and process all documentation (run once, or when docs change)
./filemagic.sh

# 3. Start the API
./main.sh

# Access the API
curl http://ada-lovelace.stanford.edu:8000/health
curl http://ada-lovelace.stanford.edu:8000/docs
```

---

## Scripts

### `main.sh` — Start the API

```bash
./main.sh        # Production mode (default)
./main.sh dev    # Dev mode — enables uvicorn --reload
./main.sh multi  # Exits with error (deprecated, see below)
```

**Production mode** (`./main.sh`):
- Stops any existing `chatapi` Apptainer instance and kills stray vLLM processes
- Cleans up orphaned POSIX shared memory blocks in `/dev/shm` left by SIGKILL'd processes
- Reads model path, port, host, and log dir from `config.yaml`
- Validates the model directory exists and contains `config.json`
- Builds `chatbot.sif` from `chatbot.def` if not already built
- Starts the Apptainer instance with GPU support; binds `$PWD -> /workspace` and the parent models directory so both the LLM and the embedding model are accessible
- Launches uvicorn serving `app.main:app` on `0.0.0.0:<port>`
- Model loading takes several minutes; watch for `Application startup complete`

**Dev mode** (`./main.sh dev`):
- Same as production but adds `--reload --reload-dir app` to uvicorn
- Each code change triggers a full model reload; use only when needed

**Multi mode**: Exits immediately with an error. `tensor_parallel_size=2` occupies both GPUs in a single worker — a second worker would OOM immediately.

---

### `filemagic.sh` — Fetch and process documentation

```bash
./filemagic.sh
```

Builds the `file_processing.sif` Apptainer container (if not already present), then runs four steps inside it in sequence:

1. **`file_magic.py`** — clones GitHub repos and converts their MkDocs documentation to flat `.md` files under `docs/`
2. **`scrape_srcc.py`** — crawls `https://srcc.stanford.edu` -> `docs/srcc/`
3. **`scrape_static_docs.py`** — crawls Carina and Nero documentation sites -> `docs/carina/`, `docs/nero/`
4. **`generate_manifests.py`** — writes `.content_manifest.json` in each docs subdirectory (SHA-256 hashes per file, used for cache invalidation on next restart)

---

### `setup_upgrade.sh` — Download model and rebuild container

```bash
./setup_upgrade.sh              # Run all steps
./setup_upgrade.sh rebuild      # Only rebuild container
./setup_upgrade.sh download     # Only download model from HuggingFace
./setup_upgrade.sh tune         # Only run interactive BM25 threshold tuning
```

**Steps:**

1. **Rebuild** — deletes and rebuilds `chatbot.sif` from `chatbot.def`. Run after changing `requirements.txt` or the container definition.
2. **Download** — downloads `Qwen/Qwen3-32B-AWQ` (~35 GB) via `huggingface-cli` and updates `config.yaml` with the new model path and type.
3. **Tune** — sends a set of test queries to the running API and displays source counts, then offers an interactive prompt to update `MIN_BM25_SCORE` in `config.yaml`. Watch `logs/myapp.log` for the per-document BM25 scores while it runs.

---

### `stop_multi_gpu.sh` — Stop services

```bash
./stop_multi_gpu.sh
```

Kills load balancer and worker processes by PID file, then by process name as a fallback. Cleans up `.worker1.pid`, `.worker2.pid`, `.loadbalancer.pid`.

---

### `comprehensive_test.sh` — Integration test suite

```bash
./comprehensive_test.sh
./comprehensive_test.sh --port 8001 --host localhost
./comprehensive_test.sh --no-cache-clear
```

Runs six test groups against the live API. Requires `jq`.

| Test | What it checks |
|------|----------------|
| 0 — Connectivity | `/health` returns 200 and lists clusters |
| 1 — Basic RAG | `/query/` returns a substantive answer with sources |
| 2 — Citation quality | Inline `[Title](URL)` links present; no raw `.md` filenames cited |
| 3 — Grounding guard | Off-topic questions refused; no false-positive disclaimers on grounded answers |
| 4 — Cross-cluster isolation | Sherlock and Farmshare queries return distinct answers tagged with correct cluster |
| 5 — Semantic cache | Exact and semantically similar repeats return cached responses in <3 s |
| 6 — Concurrent throughput | 3 parallel requests across clusters all succeed |

Clears the semantic cache before running by default (`POST /cache/clear`). Results are saved to `/tmp/chatbot_test_<timestamp>/` and can be inspected with `jq`.

---

### `test_cache.sh` — Quick cache smoke test

```bash
./test_cache.sh
```

Sends four queries (exact match, near-exact, semantically similar) against `localhost:8000` and compares responses. Faster than the full test suite when you just want to verify caching is working.

---

### `gpu.sh` — GPU compute diagnostic

```bash
./gpu.sh
```

Runs inside the `chatapi` Apptainer instance. Reports PyTorch version, CUDA version, GPU names and compute capabilities, and benchmarks FP16 matrix multiply on the L4 GPUs. Also checks `bitsandbytes` availability. Use to verify GPU compute is working before loading the full model.

---

### `diagnose_gen.sh` — Generation pipeline diagnostic

```bash
./diagnose_gen.sh
```

Loads TinyLlama (1.1B, fast) inside the container and runs five focused tests: single forward pass latency, generation with GPU utilization tracking, device-location verification, model-to-CUDA timing, and PyTorch CUDA settings. Use to distinguish CPU-fallback issues from model/config problems.

---

### Deprecated / legacy scripts

| Script | Status |
|--------|--------|
| `start_multi_gpu.sh` | **Deprecated** — exits immediately with an error. Incompatible with `tensor_parallel_size=2`; use `./main.sh` instead. |
| `bench_gemma.sh` | Legacy — benchmarks Gemma 2 9B via HuggingFace Transformers (not vLLM). No longer relevant. |
| `benchmark_llama.sh` | Legacy — benchmarks TinyLlama 1.1B. No longer relevant. |

---

## Document Ingestion

### `file_magic.py`

Processes documentation from four Stanford RC GitHub repositories:

| Repo | Output dir |
|------|------------|
| `stanford-rc/farmshare-docs` | `docs/farmshare/` |
| `stanford-rc/docs.elm.stanford.edu` | `docs/elm/` |
| `stanford-rc/docs.oak.stanford.edu` | `docs/oak/` |
| `stanford-rc/www.sherlock.stanford.edu` | `docs/sherlock/` |

For each repo:
1. Shallow-clones the repo (`git clone --depth 1`)
2. Parses `mkdocs.yml` to extract the navigation tree and `site_url`
3. Copies each doc to a flat output directory with YAML front matter (`title`, `url`)
4. Writes a URL map CSV (`docs/<repo>_url_map.csv`)
5. For Sherlock: additionally scrapes specific live pages (facts, tech specs, software list) via HTTP

**Environment variables:**

| Variable | Default | Description |
|---|---|---|
| `REPO_CLONE_DIR` | `docs` | Base directory for cloned repos and output |
| `GITHUB_TOKEN` | -- | Token for authenticated GitHub cloning |
| `LOG_FILE` | `magicFile.log` | Log file path |

---

### `scrape_srcc.py`

Two-pass scraper for `https://srcc.stanford.edu`. Output: `docs/srcc/`.

**Pass 1 — JSON:API** fetches structured Drupal content types:
- `stanford_news`, `stanford_person`, `stanford_policy`, `stanford_publication`, `stanford_course`
- Uses cursor-based pagination; requests only needed fields
- Nodes with no body content are deferred to the HTML pass

**Pass 2 — HTML crawl** picks up `stanford_page` (Layout Builder) and any pages missed by JSON:API:
- Extracts content from Layout Builder regions; deduplicates by content fingerprint (first 300 chars)
- Strips nav, header, footer, scripts, Drupal placeholders
- Filters non-HTTP links (`mailto:`, `tel:`, `javascript:`), external domains, binary assets, and noise paths (`/user`, `/admin`, `/search`, `/events`, etc.)

**Environment variables:**

| Variable | Default | Description |
|---|---|---|
| `SRCC_OUTPUT_DIR` | `docs/srcc` | Output directory |
| `SRCC_MAX_PAGES` | `500` | Maximum pages for the HTML crawl pass |
| `LOG_FILE` | `logs/scrapers.log` | Log file path |

---

### `scrape_static_docs.py`

Generic scraper for Stanford static documentation sites (Jekyll/similar, sharing the common `<main id="page-content">` template).

Currently configured for:

| Site | Output dir |
|------|------------|
| `https://docs.carina.stanford.edu` | `docs/carina/` |
| `https://nero-docs.stanford.edu` | `docs/nero/` |

Seeds the crawl queue from the site's nav, then follows internal links. Strips sidebar, nav, header, footer, and scripts. To add a new site, add an entry to the `SITES` dict at the top of the file.

Can be run standalone to scrape a single site:
```bash
python scrape_static_docs.py carina
python scrape_static_docs.py nero
```

**Environment variables:**

| Variable | Default | Description |
|---|---|---|
| `STATIC_DOCS_OUTPUT_DIR` | `docs` | Base output directory |
| `STATIC_DOCS_MAX_PAGES` | `200` | Maximum pages per site |
| `LOG_FILE` | `logs/scrapers.log` | Log file path |

---

### `generate_manifests.py` — Content manifests for cache invalidation

```bash
python generate_manifests.py              # default: docs/
python generate_manifests.py /path/to/docs
```

Writes a `.content_manifest.json` file in each docs subdirectory. Each manifest maps filenames to SHA-256 hashes. The RAG service compares these at startup to detect which docs changed since the last scrape and selectively evicts stale cache entries.

Called automatically at the end of `filemagic.sh`.

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Service status |
| `GET` | `/health` | Health check — 503 if model not loaded or no retrievers |
| `POST` | `/query/` | Submit a question; returns answer, cluster, and sources |
| `POST` | `/cache/clear` | Flush the semantic response cache |
| `GET` | `/stats` | JSON metrics: latency percentiles, cache hit rate, per-cluster counts, top queries |
| `GET` | `/dashboard` | Live monitoring dashboard (Chart.js UI, auto-refreshes every 30 s) |
| `GET` | `/docs` | Auto-generated OpenAPI UI |

**Query request:**
```json
{ "query": "How do I submit a GPU job on Sherlock?", "cluster": "sherlock" }
```
`cluster` is optional — the service will attempt to detect it from the query text.

**Query response:**
```json
{
  "answer": "...",
  "cluster": "sherlock",
  "sources": [{ "title": "...", "url": "..." }]
}
```

---

## Monitoring

### Dashboard

The built-in dashboard is at `/dashboard`. Since ada-lovelace is not directly accessible from a browser, use an SSH tunnel:

```bash
ssh -L 8000:localhost:8000 bcritt@ada-lovelace.stanford.edu
# Then open: http://localhost:8000/dashboard
```

The dashboard displays:
- **KPI cards** — total queries, cache hit rate, average latency, p95/p99 latency, error count
- **Charts** — queries by hour (last 24 h), cache hit/miss ratio, queries by cluster, latency distribution
- **Top queries** — 10 most frequent queries (grouped by semantic similarity)

Auto-refreshes every 30 seconds. All metrics are in-memory and reset on service restart.

### Logs

| File | Contents |
|------|----------|
| `logs/myapp.log` | Application log — startup, retrieval scores, cache hits/misses, errors |
| `logs/stats.jsonl` | Append-only per-query stats: timestamp, cluster, query, latency, cache hit, error |
| `logs/scrapers.log` | Output from `scrape_srcc.py` and `scrape_static_docs.py` |

`stats.jsonl` persists across restarts (append-only). The `/stats` endpoint reads from in-memory counters for fast access; `stats.jsonl` is the durable record.

---

## Configuration Reference

All settings live in `config.yaml`. Shell scripts and the Python app both read from it directly — it is the single source of truth.

### `model` — LLM settings

| Key | Default | Description |
|---|---|---|
| `path` | -- | Absolute path to the model directory on disk. Must contain `config.json`. |
| `type` | `"qwen"` | Model architecture. Controls prompt format: `"qwen"` uses system/human chat roles; `"llama"` uses `[INST]` tags. |
| `device` | `"cuda"` | `"cuda"` or `"cpu"`. |
| `use_quantization` | `false` | Whether to apply runtime quantization. `false` for pre-quantized models (AWQ). |
| `local_files_only` | `true` | Prevent HuggingFace hub downloads at runtime. |
| `dtype` | `"half"` | vLLM dtype. `"half"` (float16) required for AWQ models; `"bfloat16"` for non-quantized. |

### `generation` — Response generation

| Key | Default | Description |
|---|---|---|
| `max_new_tokens` | `1024` | Maximum tokens in the generated response. Increase for longer answers (e.g., workshop lists). |
| `do_sample` | `false` | `false` for greedy decoding (deterministic). `true` enables sampling. |
| `num_beams` | `1` | Beam search width. `1` disables beam search. |
| `temperature` | `null` | Sampling temperature. `null` disables (greedy). |

### `app` — Application metadata

| Key | Default | Description |
|---|---|---|
| `title` | `"SRC Cluster Knowledge Base API"` | Shown in the OpenAPI docs page. |
| `description` | -- | API description for OpenAPI. |
| `version` | `"1.0.0"` | API version string. |

### `api` — API settings

| Key | Default | Description |
|---|---|---|
| `cors_origins` | `[]` | List of allowed CORS origins for the frontend. |

### `caching` — Response caching

| Key | Default | Description |
|---|---|---|
| `SEMANTIC_CACHE_ENABLED` | `true` | Enable the semantic response cache. Requires `sentence-transformers`. |
| `SEMANTIC_CACHE_CLEAR_ON_STARTUP` | `false` | Flush entire cache on every restart. Use `true` during development. In production, leave `false` — content-aware invalidation handles staleness automatically. |
| `SEMANTIC_CACHE_THRESHOLD` | `0.70` | Cosine similarity threshold (0.0--1.0). Lower = more permissive matching. 0.70--0.75 recommended. |
| `SEMANTIC_CACHE_DB` | `"/workspace/.response_cache.db"` | Absolute path to the SQLite cache file. Use `/workspace/` prefix so it lands in the bind-mounted host directory. |
| `LANGCHAIN_CACHE_DB` | `"/workspace/.langchain.db"` | LangChain's internal LLM cache (exact prompt dedup). |

### `retrieval` — Document retrieval

| Key | Default | Description |
|---|---|---|
| `MAX_RETRIEVED_DOCS` | `5` | Number of documents passed to the LLM as context. |
| `CHUNK_SIZE` | `2500` | Maximum characters per chunk. Docs are split on `##` headers first; oversized sections fall back to character splitting. |
| `CHUNK_OVERLAP` | `200` | Character overlap between chunks (character splitter only). |
| `MIN_BM25_SCORE` | `1.0` | BM25 score floor. Documents below this are discarded. `0.0` disables filtering. |
| `HYBRID_ENABLED` | `true` | Combine BM25 (keyword) and FAISS (semantic) retrieval via reciprocal rank fusion. |
| `VECTOR_MODEL` | -- | Path to the sentence-transformer embedding model. Currently `gte-large-en-v1.5` (434M params, MTEB 65.4). |
| `RRF_K` | `60` | Reciprocal rank fusion constant. Higher values compress rank differences. |
| `FAISS_RRF_WEIGHT` | `2.0` | FAISS weight in RRF scoring. `>1.0` prefers semantic matches over keyword matches. |

### `grounding` — Answer safety

| Key | Default | Description |
|---|---|---|
| `GROUNDING_CHECK_ENABLED` | `true` | Append a disclaimer when the answer discusses cluster-specific topics (Slurm commands, partitions, storage) but cites no retrieved documents. |
| `REFUSAL_DISCLAIMER` | -- | The disclaimer text appended to ungrounded answers. |

### `clusters` — Documentation paths

Maps cluster names to their documentation directories (relative to `$PWD`). Each cluster gets its own BM25 + FAISS retriever pair.

```yaml
clusters:
  sherlock: "docs/sherlock/"
  farmshare: "docs/farmshare/"
  oak: "docs/oak/"
  elm: "docs/elm/"
  carina: "docs/carina/"
  nero: "docs/nero/"
```

### `shared_docs` — Cross-cluster documentation

```yaml
shared_docs: "docs/srcc/"
```

Content in this directory is merged into every cluster's retriever at startup. Use for org-wide content (workshops, policies, people) that applies regardless of cluster.

### `server` — Network settings

| Key | Default | Description |
|---|---|---|
| `api_port` | `8000` | Port for the FastAPI server. |
| `host` | `"ada-lovelace.stanford.edu"` | Hostname shown in startup messages and CORS. |

### `logging` — Log output

| Key | Default | Description |
|---|---|---|
| `log_dir` | `"logs"` | Directory for application logs. |
| `stats_log` | `"/workspace/logs/stats.jsonl"` | Per-query stats (latency, cache hit/miss, errors) in JSONL format. |

### `workers` — Deprecated

Legacy multi-worker configuration. Ignored by `main.sh`. Retained for reference only.

### Environment variable overrides

| Variable | Overrides | Description |
|---|---|---|
| `MODEL_PATH` | `model.path` | Path to the LLM model directory |
| `API_PORT` | `server.api_port` | Port for the API |
| `API_HOST` | `server.host` | Hostname shown in startup messages |
| `LOG_DIR` | `logging.log_dir` | Directory for app logs |

Pass variables into the Apptainer container with the `APPTAINERENV_` prefix:
```bash
APPTAINERENV_PYTHONPATH=/workspace   # required — activates sitecustomize.py shims
```

---

## File Structure

```
apichatbot/
|-- main.sh                  # Start the API (production or dev mode)
|-- filemagic.sh             # Fetch and process all documentation
|-- setup_upgrade.sh         # Download model, rebuild container, tune BM25
|-- stop_multi_gpu.sh        # Stop worker/load-balancer processes
|-- comprehensive_test.sh    # Full integration test suite
|-- test_cache.sh            # Quick semantic cache smoke test
|-- gpu.sh                   # GPU compute diagnostic
|-- diagnose_gen.sh          # Generation pipeline diagnostic
|-- generate_manifests.py    # Content hashes for cache invalidation
|-- start_multi_gpu.sh       # DEPRECATED -- exits with error
|-- bench_gemma.sh           # Legacy benchmark (Gemma 2, not vLLM)
|-- benchmark_llama.sh       # Legacy benchmark (TinyLlama, not vLLM)
|-- chatbot.def              # Apptainer container definition (API)
|-- chatbot.sif              # Built container (generated)
|-- file_processing.def      # Apptainer container definition (doc processing)
|-- file_processing.sif      # Built container (generated)
|-- config.yaml              # Central configuration -- source of truth
|-- requirements.txt         # Python dependencies for chatbot container
|-- sitecustomize.py         # Apptainer/vLLM compatibility shims (auto-loaded)
|-- pynvml.py                # Standalone pynvml mock (fallback shim)
|-- file_magic.py            # Clone GitHub repos, flatten MkDocs docs to .md
|-- scrape_srcc.py           # Scrape srcc.stanford.edu -> docs/srcc/
|-- scrape_static_docs.py    # Scrape Carina/Nero docs -> docs/carina/, docs/nero/
|-- var_clean_up.py          # Markdown variable substitution utility
|-- app/
|   |-- main.py              # FastAPI app and route definitions
|   |-- rag_service.py       # Model loading, retrieval, generation, caching
|   |-- config.py            # Settings -- reads config.yaml + env vars
|   |-- prompts.py           # System prompt and chat template logic
|   |-- semantic_cache.py    # SQLite semantic response cache
|   |-- load_balancer.py     # DEPRECATED -- old multi-worker load balancer
|   +-- models.py            # Pydantic request/response models
|-- docs/                    # Generated documentation (output of filemagic.sh)
|   |-- sherlock/
|   |-- farmshare/
|   |-- oak/
|   |-- elm/
|   |-- carina/
|   |-- nero/
|   +-- srcc/
+-- logs/
    +-- myapp.log
```

---

## Troubleshooting

### Model not loading / wrong model detected

```bash
# Verify the path in config.yaml exists and has the right files
ls /home/users/bcritt/apichatbot/models/Qwen3-32B-AWQ/config.json
python3 -c "import json; print(json.load(open('/home/users/bcritt/apichatbot/models/Qwen3-32B-AWQ/config.json'))['model_type'])"
```

vLLM detects the model architecture from `config.json`. If the path is wrong or missing, vLLM may fall back to a cached model (with a different architecture and dtype requirements). AWQ models require `dtype: "half"` in `config.yaml`.

### Embedding model fails with `trust_remote_code` error

The gte-large-en-v1.5 embedding model uses custom Alibaba-NLP code (`modeling.py`, `configuration.py`). These files must be present in the model directory, and `config.json` must reference them as local paths (not `Alibaba-NLP/new-impl--`).

```bash
# Verify custom code files exist
ls /home/users/bcritt/apichatbot/models/gte-large-en-v1.5/*.py

# Verify config.json uses local paths
grep auto_map /home/users/bcritt/apichatbot/models/gte-large-en-v1.5/config.json
# Should show "configuration.NewConfig", not "Alibaba-NLP/new-impl--configuration.NewConfig"
```

If the `.py` files are missing, download them from the `Alibaba-NLP/new-impl` repo:
```bash
huggingface-cli download Alibaba-NLP/new-impl --include "*.py" \
  --local-dir /home/users/bcritt/apichatbot/models/gte-large-en-v1.5
```

If `config.json` still references the remote prefix:
```bash
sed -i 's|Alibaba-NLP/new-impl--||g' /home/users/bcritt/apichatbot/models/gte-large-en-v1.5/config.json
```

### Startup hangs or crashes with "Device string must not be empty"

The pynvml shims in `sitecustomize.py` are not being loaded. Ensure `APPTAINERENV_PYTHONPATH=/workspace` is set (this is done by `main.sh`). If running uvicorn directly, set `PYTHONPATH=$PWD` before the command.

### Startup hangs with "No available shared memory broadcast block"

Orphaned shared memory from a previous SIGKILL'd vLLM process. `main.sh` cleans this up automatically, but if running manually:

```bash
find /dev/shm -maxdepth 1 -user "$USER" -name "psm_*" -delete
```

### API returns 503

```bash
tail -100 logs/myapp.log | grep -E "startup|ERROR|FATAL|CRITICAL"
nvidia-smi
```

### Cache returning stale responses

Content-aware invalidation runs automatically on restart after a scrape. To force a full flush:

```bash
curl -X POST http://ada-lovelace.stanford.edu:8000/cache/clear
```

### Container won't build

```bash
apptainer build --sandbox test_sandbox chatbot.def
```

---

## Known Limitations

### CUDA Graphs disabled (`enforce_eager=True`)

vLLM runs with `enforce_eager=True`, which disables CUDA graph capture and `torch.compile`.

**Root cause:** Even with `sitecustomize.py` working around the NVML crash at startup, PyTorch's `CUDACachingAllocator` contains a separate hard C++ assertion on `nvmlInit_v2_()` at `CUDACachingAllocator.cpp:1124` that fires during CUDA graph warmup. This assertion cannot be patched from Python.

**Impact:** ~5--15% higher per-token latency. Acceptable for a documentation chatbot.

**To restore CUDA graphs** (requires NVML to work natively inside the container):
1. Verify: `apptainer exec --nv instance://chatapi python -c "import pynvml; pynvml.nvmlInit(); print('ok')"`
2. If that fails, check that `--nv` is binding `libnvidia-ml.so` from the host and that its version matches the driver: `nvidia-smi` on the host vs. `ldconfig -p | grep nvidia-ml` inside the container
3. Once NVML initializes cleanly (no warnings in logs), remove `enforce_eager=True` from `_load_model()` in `app/rag_service.py`

### NCCL socket transport

NCCL all-reduce between the two L4 GPUs uses TCP socket over loopback (`NCCL_P2P_DISABLE=1`, `NCCL_SHM_DISABLE=1`). P2P is blocked by IOMMU inside Apptainer; SHM has namespace collisions. This adds ~32 ms per forward pass and makes the initial warmup slow. See `_load_model()` in `app/rag_service.py` for details.

---

## Support

For issues with the clusters or API: `srcc-support@stanford.edu`

For debugging the chatbot:
1. Run `./comprehensive_test.sh` to pinpoint which component is failing
2. Check `logs/myapp.log`
3. Run `./gpu.sh` to verify GPU compute is working
4. Verify model path and `dtype` in `config.yaml`
