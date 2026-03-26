# SRCC Chatbot

A RAG (Retrieval-Augmented Generation) API for Stanford Research Computing Center (SRCC) documentation. Answers questions about the Sherlock, Farmshare, Oak, and Elm HPC clusters using documentation scraped from their GitHub repos and the SRCC website.

---

## Architecture

```
Client → FastAPI (app/main.py)
           └─ RAGService
                ├─ vLLM (Qwen2.5-32B-Instruct-AWQ, tensor_parallel_size=2)
                ├─ BM25 retriever (per cluster)
                ├─ FAISS vector index (per cluster, hybrid mode)
                └─ Semantic response cache (SQLite + sentence-transformers)
```

The model runs with `tensor_parallel_size=2`, occupying both NVIDIA L4 GPUs in a single vLLM worker. vLLM's async continuous-batching engine handles all concurrent requests natively — no separate load balancer is needed.

---

## Prerequisites

- Apptainer/Singularity
- Python 3.x with PyYAML on the host (for reading `config.yaml` in shell scripts)
- 2× NVIDIA L4 GPUs (or equivalent with 22+ GiB VRAM each)
- Model files downloaded to the path set in `config.yaml`

---

## Quick Start

```bash
# 1. Fetch and process all documentation (run once, or when docs change)
./filemagic.sh

# 2. Start the API
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
```

**Production mode** (`./main.sh`):
- Stops any existing `chatapi` Apptainer instance and kills stray vLLM processes
- Cleans up orphaned POSIX shared memory blocks in `/dev/shm` left by SIGKILL'd processes
- Reads model path, port, host, and log dir from `config.yaml`
- Validates the model directory exists and contains `config.json`
- Builds `chatbot.sif` from `chatbot.def` if not already built
- Starts the Apptainer instance with GPU support and bind mounts (`$PWD → /workspace`, model dir)
- Launches uvicorn serving `app.main:app` on `0.0.0.0:<port>`
- Model loading takes ~3–4 minutes; watch for `Application startup complete` in the output

**Dev mode** (`./main.sh dev`):
- Same as production but adds `--reload --reload-dir app` to uvicorn
- Each code change triggers a full model reload (~3–4 min); use only when needed

---

### `filemagic.sh` — Fetch and process documentation

```bash
./filemagic.sh
```

Builds the `file_processing.sif` Apptainer container (if not already present), then runs two scripts inside it in sequence:

1. **`file_magic.py`** — clones GitHub repos and converts their MkDocs documentation to flat `.md` files under `docs/`
2. **`scrape_srcc.py`** — crawls `https://srcc.stanford.edu` and writes scraped content as `.md` files to `docs/srcc/`

---

### `stop_multi_gpu.sh` — Stop services

```bash
./stop_multi_gpu.sh
```

Kills load balancer and worker processes by PID file, then by process name as a fallback. Cleans up `.worker1.pid`, `.worker2.pid`, `.loadbalancer.pid` files. (Primarily relevant for the deprecated multi-GPU architecture, but safe to run.)

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
3. Copies each doc file to a flat output directory, injecting YAML front matter (`title`, `url`)
4. Writes a URL map CSV (`docs/<repo>_url_map.csv`)
5. For Sherlock: additionally scrapes specific live pages (facts, tech specs, software list) via HTTP

**Environment variables:**

| Variable | Default | Description |
|---|---|---|
| `REPO_CLONE_DIR` | `docs` | Base directory for cloned repos and output |
| `GITHUB_TOKEN` | — | Token for authenticated GitHub cloning |
| `LOG_FILE` | `magicFile.log` | Log file path |

---

### `scrape_srcc.py`

Two-pass scraper for `https://srcc.stanford.edu`:

**Pass 1 — JSON:API** fetches structured Drupal content types:
- `stanford_news`, `stanford_person`, `stanford_policy`, `stanford_publication`, `stanford_course`
- Uses cursor-based pagination; requests only needed fields
- Falls back to HTML crawl for nodes with no body content

**Pass 2 — HTML crawl** picks up `stanford_page` (Layout Builder) and any pages missed by JSON:API:
- Extracts content from Layout Builder regions (`layout-builder__region`, `layout__region`, etc.)
- Strips nav, header, footer, scripts, Drupal placeholders
- Skips non-HTTP links (`mailto:`, `tel:`, `javascript:`), external domains, binary assets, and noise paths (`/user`, `/admin`, `/search`, `/events`, etc.)

Output: one `.md` file per page in `docs/srcc/`, with YAML front matter (`title`, `url`, `source`).

**Environment variables:**

| Variable | Default | Description |
|---|---|---|
| `SRCC_OUTPUT_DIR` | `docs/srcc` | Output directory |
| `SRCC_MAX_PAGES` | `500` | Maximum pages for the HTML crawl pass |
| `LOG_FILE` | `magicFile.log` | Log file path |

---

## API Endpoints

All endpoints are served by the single uvicorn worker started by `main.sh`.

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Service status |
| `GET` | `/health` | Health check — 503 if model not loaded or no retrievers |
| `POST` | `/query/` | Submit a question; returns answer, cluster, and sources |
| `POST` | `/cache/clear` | Flush the semantic response cache |
| `GET` | `/stats` | Worker info: device, model type, clusters, cache status |
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

## Configuration

All settings live in `config.yaml`. Shell scripts and the Python app both read from it directly — it is the single source of truth.

```yaml
model:
  path: /path/to/your/model   # Must exist on disk; validated at startup
  type: "qwen"                # Model architecture — controls prompt format
                              # ("qwen" uses system/human roles; "llama" uses [INST] tags)
  device: "cuda"              # "cpu" or "cuda"; overridden per-worker by WORKER_GPU env var
  use_quantization: false
  local_files_only: true
  dtype: "half"               # vLLM dtype: "half" (float16) required for AWQ models;
                              # use "bfloat16" for non-quantized models that don't support float16

generation:
  max_new_tokens: 512
  do_sample: false            # Greedy decoding
  num_beams: 1
  temperature: null

app:
  title: "SRC Cluster Knowledge Base API"
  description: "..."
  version: "1.0.0"

api:
  cors_origins:
    - "http://localhost:4000"
    - "https://docs-dev.carina.stanford.edu"
    - "https://ada-lovelace.stanford.edu"

caching:
  SEMANTIC_CACHE_ENABLED: true
  SEMANTIC_CACHE_THRESHOLD: 0.70   # Cosine similarity threshold (0–1); lower = more permissive
  SEMANTIC_CACHE_DB: "/workspace/.response_cache.db"
  LANGCHAIN_CACHE_DB: "/workspace/.langchain.db"

retrieval:
  MAX_RETRIEVED_DOCS: 5
  MIN_BM25_SCORE: 1.0              # Docs below this BM25 score are discarded; set 0.0 to disable
  HYBRID_ENABLED: true             # Combine BM25 + FAISS via reciprocal rank fusion
  VECTOR_MODEL: "all-MiniLM-L6-v2"
  RRF_K: 60                        # Reciprocal rank fusion constant

grounding:
  GROUNDING_CHECK_ENABLED: true
  REFUSAL_DISCLAIMER: "Note: This answer may not reflect your cluster's specific configuration..."

clusters:
  sherlock: "docs/sherlock/"
  farmshare: "docs/farmshare/"
  oak: "docs/oak/"
  elm: "docs/elm/"

server:
  api_port: 8000
  host: "ada-lovelace.stanford.edu"

logging:
  log_dir: "logs"

workers:                           # Used only by the deprecated load_balancer.py
  - url: "http://localhost:8001"
    port: 8001
    gpu: "cuda:0"
  - url: "http://localhost:8002"
    port: 8002
    gpu: "cuda:1"
```

### Environment variable overrides

These override the corresponding `config.yaml` values at runtime:

| Variable | Overrides | Description |
|---|---|---|
| `MODEL_PATH` | `model.path` | Path to the LLM model directory |
| `API_PORT` | `server.api_port` | Port for the API |
| `API_HOST` | `server.host` | Hostname shown in startup messages |
| `LOG_DIR` | `logging.log_dir` | Directory for app logs |

Pass variables into the Apptainer container with the `APPTAINERENV_` prefix:
```bash
APPTAINERENV_WORKER_GPU=cuda:0   # assign a specific GPU
APPTAINERENV_PYTHONPATH=/workspace
```

---

## File Structure

```
apichatbot/
├── main.sh                  # Start the API (production or dev mode)
├── filemagic.sh             # Fetch and process all documentation
├── stop_multi_gpu.sh        # Stop worker/load-balancer processes
├── start_multi_gpu.sh       # DEPRECATED — exits with error
├── chatbot.def              # Apptainer container definition (API)
├── chatbot.sif              # Built container (generated)
├── file_processing.def      # Apptainer container definition (doc processing)
├── file_processing.sif      # Built container (generated)
├── config.yaml              # Central configuration — source of truth
├── file_magic.py            # Clone GitHub repos, flatten MkDocs docs to .md
├── scrape_srcc.py           # Scrape srcc.stanford.edu → docs/srcc/
├── var_clean_up.py          # Markdown variable substitution utility
├── app/
│   ├── main.py              # FastAPI app and route definitions
│   ├── rag_service.py       # Model loading, retrieval, generation, caching
│   ├── config.py            # Settings — reads config.yaml + env vars
│   ├── prompts.py           # System prompt and chat template logic
│   ├── semantic_cache.py    # SQLite semantic response cache
│   ├── load_balancer.py     # DEPRECATED — queue-based multi-worker load balancer
│   └── models.py            # Pydantic request/response models
├── docs/                    # Generated documentation (output of filemagic.sh)
│   ├── sherlock/
│   ├── farmshare/
│   ├── oak/
│   ├── elm/
│   └── srcc/                # Scraped srcc.stanford.edu content
└── logs/                    # Runtime logs (generated)
    └── myapp.log
```

---

## Troubleshooting

### Model not loading / wrong model detected

```bash
# Verify the path in config.yaml exists and has the right files
ls -la /path/to/model/
cat /path/to/model/config.json | grep model_type
```

vLLM detects the model architecture from `config.json`. If the path is wrong or missing, vLLM may fall back to a cached model. AWQ models require `dtype: "half"` in `config.yaml`; non-quantized models that don't support float16 (e.g. gemma2) need `dtype: "bfloat16"`.

### Startup hangs after "Loading model..."

vLLM is running with `enforce_eager=True` (see Known Limitations). If startup hangs indefinitely, check for orphaned shared memory from a previous run:

```bash
# Clean up manually
find /dev/shm -maxdepth 1 -user "$USER" -name "psm_*" -delete
```

`main.sh` does this automatically on each start, but if you're running uvicorn directly you may need to do it manually.

### Container won't build

```bash
apptainer build --sandbox test_sandbox chatbot.def
```

### API returns 503

```bash
# Check that the model loaded successfully
tail -100 logs/myapp.log | grep -E "startup|ERROR|FATAL"

# Check GPU availability
nvidia-smi
```

### Cache returning stale responses

```bash
curl -X POST http://localhost:8000/cache/clear
```

---

## Known Limitations

### CUDA Graphs disabled (`enforce_eager=True`)

vLLM runs with `enforce_eager=True`, which disables CUDA graph capture and `torch.compile`.

**Root cause:** Inside Apptainer on ada-lovelace, the NVIDIA Management Library (`libnvidia-ml.so`) fails to initialize. PyTorch's `CUDACachingAllocator` hits a hard assertion on `nvmlInit_v2_()` during graph warmup and aborts. CUDA compute itself works normally.

**Impact:** ~5–15% higher per-token latency. Acceptable for a documentation chatbot.

**To restore CUDA graphs:**
1. Verify NVML is accessible inside the container: `apptainer exec --nv instance://chatapi ldconfig -p | grep nvidia-ml`
2. Confirm the NVML version matches the host driver: `nvidia-smi`
3. If missing, request SRCC bind `libnvidia-ml.so` from the host via `--nv`
4. Once NVML initializes cleanly, remove `enforce_eager=True` from `_load_model()` in `app/rag_service.py`

---

## Support

For issues with the clusters or API: `srcc-support@stanford.edu`

For debugging the chatbot:
1. Check `logs/myapp.log`
2. Run `nvidia-smi` to confirm GPU availability
3. Verify model path and `dtype` in `config.yaml`
