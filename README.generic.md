# SRCC Chatbot

A RAG (Retrieval-Augmented Generation) API for Stanford Research Computing Center (SRCC) documentation. Answers questions about HPC clusters using documentation scraped from GitHub repos and websites.

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

The model runs with `tensor_parallel_size=2`, occupying two GPUs in a single vLLM worker. vLLM's async continuous-batching engine handles all concurrent requests natively — no separate load balancer is needed.

**Retrieval pipeline:** Each query is run through both BM25 (keyword) and FAISS (semantic) retrievers. Results are merged via Reciprocal Rank Fusion (RRF), with FAISS weighted higher by default. Documents are split on `##` header boundaries to preserve section semantics.

**Cache invalidation:** When the scrapers run, they produce content manifests (SHA-256 hashes per file). On the next service restart, the RAG service compares the current manifest against the previous one and evicts only the cache entries whose source documents changed. Stable content stays cached.

---

## Prerequisites

- Apptainer/Singularity
- Python 3.x with PyYAML on the host (for reading `config.yaml` in shell scripts)
- 2x NVIDIA GPUs with 22+ GiB VRAM each (tested on L4s)
- Model files downloaded locally (see `setup_upgrade.sh`)

---

## Quick Start

```bash
# 1. Edit config.yaml — set model.path, server.host, cluster doc paths
vi config.yaml

# 2. Download the model and rebuild the container
./setup_upgrade.sh

# 3. Fetch and process all documentation
./filemagic.sh

# 4. Start the API
./main.sh

# Access the API
curl http://localhost:8000/health
curl http://localhost:8000/docs
```

---

## Scripts

### `main.sh` — Start the API

```bash
./main.sh        # Production mode (default)
./main.sh dev    # Dev mode — enables uvicorn --reload
```

**Production mode:**
- Stops any existing `chatapi` Apptainer instance and kills stray vLLM processes
- Cleans up orphaned POSIX shared memory blocks in `/dev/shm`
- Reads all settings from `config.yaml`
- Validates the model directory exists and contains `config.json`
- Builds `chatbot.sif` from `chatbot.def` if not already built
- Starts the Apptainer instance with GPU support; binds `$PWD -> /workspace` and the models directory
- Launches uvicorn; watch for `Application startup complete`

**Dev mode** (`./main.sh dev`):
- Same as production but adds `--reload --reload-dir app` to uvicorn
- Each code change triggers a full model reload; use only when needed

---

### `filemagic.sh` — Fetch and process documentation

```bash
./filemagic.sh
```

Runs four steps inside the `file_processing.sif` container:

1. **`file_magic.py`** — clones GitHub repos and converts MkDocs documentation to flat `.md` files
2. **`scrape_srcc.py`** — crawls the SRCC website
3. **`scrape_static_docs.py`** — crawls additional documentation sites
4. **`generate_manifests.py`** — writes `.content_manifest.json` per docs directory (for cache invalidation)

---

### `setup_upgrade.sh` — Download model and rebuild container

```bash
./setup_upgrade.sh              # Run all steps
./setup_upgrade.sh rebuild      # Only rebuild container
./setup_upgrade.sh download     # Only download model
./setup_upgrade.sh tune         # Only run BM25 threshold tuning
```

---

### `comprehensive_test.sh` — Integration test suite

```bash
./comprehensive_test.sh
./comprehensive_test.sh --port 8001 --host localhost
./comprehensive_test.sh --no-cache-clear
```

Requires `jq`. Runs six test groups:

| Test | What it checks |
|------|----------------|
| 0 — Connectivity | `/health` returns 200 and lists clusters |
| 1 — Basic RAG | `/query/` returns a substantive answer with sources |
| 2 — Citation quality | Inline `[Title](URL)` links present; no raw `.md` filenames |
| 3 — Grounding guard | Off-topic questions refused; no false-positive disclaimers |
| 4 — Cross-cluster isolation | Different clusters return distinct, correctly tagged answers |
| 5 — Semantic cache | Similar repeats return cached responses in <3 s |
| 6 — Concurrent throughput | 3 parallel requests across clusters all succeed |

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
{ "query": "How do I submit a GPU job?", "cluster": "sherlock" }
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

The built-in dashboard is at `/dashboard`. If the server is not directly accessible from a browser, use an SSH tunnel:

```bash
ssh -L 8000:localhost:8000 user@your-server
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
| `max_new_tokens` | `1024` | Maximum tokens in the generated response. |
| `do_sample` | `false` | `false` for greedy decoding (deterministic). `true` enables sampling. |
| `num_beams` | `1` | Beam search width. `1` disables beam search. |
| `temperature` | `null` | Sampling temperature. `null` disables (greedy). |

### `caching` — Response caching

| Key | Default | Description |
|---|---|---|
| `SEMANTIC_CACHE_ENABLED` | `true` | Enable the semantic response cache. |
| `SEMANTIC_CACHE_CLEAR_ON_STARTUP` | `false` | Flush entire cache on every restart. In production, leave `false` — content-aware invalidation handles staleness automatically. |
| `SEMANTIC_CACHE_THRESHOLD` | `0.70` | Cosine similarity threshold (0.0--1.0). Lower = more permissive. 0.70--0.75 recommended. |
| `SEMANTIC_CACHE_DB` | `"/workspace/.response_cache.db"` | Path to the SQLite cache file. Use `/workspace/` prefix for Apptainer bind mounts. |
| `LANGCHAIN_CACHE_DB` | `"/workspace/.langchain.db"` | LangChain's internal LLM cache (exact prompt dedup). |

### `retrieval` — Document retrieval

| Key | Default | Description |
|---|---|---|
| `MAX_RETRIEVED_DOCS` | `5` | Number of documents passed to the LLM as context. |
| `CHUNK_SIZE` | `2500` | Maximum characters per chunk. Docs split on `##` headers first; oversized sections fall back to character splitting. |
| `CHUNK_OVERLAP` | `200` | Character overlap between chunks (character splitter only). |
| `MIN_BM25_SCORE` | `1.0` | BM25 score floor. Documents below this are discarded. `0.0` disables. |
| `HYBRID_ENABLED` | `true` | Combine BM25 + FAISS via reciprocal rank fusion. |
| `VECTOR_MODEL` | -- | Path to the sentence-transformer embedding model. |
| `RRF_K` | `60` | Reciprocal rank fusion constant. Higher values compress rank differences. |
| `FAISS_RRF_WEIGHT` | `2.0` | FAISS weight in RRF. `>1.0` prefers semantic over keyword matches. |

### `grounding` — Answer safety

| Key | Default | Description |
|---|---|---|
| `GROUNDING_CHECK_ENABLED` | `true` | Append a disclaimer when the answer discusses cluster-specific topics but cites no retrieved documents. |
| `REFUSAL_DISCLAIMER` | -- | The disclaimer text appended to ungrounded answers. |

### `clusters` — Documentation paths

Maps cluster names to their documentation directories (relative to `$PWD`). Each cluster gets its own BM25 + FAISS retriever pair.

```yaml
clusters:
  sherlock: "docs/sherlock/"
  farmshare: "docs/farmshare/"
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
| `host` | -- | Hostname for startup messages and CORS. |

### `logging` — Log output

| Key | Default | Description |
|---|---|---|
| `log_dir` | `"logs"` | Directory for application logs. |
| `stats_log` | `"/workspace/logs/stats.jsonl"` | Per-query stats (latency, cache hit/miss, errors) in JSONL format. |

### `data_dir` — Runtime data root

```yaml
data_dir: "/var/lib/ada-chatbot"
```

Optional. When set, any path in the config that starts with `/workspace/` is remapped to this directory at runtime. Leave empty (or omit) to use the app directory. Use this to separate application code from runtime data (caches, logs, docs) when deploying to a read-only or shared location.

### `container` — Apptainer image paths

| Key | Description |
|---|---|
| `sif_path` | Path to the main API container image (`chatbot.sif`). Defaults to `chatbot.sif` in the app directory. |
| `file_processing_sif` | Path to the file-processing container image (`file_processing.sif`). |

### `github` — Source credential

| Key | Description |
|---|---|
| `token_file` | Path to a file containing a GitHub personal access token (no newline). Token is exported as `GITHUB_TOKEN` before `file_magic.py` runs. Required for private repos; omit for public-only. |

### Environment variable overrides

| Variable | Overrides | Description |
|---|---|---|
| `ADA_CONFIG` | -- | Absolute path to `config.yaml`. Takes precedence over the default search path. Use this when deploying config to a system location (e.g. `/etc/ada-chatbot/config.yaml`). |
| `MODEL_PATH` | `model.path` | Path to the LLM model directory |
| `API_PORT` | `server.api_port` | Port for the API |
| `API_HOST` | `server.host` | Hostname |
| `LOG_DIR` | `logging.log_dir` | Directory for app logs |

Pass variables into the Apptainer container with the `APPTAINERENV_` prefix:
```bash
APPTAINERENV_PYTHONPATH=/workspace   # required — activates sitecustomize.py shims
```

---

## Systemd Integration

For production deployments, use the provided unit files in `systemd/` to manage the service and automate daily document scraping.

### Install

```bash
sudo cp systemd/ada-chatbot.service /etc/systemd/system/
sudo cp systemd/ada-chatbot-scrape.service /etc/systemd/system/
sudo cp systemd/ada-chatbot-scrape.timer /etc/systemd/system/

sudo systemctl daemon-reload
sudo systemctl enable --now ada-chatbot.service
sudo systemctl enable --now ada-chatbot-scrape.timer
```

### Units

| Unit | Type | Description |
|------|------|-------------|
| `ada-chatbot.service` | `simple` | Main API service. Starts `main.sh`, restarts on failure. |
| `ada-chatbot-scrape.service` | `oneshot` | Runs `filemagic.sh` (scrape + manifest generation). |
| `ada-chatbot-scrape.timer` | timer | Triggers `ada-chatbot-scrape.service` daily at 02:00. |

### Useful commands

```bash
# API service
sudo systemctl status ada-chatbot
sudo journalctl -u ada-chatbot -f

# Scrape
sudo systemctl start ada-chatbot-scrape   # run immediately
sudo journalctl -u ada-chatbot-scrape -f

# Timer
systemctl list-timers ada-chatbot-scrape
```

### GitHub token

If your documentation repos are private, store a fine-grained PAT in a secrets file:

```bash
sudo mkdir -p /etc/ada-chatbot/secrets
echo -n "github_pat_..." | sudo tee /etc/ada-chatbot/secrets/github_token
sudo chmod 600 /etc/ada-chatbot/secrets/github_token
```

Then reference it in `config.yaml`:
```yaml
github:
  token_file: "/etc/ada-chatbot/secrets/github_token"
```

The token is only used during `filemagic.sh` (cloning) and is never logged or stored elsewhere.

---

## Adapting for a New Deployment

1. **Edit `config.yaml`**: Set `model.path` to your local model directory, `server.host` to your hostname, and update `clusters` to point to your documentation directories.

2. **Set config location** (production): If deploying config to a system path, set `ADA_CONFIG` in the systemd unit or your environment:
   ```
   Environment=ADA_CONFIG=/etc/mybot/config.yaml
   ```
   The app and shell scripts both respect this variable; all relative paths in the config resolve from the config file's directory.

3. **Separate data from code** (optional): Set `data_dir` in `config.yaml` to a writable directory for caches, logs, and docs. Paths starting with `/workspace/` are remapped there automatically:
   ```yaml
   data_dir: "/var/lib/mybot"
   ```

4. **Download the embedding model**: `huggingface-cli download Alibaba-NLP/gte-large-en-v1.5 --local-dir /path/to/models/gte-large-en-v1.5`. Then download custom code: `huggingface-cli download Alibaba-NLP/new-impl --include "*.py" --local-dir /path/to/models/gte-large-en-v1.5` and run `sed -i 's|Alibaba-NLP/new-impl--||g' /path/to/models/gte-large-en-v1.5/config.json`.

5. **Download the LLM**: `./setup_upgrade.sh download` or manually via `huggingface-cli`.

6. **Update scrapers**: Modify `scrape_srcc.py` / `scrape_static_docs.py` for your documentation sites, or remove them and populate `docs/` manually.

7. **Build and run**: `./setup_upgrade.sh rebuild && ./filemagic.sh && ./main.sh`

8. **Automate scraping** (optional): Install the systemd units from `systemd/`. See the Systemd Integration section above.

---

## Troubleshooting

### Model not loading / wrong model detected

```bash
ls /path/to/model/config.json
python3 -c "import json; print(json.load(open('/path/to/model/config.json'))['model_type'])"
```

vLLM detects the model architecture from `config.json`. AWQ models require `dtype: "half"` in `config.yaml`.

### Embedding model fails with `trust_remote_code` error

The gte-large-en-v1.5 model uses custom code (`modeling.py`, `configuration.py`). These must be in the model directory with local references in `config.json`. See "Adapting for a New Deployment" above.

### Startup hangs or crashes with "Device string must not be empty"

The pynvml shims in `sitecustomize.py` are not being loaded. Ensure `APPTAINERENV_PYTHONPATH=/workspace` is set.

### Startup hangs with "No available shared memory broadcast block"

Orphaned shared memory from a previous vLLM process. `main.sh` cleans this up automatically, but if running manually:

```bash
find /dev/shm -maxdepth 1 -user "$USER" -name "psm_*" -delete
```

### Cache returning stale responses

Content-aware invalidation runs automatically on restart after a scrape. To force a full flush:

```bash
curl -X POST http://localhost:8000/cache/clear
```

---

## Known Limitations

### CUDA Graphs disabled (`enforce_eager=True`)

vLLM runs with `enforce_eager=True`, disabling CUDA graph capture. This is required inside Apptainer due to a hard C++ assertion on `nvmlInit_v2_()` in PyTorch's CUDA caching allocator. Impact: ~5--15% higher per-token latency.

### NCCL socket transport

NCCL all-reduce uses TCP socket over loopback (`NCCL_P2P_DISABLE=1`, `NCCL_SHM_DISABLE=1`). P2P is blocked by IOMMU inside Apptainer; SHM has namespace collisions. Adds ~32 ms per forward pass. See `_load_model()` in `app/rag_service.py` for details and potential fixes.

---

## Support

For debugging:
1. Run `./comprehensive_test.sh` to pinpoint which component is failing
2. Check `logs/myapp.log`
3. Run `./gpu.sh` to verify GPU compute
4. Verify model path and `dtype` in `config.yaml`
