# Chatbot Upgrade — Session Context
*Last updated: 2026-03-20*

---

## What This Is
RAG-based documentation chatbot for Stanford Research Computing clusters (Sherlock, Farmshare, Oak, Elm). FastAPI backend, multi-GPU workers behind a load balancer, served via Apptainer container on `ada-lovelace.stanford.edu`.

**Primary goal:** Make it answer from cluster documentation first, not generic HPC knowledge. Stanford's clusters have custom SLURM configs, tools (`sdev`), and policies that generic knowledge gets wrong.

---

## Server Environment
- Host: `ada-lovelace.stanford.edu`
- Working dir: `/srv/scratch/bcritt/chatbot`
- Container: `chatbot.sif` (Apptainer, built from `chatbot.def`)
- Python env inside container: `/opt/chatbot-env` (managed by `uv`)
- GPUs: 2× NVIDIA L4 (24GB each) — **ARM64 server** (sbsa architecture)
- CUDA: 13.0 (`cu130`)
- PyTorch: 2.10.0+cu130
- HuggingFace auth: active in `chatbot2` uv venv on host (not inside container)
- Model storage: `/home/users/bcritt/apichatbot/models/`

---

## Current Model
**`Qwen/Qwen2.5-32B-Instruct-AWQ`** — downloaded to:
`/home/users/bcritt/apichatbot/models/Qwen2.5-32B-Instruct-AWQ`

- ~19GB at AWQ-INT4 — fits on a single L4 with ~5GB headroom
- First-party quantization from Qwen team
- 128K context window, excellent structured output / citation behavior
- IFEval: 79.5% (strong instruction following)

**Previous model:** Gemma 2 9B IT (Sara's path, `/home/users/saracook/...`) — now replaced in config.

### Why we chose Qwen 2.5 32B over Llama 3.3 70B
Llama 3.3 70B AWQ = ~35GB, doesn't fit on a single L4 (24GB). Qwen 2.5 32B AWQ = ~19GB, fits with headroom. Would have selected Gemma 3 27B (IFEval 90.4%) but its VRAM is framework-dependent and it needs TorchAO backend.

---

## Inference Stack: vLLM
**Why vLLM:** The ARM64 server cannot compile ExLlama v2 CUDA kernels that both `autoawq` and `gptqmodel` require. vLLM ships pre-built ARM64+CUDA kernels in its wheel. It's also the inference standard autoawq itself recommends (their deprecation notice points to vLLM).

**Evolution of attempts:**
1. `autoawq` → incompatible with transformers 5.x (which dropped autoawq)
2. `gptqmodel` → needs `--no-build-isolation` to find torch during build; ExLlama v2 AWQ kernels still fail on ARM64
3. **`vllm`** ✅ → current approach, ARM64-native, handles AWQ internally

---

## What Was Done This Session

### Code Changes (all committed/pushed)
| File | Change |
|------|--------|
| `app/rag_service.py` | Replaced transformers model loading + generate loop with vLLM `LLM` + `SamplingParams`; GPU pinning via `CUDA_VISIBLE_DEVICES` (set before vLLM init); removed `torch` import |
| `app/rag_service.py` | `device_map` now pins to `WORKER_GPU` env var (was always `"auto"`, causing both workers to race for both GPUs) |
| `app/rag_service.py` | `_identify_cluster` uses `\b` word boundaries (was substring, risked false matches) |
| `app/load_balancer.py` | Fixed worker selection race condition with `asyncio.Lock` (`get_and_claim_worker`) |
| `app/main.py` | Health check now also verifies `rag_service.retrievers` is non-empty |
| `app/semantic_cache.py` | Added `PRAGMA journal_mode=WAL` + `busy_timeout=5000` for safe concurrent SQLite access from 2 workers |
| `app/prompts.py` | Updated comment to clarify Qwen uses else branch; `apply_chat_template` in rag_service handles the ChatML wrapping |
| `config.yaml` | Model path → Qwen2.5-32B-Instruct-AWQ; type → `qwen`; device → `cuda`; max_new_tokens → 512; hybrid retrieval enabled; grounding check enabled |
| `requirements.txt` | Added `vllm`, `tiktoken`, `faiss-cpu`, `autoawq` (later removed), `gptqmodel` (later removed); pinned `transformers>=5.0.0` |
| `chatbot.def` | Removed gptqmodel special install step |
| `start_multi_gpu.sh` | Health check timeout 60→120 attempts (5→10 min) |
| `main.sh` | GPU check is now dynamic (`torch.cuda.is_available()`) not hardcoded "CPU" |

### Infrastructure / Process
- `setup_upgrade.sh` created — handles container rebuild, model download, BM25 tuning
- Model download: `hf download Qwen/Qwen2.5-32B-Instruct-AWQ --local-dir /home/users/bcritt/apichatbot/models/Qwen2.5-32B-Instruct-AWQ`
- Old Llama 3.3 70B download was in progress at `/home/users/bcritt/apichatbot/models/Meta-Llama-3.3-70B-Instruct-AWQ-INT4` — can be deleted to reclaim space

---

## State When User Left
Workers were failing with `RuntimeError: Device string must not be empty` from vLLM's platform detection. Root cause: NVML cannot initialize in Apptainer (`Can't initialize NVML`), so `current_platform.device_type` returns empty string. `APPTAINERENV_VLLM_PLATFORM=cuda` was added to launch scripts but didn't resolve it (env var injection through the exec chain wasn't reaching vLLM at import time).

**Fix applied (needs rebuild):** Added `VLLM_PLATFORM=cuda` and `HF_HUB_OFFLINE=1` directly to `chatbot.def` `%environment` section. This bakes both vars into the container itself — no reliance on `APPTAINERENV_` prefix mechanics. After pushing, rebuild on the cluster:
```bash
apptainer instance stop chatapi 2>/dev/null || true
rm chatbot.sif
apptainer build chatbot.sif chatbot.def
./main.sh multi
```

### Most Likely Issues to Hit Next Session
1. **Container rebuild required** — `chatbot.def` was updated (VLLM_PLATFORM baked in). Must rebuild before workers will start.
2. **vLLM install** — handled. vLLM 0.18.0+cu130 installed from GitHub release wheel in chatbot.def. **Do not add vllm to requirements.txt.**
3. **vLLM + multi-worker conflict** — vLLM uses its own process pool internally. Two separate uvicorn processes each running a `LLM()` instance should be fine since `CUDA_VISIBLE_DEVICES` isolates them, but watch worker2.log for NCCL/CUDA errors.
4. **Qwen model** — verify complete: `ls -lh /home/users/bcritt/apichatbot/models/Qwen2.5-32B-Instruct-AWQ/` should show ~9 safetensors files totaling ~19GB.
5. **`config.yaml` model path** — git pulls keep resetting to Sara's Gemma path. After pull, verify: `grep path: config.yaml` should show Qwen path. Fix if needed: `sed -i 's|path:.*|path: /home/users/bcritt/apichatbot/models/Qwen2.5-32B-Instruct-AWQ|' config.yaml`

### Diagnostic Commands
```bash
# Check model files are complete
ls -lh /home/users/bcritt/apichatbot/models/Qwen2.5-32B-Instruct-AWQ/

# Check config is correct
grep -E "path:|type:|device:" /srv/scratch/bcritt/chatbot/config.yaml

# Check if vllm is in the container
apptainer exec instance://chatapi /opt/chatbot-env/bin/python -c "import vllm; print(vllm.__version__)"

# Watch worker logs in real time
tail -f /srv/scratch/bcritt/chatbot/logs/worker1.log
tail -f /srv/scratch/bcritt/chatbot/logs/worker2.log

# Manual quick test once workers are up
curl -s -X POST http://localhost:8001/query/ \
  -H "Content-Type: application/json" \
  -d '{"query":"How do I submit a batch job on Sherlock?","cluster":"sherlock"}' | python3 -m json.tool
```

---

## Outstanding Work (Not Yet Done)

### From Original Plan
- **BM25 tuning** (`./setup_upgrade.sh tune`) — needs to be run once workers are healthy to dial in `MIN_BM25_SCORE`. Currently set to `2.0` in config.yaml; run tune to see actual scores against your docs. Set to `0.0` to disable if scores are consistently low.
- **Latency benchmark** — 32B model at AWQ on L4 will be slower than 9B Gemma. Measure tokens/sec and decide if acceptable. Target ~12 tok/s per worker via vLLM (vLLM is significantly faster than transformers generate() for this model size).

### Known Remaining Issues
- **Old Llama model directory mess** — `/home/users/bcritt/apichatbot/models/Meta-Llama-3.3-70B-Instruct-AWQ-INT4/` contains an HF cache from `all-MiniLM-L6-v2` accidentally written there (when HF_HOME pointed to the wrong path). Safe to delete the whole directory.
- **Stray `Qwen2.5-32B-Instruct-AWQ/` in working dir** — caused by `setup_upgrade.sh` running when `config.yaml` had the old Gemma/Llama model path; `dirname` of a relative or wrong path returns `.`, so the download landed in `/srv/scratch/bcritt/chatbot/Qwen2.5-32B-Instruct-AWQ/`. Safe to delete: `rm -rf /srv/scratch/bcritt/chatbot/Qwen2.5-32B-Instruct-AWQ`. A path guard was added to `setup_upgrade.sh` to prevent recurrence.
- **`HF_HUB_OFFLINE=1`** — now baked into container via chatbot.def `%environment`. No HF network calls from inside the container after rebuild.
- **Semantic cache WAL mode** — only applies to new connections after the PRAGMA. If the DB was created before this change, `PRAGMA journal_mode=WAL` needs to run once against the existing file: `sqlite3 .response_cache.db "PRAGMA journal_mode=WAL;"`

---

## Key Files
```
chatbot/
├── main.sh                  # Single-GPU / dev mode launcher
├── start_multi_gpu.sh       # Multi-GPU launcher (2 workers + load balancer)
├── stop_multi_gpu.sh        # Teardown
├── setup_upgrade.sh         # Rebuild / download / BM25 tune
├── chatbot.def              # Apptainer container definition
├── requirements.txt         # Python dependencies
├── config.yaml              # All runtime config (model, retrieval, caching, etc.)
└── app/
    ├── main.py              # FastAPI worker app + health check
    ├── load_balancer.py     # Request router (round-robin with async lock)
    ├── rag_service.py       # Core: model loading, retrieval, generation, post-processing
    ├── prompts.py           # System instructions + chat templates
    ├── config.py            # Settings class (reads config.yaml + env vars)
    ├── semantic_cache.py    # SQLite-backed semantic response cache
    └── models.py            # Pydantic request/response models
```
