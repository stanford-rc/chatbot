#!/bin/bash
# main.sh — start the Stanford HPC chatbot.
#
# Usage:
#   ./main.sh           # production mode — single TP=2 vLLM worker, no auto-reload
#   ./main.sh dev       # dev mode — enables uvicorn --reload; model restarts on code change
#                       #            (slow: ~3-4 min per restart while model reloads)
#   ./main.sh multi     # DEPRECATED — incompatible with tensor_parallel_size=2, exits
#
# Architecture:
#   The model is loaded with tensor_parallel_size=2.  One vLLM worker occupies
#   both NVIDIA L4 GPUs via NCCL.  vLLM's async continuous-batching engine handles
#   all concurrent requests natively — no nginx load balancer is needed.
#   start_multi_gpu.sh is incompatible: it would launch a second independent worker
#   on already-occupied GPUs and OOM immediately.

set -e

# ── Locate the app and its config ─────────────────────────────────────────
# APP_DIR is the directory containing this script — works regardless of CWD.
APP_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"

# ADA_CONFIG: path to config.yaml.  Override to use a system-level config
# (e.g. /etc/ada-chatbot/config.yaml) without changing the app directory.
# Defaults to config.yaml alongside this script.
export ADA_CONFIG="${ADA_CONFIG:-$APP_DIR/config.yaml}"

if [[ ! -f "$ADA_CONFIG" ]]; then
    echo "❌ ERROR: config file not found: $ADA_CONFIG"
    echo "   Set ADA_CONFIG=/path/to/config.yaml to specify its location."
    exit 1
fi

echo "Config  : $ADA_CONFIG"
echo "App dir : $APP_DIR"

SIF_DEF="$APP_DIR/chatbot.def"
DATABASE_FILE="$APP_DIR/.langchain.db"
MODE=${1:-single}

# ── 'multi' mode is incompatible with TP=2 ────────────────────────────────
if [[ "$MODE" == "multi" ]]; then
    echo ""
    echo "ERROR: 'multi' mode is incompatible with tensor_parallel_size=2."
    echo ""
    echo "  rag_service._load_model() sets tensor_parallel_size=2, which occupies"
    echo "  both L4 GPUs inside a single vLLM worker via NCCL.  start_multi_gpu.sh"
    echo "  would attempt to start a second independent worker on the same GPUs"
    echo "  and OOM immediately (worker2 exits with 'Engine core initialization failed')."
    echo ""
    echo "  vLLM's built-in continuous batching handles all concurrent requests"
    echo "  natively at up to ~12 in-flight requests on this hardware."
    echo ""
    echo "  Just run:  ./main.sh"
    echo ""
    exit 1
fi

# ── Read settings from config ─────────────────────────────────────────────
_cfg() { python3 -c "import yaml; c=yaml.safe_load(open('$ADA_CONFIG')); print($1)"; }

MODEL_PATH=$(_cfg "c['model']['path']")
PORT=$(_cfg "c.get('server',{}).get('api_port',8000)")
HOST=$(_cfg "c.get('server',{}).get('host','localhost')")
LOG_DIR=$(_cfg "c.get('logging',{}).get('log_dir','$APP_DIR/logs')")
HF_HOME=$(_cfg "c.get('hf_home','/workspace/.hf_cache')")
SIF_NAME=$(_cfg "c.get('container',{}).get('sif_path','$APP_DIR/chatbot.sif')")

echo "Using model: $MODEL_PATH"
echo "API port: $PORT, host: $HOST, log dir: $LOG_DIR"

# ── Housekeeping ──────────────────────────────────────────────────────────
echo "Stopping any existing chatapi instance..."
apptainer instance stop chatapi 2>/dev/null || true

# Kill any stray vLLM/uvicorn processes that might hold GPU memory or shm
pkill -9 -f "EngineCore" 2>/dev/null || true
pkill -9 -f "vllm"       2>/dev/null || true
pkill -9 -f "uvicorn app.main" 2>/dev/null || true
sleep 2

# Clean up orphaned POSIX shared memory left by SIGKILL'd vLLM processes.
# Python's multiprocessing.shared_memory writes files to /dev/shm/ (psm_*).
# If vLLM is killed with -9 those files are never deleted; the next startup
# finds all broadcast blocks "in use" and hangs for minutes before erroring.
if find /dev/shm -maxdepth 1 -user "$USER" -name "psm_*" 2>/dev/null | grep -q .; then
    echo "Cleaning up orphaned shared memory blocks from /dev/shm ..."
    find /dev/shm -maxdepth 1 -user "$USER" -delete 2>/dev/null || true
    echo "  ✓ Done"
fi

# Ensure log directory and LangChain DB file exist on the host before binding
mkdir -p "$LOG_DIR"
if [ ! -f "$DATABASE_FILE" ]; then
    touch "$DATABASE_FILE"
    chmod 666 "$DATABASE_FILE"
fi

# ── Verify model path ─────────────────────────────────────────────────────
if [ -d "$MODEL_PATH" ]; then
    echo "✓ Found model at: $MODEL_PATH"
    if [ ! -f "$MODEL_PATH/config.json" ]; then
        echo "⚠ WARNING: config.json not found in model directory"
    fi
else
    echo "❌ ERROR: Model path does not exist: $MODEL_PATH"
    echo "Please check $ADA_CONFIG and ensure the model path is correct"
    exit 1
fi

# ── Build SIF if not present ──────────────────────────────────────────────
if [ ! -f "$SIF_NAME" ]; then
    echo "Building container..."
    apptainer build "$SIF_NAME" "$SIF_DEF"
fi

# ── Prepare bind mounts ───────────────────────────────────────────────────
# APP_DIR always maps to /workspace (code + sitecustomize.py).
# Additional mounts are auto-generated from every absolute path in config:
#   model dir, doc dirs, data_dir (cache DBs, logs, HF cache), config dir.
# This ensures the container can see all referenced paths regardless of
# where they live on the host filesystem.
BIND_MOUNTS=$(python3 - "$APP_DIR" "$ADA_CONFIG" <<'PYEOF'
import sys, os, yaml

app_dir  = sys.argv[1]
cfg_path = sys.argv[2]

with open(cfg_path) as f:
    c = yaml.safe_load(f)

# data_dir: where /workspace/ paths live on the host.
# Empty or absent → use app_dir (same as the /workspace mount).
data_dir = c.get('data_dir', '') or app_dir

cfg_dir = os.path.dirname(os.path.abspath(cfg_path))

def resolve(path):
    """Mirrors config.py _resolve(): remap /workspace/ prefix to data_dir,
    then make relative paths absolute from the config file's directory."""
    if not path:
        return path
    path = str(path)
    if path == '/workspace':
        return data_dir
    if path.startswith('/workspace/'):
        return os.path.join(data_dir, path[len('/workspace/'):])
    if not os.path.isabs(path):
        return os.path.join(cfg_dir, path)
    return path

dirs = set()

def add(path):
    path = resolve(path)
    if not path:
        return
    # Use parent dir for file paths (has an extension and isn't an existing dir)
    d = path if os.path.isdir(path) or not os.path.splitext(path)[1] else os.path.dirname(path)
    if d and d != '/':
        # Create the directory on the host if it doesn't exist so Apptainer
        # can bind-mount it without failing.
        os.makedirs(d, exist_ok=True)
        dirs.add(d)

# Model
add(c['model']['path'])

# Docs
for p in c.get('clusters', {}).values():
    add(p)
add(c.get('shared_docs', ''))

# Caching DBs
add(c.get('caching', {}).get('SEMANTIC_CACHE_DB', ''))
add(c.get('caching', {}).get('LANGCHAIN_CACHE_DB', ''))

# Logs
add(c.get('logging', {}).get('log_dir', ''))
add(c.get('logging', {}).get('stats_log', ''))

# HF cache
add(c.get('hf_home', ''))

# Explicit data_dir
add(c.get('data_dir', ''))

# Config file dir (if outside app_dir)
add(cfg_dir)

# Build args: APP_DIR → /workspace, everything else maps to itself
parts = [f'--bind {app_dir}:/workspace']
for d in sorted(dirs):
    try:
        if os.path.commonpath([d, app_dir]) == app_dir:
            continue  # inside app_dir, already covered by /workspace mount
    except ValueError:
        pass  # different drives on Windows; shouldn't happen on Linux
    parts.append(f'--bind {d}:{d}')

print(' '.join(parts))
PYEOF
)

# ── Resolve ADA_CONFIG path as seen inside the container ─────────────────
# The config file's directory is already in BIND_MOUNTS (either covered by
# the APP_DIR→/workspace mount or bound as itself).  Translate the host path
# to the equivalent in-container path so the app can find it.
if [[ "$ADA_CONFIG" == "$APP_DIR"/* ]]; then
    # Config lives inside the app dir — it appears under /workspace inside the container
    CONTAINER_ADA_CONFIG="/workspace/${ADA_CONFIG#"$APP_DIR/"}"
else
    # Config is outside the app dir — bound as itself, path unchanged
    CONTAINER_ADA_CONFIG="$ADA_CONFIG"
fi
echo "Container config path: $CONTAINER_ADA_CONFIG"

echo "Bind mounts: $BIND_MOUNTS"

# ── Start Apptainer instance ──────────────────────────────────────────────
echo "Starting chatapi instance..."
apptainer instance start --nv $BIND_MOUNTS "$SIF_NAME" chatapi

# ── Quick sanity check ────────────────────────────────────────────────────
echo "--- Environment Check ---"
apptainer exec --nv instance://chatapi /opt/chatbot-env/bin/python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
print('✓ Container started')
"

echo "--- Model Files Check ---"
apptainer exec instance://chatapi ls -la "$MODEL_PATH" | head -10

# ── Launch uvicorn ────────────────────────────────────────────────────────
echo ""
if [[ "$MODE" == "dev" ]]; then
    echo "=== Starting in DEV mode (auto-reload enabled) ==="
    echo "  WARNING: each code change triggers a full model reload (~3-4 min)."
    echo "  Use production mode (./main.sh) for normal operation."
    echo ""
    echo "Access at : http://$HOST:$PORT"
    echo "API docs  : http://$HOST:$PORT/docs"
    echo ""
    APPTAINERENV_PYTHONPATH=/workspace \
    APPTAINERENV_HF_HOME="$HF_HOME" \
    APPTAINERENV_ADA_CONFIG="$CONTAINER_ADA_CONFIG" \
    APPTAINERENV_VLLM_USE_V1=0 \
    apptainer exec --nv instance://chatapi \
        /opt/chatbot-env/bin/python -m uvicorn app.main:app \
        --host 0.0.0.0 \
        --port "$PORT" \
        --log-level info \
        --reload --reload-dir app
else
    echo "=== Starting in PRODUCTION mode (single TP=2 worker) ==="
    echo "  vLLM tensor_parallel_size=2 — both L4 GPUs will be used."
    echo "  Model loading takes ~3-4 minutes.  Watch for 'Application startup complete'."
    echo ""
    echo "Access at : http://$HOST:$PORT"
    echo "API docs  : http://$HOST:$PORT/docs"
    echo ""
    APPTAINERENV_PYTHONPATH=/workspace \
    APPTAINERENV_HF_HOME="$HF_HOME" \
    APPTAINERENV_ADA_CONFIG="$CONTAINER_ADA_CONFIG" \
    APPTAINERENV_VLLM_USE_V1=0 \
    apptainer exec --nv instance://chatapi \
        /opt/chatbot-env/bin/python -m uvicorn app.main:app \
        --host 0.0.0.0 \
        --port "$PORT" \
        --log-level info
fi
