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

SIF_NAME="chatbot.sif"
SIF_DEF="chatbot.def"
DATABASE_FILE=".langchain.db"
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

# ── Read settings from config.yaml ────────────────────────────────────────
MODEL_PATH=$(python3 -c "import yaml; c=yaml.safe_load(open('config.yaml')); print(c['model']['path'])")
PORT=$(python3 -c "import yaml; c=yaml.safe_load(open('config.yaml')); print(c.get('server',{}).get('api_port',8000))")
HOST=$(python3 -c "import yaml; c=yaml.safe_load(open('config.yaml')); print(c.get('server',{}).get('host','localhost'))")
LOG_DIR=$(python3 -c "import yaml; c=yaml.safe_load(open('config.yaml')); print(c.get('logging',{}).get('log_dir','logs'))")

echo "Using model from config.yaml: $MODEL_PATH"
echo "API port: $PORT, host: $HOST, log dir: $LOG_DIR"

# ── Housekeeping ──────────────────────────────────────────────────────────
echo "Stopping any existing chatapi instance..."
apptainer instance stop chatapi 2>/dev/null || true

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
    echo "Please check config.yaml and ensure the model path is correct"
    exit 1
fi

# ── Build SIF if not present ──────────────────────────────────────────────
if [ ! -f "$SIF_NAME" ]; then
    echo "Building container..."
    apptainer build "$SIF_NAME" "$SIF_DEF"
fi

# ── Prepare bind mounts ───────────────────────────────────────────────────
# $PWD → /workspace  (code, docs, DB files, sitecustomize.py)
BIND_MOUNTS="--bind $PWD:/workspace"
# Model directory (may be outside $PWD)
if [ -d "$MODEL_PATH" ]; then
    BIND_MOUNTS="$BIND_MOUNTS --bind $MODEL_PATH:$MODEL_PATH"
fi

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
    apptainer exec --nv instance://chatapi \
        /opt/chatbot-env/bin/python -m uvicorn app.main:app \
        --host 0.0.0.0 \
        --port "$PORT" \
        --log-level info
fi
