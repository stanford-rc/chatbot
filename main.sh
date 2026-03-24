#!/bin/bash
set -e

# Variables
SIF_NAME="chatbot.sif"
SIF_DEF="chatbot.def"
DATABASE_FILE=".langchain.db"
MODE=${1:-single}  # single or multi

# Read settings from centralized config.yaml
MODEL_PATH=$(python3 -c "import yaml; c=yaml.safe_load(open('config.yaml')); print(c['model']['path'])")
PORT=$(python3 -c "import yaml; c=yaml.safe_load(open('config.yaml')); print(c.get('server',{}).get('api_port',8000))")
HOST=$(python3 -c "import yaml; c=yaml.safe_load(open('config.yaml')); print(c.get('server',{}).get('host','localhost'))")
LOG_DIR=$(python3 -c "import yaml; c=yaml.safe_load(open('config.yaml')); print(c.get('logging',{}).get('log_dir','logs'))")
echo "Using model from config.yaml: $MODEL_PATH"
echo "API port: $PORT, host: $HOST, log dir: $LOG_DIR"

# Kill any old instances
echo "Stopping existing instances..."
apptainer instance stop chatapi 2>/dev/null || true

# Ensure Database File Exists
if [ ! -f $DATABASE_FILE ]; then
    touch $DATABASE_FILE
    chmod 666 $DATABASE_FILE
fi

# Verify Model Path exists on host
if [ -d "$MODEL_PATH" ]; then
    echo "✓ Found model at: $MODEL_PATH"
    # Check for required model files
    if [ ! -f "$MODEL_PATH/config.json" ]; then
        echo "⚠ WARNING: config.json not found in model directory"
    fi
else
    echo "❌ ERROR: Model path does not exist: $MODEL_PATH"
    echo "Please check config.yaml and ensure the model path is correct"
    exit 1
fi

# Build SIF if not present
if [ ! -f $SIF_NAME ]; then
    echo "Building container..."
    apptainer build $SIF_NAME $SIF_DEF
fi

# Prepare bind mounts
# Bind current directory as /workspace
BIND_MOUNTS="--bind $PWD:/workspace"

# Since model is now in ~/apichatbot/models/ which is under $PWD,
# it's already accessible. But we can add explicit bind for clarity:
if [ -d "$MODEL_PATH" ]; then
    BIND_MOUNTS="$BIND_MOUNTS --bind $MODEL_PATH:$MODEL_PATH"
fi

echo "Bind mounts: $BIND_MOUNTS"

# Start instance 
echo "Starting chatbot instance..."
apptainer instance start --nv $BIND_MOUNTS $SIF_NAME chatapi

# Quick check
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

# Verify model files are accessible inside container
echo "--- Model Files Check ---"
apptainer exec instance://chatapi ls -la "$MODEL_PATH" | head -10

# Check which mode to run
if [ "$MODE" = "multi" ]; then
    echo ""
    echo "=== Starting in MULTI-GPU mode ==="
    exec ./start_multi_gpu.sh
else
    echo ""
    echo "=== Starting in SINGLE-WORKER development mode ==="
    echo "Starting FastAPI server on port $PORT..."
    echo "Access at: http://$HOST:$PORT"
    echo "API docs: http://$HOST:$PORT/docs"
    echo ""

    APPTAINERENV_PYTHONPATH=/workspace \
    apptainer exec --nv instance://chatapi \
        /opt/chatbot-env/bin/python -m uvicorn app.main:app \
        --host 0.0.0.0 \
        --port $PORT \
        --reload --reload-dir app
fi
