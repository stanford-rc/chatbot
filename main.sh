#!/bin/bash
set -e

# Variables
SIF_NAME="chatbot.sif"
SIF_DEF="chatbot.def"
DATABASE_FILE=".langchain.db"
PORT="8000"

# Read MODEL_PATH from centralized config.yaml
MODEL_PATH=$(python3 -c "import yaml; config = yaml.safe_load(open('config.yaml')); print(config['model']['path'])")
echo "Using model from config.yaml: $MODEL_PATH"

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
    BIND_MOUNTS="$BIND_MOUNTS --bind $MODEL_PATH:$MODEL_PATH:ro"
fi

echo "Bind mounts: $BIND_MOUNTS"

# Start instance (no --nv needed for CPU)
echo "Starting chatbot instance..."
apptainer instance start $BIND_MOUNTS $SIF_NAME chatapi

# Quick check
echo "--- Environment Check ---"
apptainer exec instance://chatapi python -c "
import torch
import os
print(f'PyTorch: {torch.__version__}')
print(f'Running on: CPU')
print('✓ Container started')
"

# Verify model files are accessible inside container
echo "--- Model Files Check ---"
apptainer exec instance://chatapi ls -la "$MODEL_PATH" | head -10

# Start FastAPI server on port $PORT
echo ""
echo "Starting FastAPI server on port $PORT..."
echo "Access at: http://ada-lovelace.stanford.edu:$PORT"
echo "API docs: http://ada-lovelace.stanford.edu:$PORT/docs"
echo ""

apptainer exec instance://chatapi \
    uvicorn app.main:app \
    --host 0.0.0.0 \
    --port $PORT

echo ""
echo "Server stopped"
