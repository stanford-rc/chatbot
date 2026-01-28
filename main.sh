#!/bin/bash
 
# Variables
SIF_NAME="chatbot.sif"
SIF_DEF="chatbot.def"
MODEL_PATH=".cache/huggingface/hub"
DATABASE_FILE=".langchain.db"
URL_MAP_PATH="docs/url_map.txt"

# Remove Existing Singularity Images
# if [ -f $SIF_NAME ]; then
#    rm -f $SIF_NAME
# fi

# kill any old instances
apptainer instance stop --all

# Ensure Database File Exists with Correct Permissions
echo "Ensuring the database file exists"
if [ ! -f $DATABASE_FILE ]; then
    touch $DATABASE_FILE
fi

# Verify Model Path or use HuggingFace download
if [ -d "$MODEL_PATH" ]; then
    echo "Using local model at: $MODEL_PATH"
else
    echo "Local model not found at: $MODEL_PATH"
    echo "Model will download from HuggingFace using MODEL_PATH ID"
fi

echo "Refreshing documentation repositories..."
bash filemagic.sh

# Uncomment these if you want to verify cuda/pytorch works:
# apptainer exec --nv $SIF_NAME python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
# apptainer exec --nv $SIF_NAME python3 -c "import transformers; print(transformers.__version__)"

# Start chatbot instance WITH docs bind mount
echo "Starting chatbot instance..."
apptainer instance start \
    --nv \
    --bind "$PWD:$PWD" \
    $SIF_NAME chatapi

# Run your FastAPI app
echo "Starting FastAPI server..."
apptainer exec --nv instance://chatapi uvicorn app.main:app --reload --reload-dir app
