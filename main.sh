#!/bin/bash
 
# Variables
SIF_NAME="chatbot.sif"
SIF_DEF="chatbot.def"
MODEL_PATH="/oak/stanford/groups/ruthm/bcritt/.cache/huggingface/hub/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/e0bc86c23ce5aae1db576c8cca6f06f1f73af2db"
DATABASE_FILE=".langchain.db"
URL_MAP_PATH="/scratch/groups/bprogers/bcritt/url_map.txt"

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

# Verify Model Path Exists with Correct Permissions
echo "Ensuring the model path exists"
if [ ! -d $MODEL_PATH ]; then
    echo "Model path does not exist: $MODEL_PATH"
    exit 1
fi


# source filemagic.sh

echo "Apptaining from main.sh"

# Comment this out if you don't want to build the sif
 
   #apptainer build -F --nv $SIF_NAME $SIF_DEF

module load devel
module load cuda


apptainer exec --nv $SIF_NAME python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_arch_list()); torch.zeros(1).to('cuda')"


apptainer exec --nv $SIF_NAME python3 -c "import transformers; print(transformers.__version__)"

apptainer instance start  --nv $SIF_NAME chatapi

#apptainer exec --nv instance://chatapi python3 chat.py
apptainer exec --nv instance://chatapi uvicorn main:app --reload
#apptainer exec instance://chatapi fastapi run 
#apptainer exec instance://chatapi uvicorn robotchat:app

