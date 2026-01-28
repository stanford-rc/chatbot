#!/bin/bash
 
# Variables
SIF_NAME="file_processing.sif"
SIF_DEF="file_processing.def"
DOCS_DIR="docs"
export REPO_CLONE_DIR="docs"

# Ensure docs dir exists on host before starting container
echo "Ensuring docs directory exists: $DOCS_DIR"
mkdir -p "$DOCS_DIR"

echo "Apptaining from filemagic.sh"
# Comment this out if you don't want to build the sif

apptainer build $SIF_NAME $SIF_DEF

apptainer exec \
    --bind "$PWD:$PWD" \
    --pwd "$PWD" \
    $SIF_NAME \
    python3 file_magic.py
