#!/bin/bash
 
# Variables
SIF_NAME="file_processing.sif"
SIF_DEF="file_processing.def"
DOCS_DIR="docs"
export REPO_CLONE_DIR="docs"

# Ensure docs dir exists on host before starting container
echo "Ensuring docs directory exists: $DOCS_DIR"
mkdir -p "$DOCS_DIR"

if [ ! -f $SIF_NAME ]; then
    echo "Building container..."
    apptainer build $SIF_NAME $SIF_DEF
fi

apptainer exec \
    --bind "$PWD:$PWD" \
    --pwd "$PWD" \
    $SIF_NAME \
    python3 file_magic.py

apptainer exec \
    --bind "$PWD:$PWD" \
    --pwd "$PWD" \
    $SIF_NAME \
    python3 scrape_srcc.py

apptainer exec \
    --bind "$PWD:$PWD" \
    --pwd "$PWD" \
    $SIF_NAME \
    python3 scrape_static_docs.py

# Generate content manifests for cache invalidation.
# The RAG service compares these at startup to detect changed docs
# and selectively evict stale semantic cache entries.
echo ""
echo "Generating content manifests..."
python3 generate_manifests.py "$DOCS_DIR"
