#!/bin/bash

# ── Locate the app directory ───────────────────────────────────────────────
APP_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
export ADA_CONFIG="${ADA_CONFIG:-$APP_DIR/config.yaml}"

SIF_DEF="$APP_DIR/file_processing.def"
SIF_NAME=$(python3 -c "import yaml; c=yaml.safe_load(open('$ADA_CONFIG')); print(c.get('container',{}).get('file_processing_sif','$APP_DIR/file_processing.sif'))")

# ── GitHub token ──────────────────────────────────────────────────────────
# Read from the file referenced in config. Exported so file_magic.py picks
# it up via os.getenv("GITHUB_TOKEN").
GITHUB_TOKEN_FILE=$(python3 -c "import yaml; c=yaml.safe_load(open('$ADA_CONFIG')); print(c.get('github',{}).get('token_file',''))")
if [[ -n "$GITHUB_TOKEN_FILE" && -f "$GITHUB_TOKEN_FILE" ]]; then
    export GITHUB_TOKEN
    GITHUB_TOKEN=$(tr -d '[:space:]' < "$GITHUB_TOKEN_FILE")
    echo "✓ GitHub token loaded from $GITHUB_TOKEN_FILE"
else
    echo "⚠ No GitHub token found — private repos will prompt for credentials"
fi

# Docs dir: absolute path derived from APP_DIR so this script works from any CWD
DOCS_DIR="$APP_DIR/docs"
export REPO_CLONE_DIR="$DOCS_DIR"

# Ensure docs dir exists on host before starting container
echo "Ensuring docs directory exists: $DOCS_DIR"
mkdir -p "$DOCS_DIR"

if [ ! -f "$SIF_NAME" ]; then
    echo "Building container..."
    apptainer build "$SIF_NAME" "$SIF_DEF"
fi

apptainer exec \
    --bind "$APP_DIR:$APP_DIR" \
    --pwd "$APP_DIR" \
    "$SIF_NAME" \
    python3 file_magic.py

apptainer exec \
    --bind "$APP_DIR:$APP_DIR" \
    --pwd "$APP_DIR" \
    "$SIF_NAME" \
    python3 scrape_srcc.py

apptainer exec \
    --bind "$APP_DIR:$APP_DIR" \
    --pwd "$APP_DIR" \
    "$SIF_NAME" \
    python3 scrape_static_docs.py

# Generate content manifests for cache invalidation.
# The RAG service compares these at startup to detect changed docs
# and selectively evict stale semantic cache entries.
echo ""
echo "Generating content manifests..."
python3 "$APP_DIR/generate_manifests.py" "$DOCS_DIR"
