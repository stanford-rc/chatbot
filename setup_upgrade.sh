#!/bin/bash
# Setup script for chatbot improvements:
#   1. Rebuild container (picks up faiss-cpu from requirements.txt)
#   2. Download Llama 3.1 70B Instruct (4-bit AWQ) from HuggingFace
#   3. Tune MIN_BM25_SCORE with interactive queries against the running service
#
# Usage:
#   ./setup_upgrade.sh              # Run all steps
#   ./setup_upgrade.sh rebuild      # Only rebuild container
#   ./setup_upgrade.sh download     # Only download model
#   ./setup_upgrade.sh tune         # Only run BM25 tuning

set -e
cd "$(dirname "$0")"

# ─── Configuration ───────────────────────────────────────────────────────────
MODEL_NAME="ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4"
MODEL_DIR="$(dirname "$(python3 -c "import yaml; print(yaml.safe_load(open('config.yaml'))['model']['path'])")")"
MODEL_LOCAL_PATH="$MODEL_DIR/Meta-Llama-3.3-70B-Instruct-AWQ-INT4"

SIF_NAME="chatbot.sif"
SIF_DEF="chatbot.def"
ENDPOINT="http://localhost:8000/query/"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# ─── Step 1: Rebuild Container ──────────────────────────────────────────────
rebuild_container() {
    echo ""
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}STEP 1: Rebuild Container${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo "This rebuilds chatbot.sif to pick up new dependencies (faiss-cpu)."
    echo ""

    # Stop running instances
    echo "Stopping any running instances..."
    ./stop_multi_gpu.sh 2>/dev/null || true
    apptainer instance stop chatapi 2>/dev/null || true

    # Remove old SIF and rebuild
    if [ -f "$SIF_NAME" ]; then
        echo "Removing old $SIF_NAME..."
        rm "$SIF_NAME"
    fi

    echo "Building new container (this may take several minutes)..."
    apptainer build "$SIF_NAME" "$SIF_DEF"

    echo ""
    echo -e "${GREEN}✓ Container rebuilt successfully${NC}"

    # Verify faiss is available
    echo "Verifying faiss-cpu installation..."
    apptainer exec "$SIF_NAME" /opt/chatbot-env/bin/python -c "import faiss; print(f'faiss version: {faiss.__version__}')" 2>/dev/null \
        && echo -e "${GREEN}✓ faiss-cpu is available${NC}" \
        || echo -e "${YELLOW}⚠ faiss-cpu import failed — hybrid retrieval will fall back to BM25 only${NC}"
}

# ─── Step 2: Download Model ─────────────────────────────────────────────────
download_model() {
    echo ""
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}STEP 2: Download Llama 3.3 70B Instruct (4-bit AWQ)${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo "Model:       $MODEL_NAME"
    echo "Destination: $MODEL_LOCAL_PATH"
    echo "Size:        ~35 GB (fits on a single L40 with room for KV cache)"
    echo ""

    # Check if model already exists
    if [ -f "$MODEL_LOCAL_PATH/config.json" ]; then
        echo -e "${GREEN}✓ Model already downloaded at $MODEL_LOCAL_PATH${NC}"
        echo "  To re-download, delete the directory first."
    else
        # Check for huggingface-cli
        if ! command -v huggingface-cli &>/dev/null; then
            echo "Installing huggingface_hub CLI..."
            pip3 install --user huggingface_hub[cli]
        fi

        echo ""
        echo "NOTE: This model requires accepting the Llama 3.3 license on HuggingFace."
        echo "If you haven't already, visit: https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct"
        echo "and accept the license, then run: huggingface-cli login"
        echo ""
        read -r -p "Press Enter to start download (or Ctrl+C to cancel)..."

        mkdir -p "$MODEL_LOCAL_PATH"
        huggingface-cli download "$MODEL_NAME" \
            --local-dir "$MODEL_LOCAL_PATH" \
            --local-dir-use-symlinks False

        echo ""
        echo -e "${GREEN}✓ Model downloaded to $MODEL_LOCAL_PATH${NC}"
    fi

    # Update config.yaml with new model settings
    echo ""
    echo "Updating config.yaml..."

    python3 -c "
import yaml

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

old_path = config['model']['path']
old_type = config['model']['type']
old_quant = config['model']['use_quantization']

config['model']['path'] = '$MODEL_LOCAL_PATH'
config['model']['type'] = 'llama'
config['model']['use_quantization'] = True

with open('config.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

print(f'  model.path: {old_path} → $MODEL_LOCAL_PATH')
print(f'  model.type: {old_type} → llama')
print(f'  model.use_quantization: {old_quant} → True')
"

    echo ""
    echo -e "${GREEN}✓ config.yaml updated${NC}"
    echo ""
    echo -e "${YELLOW}NOTE: To revert to Gemma 2 9B, update config.yaml:${NC}"
    echo "  model.path: <original path>"
    echo "  model.type: gemma"
    echo "  model.use_quantization: false"
}

# ─── Step 3: BM25 Score Tuning ──────────────────────────────────────────────
tune_bm25() {
    echo ""
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}STEP 3: Tune MIN_BM25_SCORE${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo "This sends test queries to the running service and shows retrieval"
    echo "scores so you can decide on a good MIN_BM25_SCORE threshold."
    echo ""
    echo "Current threshold: $(python3 -c "import yaml; print(yaml.safe_load(open('config.yaml')).get('retrieval',{}).get('MIN_BM25_SCORE', 2.0))")"
    echo ""

    # Check if the service is running
    if ! curl -sf "$ENDPOINT" -X POST -H "Content-Type: application/json" \
         -d '{"query":"test","cluster":"sherlock"}' > /dev/null 2>&1; then
        # Try health check instead (the query might fail but service could be up)
        if ! curl -sf "http://localhost:8000/health" > /dev/null 2>&1; then
            echo -e "${RED}ERROR: Chatbot service is not running on port 8000.${NC}"
            echo "Start it first with: ./main.sh multi"
            exit 1
        fi
    fi

    echo "The service is running. Sending test queries..."
    echo "Watch the worker logs for BM25 scores while queries run."
    echo ""

    RESULTS_DIR="/tmp/bm25_tune_$(date +%s)"
    mkdir -p "$RESULTS_DIR"

    # Test queries: mix of things that should match docs well, match poorly, and not match at all
    declare -a QUERIES=(
        "How do I submit a batch job on Sherlock?"
        "What partitions are available on Sherlock?"
        "How do I transfer files to Oak storage?"
        "What is the meaning of life?"
        "How do I use Python on Farmshare?"
        "What are the memory limits per node?"
        "Can I run Docker containers on Sherlock?"
        "How do I check my storage quota?"
    )
    declare -a CLUSTERS=(
        "sherlock"
        "sherlock"
        "oak"
        "sherlock"
        "farmshare"
        "sherlock"
        "sherlock"
        "sherlock"
    )

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    printf "%-50s %-10s %-8s\n" "QUERY" "CLUSTER" "SOURCES"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    for i in "${!QUERIES[@]}"; do
        query="${QUERIES[$i]}"
        cluster="${CLUSTERS[$i]}"
        result_file="$RESULTS_DIR/query_${i}.json"

        curl -s -X POST "$ENDPOINT" \
            -H "Content-Type: application/json" \
            -d "{\"query\": \"$query\", \"cluster\": \"$cluster\"}" \
            > "$result_file"

        source_count=$(jq '.sources | length' "$result_file" 2>/dev/null || echo "?")
        printf "%-50s %-10s %-8s\n" "${query:0:50}" "$cluster" "$source_count"
    done

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "Full responses saved to: $RESULTS_DIR/"
    echo ""
    echo "To inspect a response:"
    echo "  jq -r '.answer' $RESULTS_DIR/query_0.json"
    echo ""
    echo "To see BM25 scores, check the worker logs:"
    LOG_DIR=$(python3 -c "import yaml; print(yaml.safe_load(open('config.yaml')).get('logging',{}).get('log_dir','logs'))")
    echo "  grep 'BM25' $LOG_DIR/worker1.log | tail -20"
    echo "  grep 'BM25' $LOG_DIR/worker2.log | tail -20"
    echo ""

    # Interactive threshold adjustment
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Adjust threshold?"
    echo "  - Lower value (e.g., 1.0) = more docs pass through, risk of noise"
    echo "  - Higher value (e.g., 3.0) = stricter filtering, may miss some relevant docs"
    echo "  - Set to 0.0 to disable filtering entirely"
    echo ""
    read -r -p "New MIN_BM25_SCORE (or Enter to keep current): " new_score

    if [ -n "$new_score" ]; then
        python3 -c "
import yaml

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

old = config.get('retrieval', {}).get('MIN_BM25_SCORE', 2.0)
config.setdefault('retrieval', {})['MIN_BM25_SCORE'] = float('$new_score')

with open('config.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

print(f'  MIN_BM25_SCORE: {old} → $new_score')
"
        echo -e "${GREEN}✓ config.yaml updated${NC}"
        echo ""
        echo -e "${YELLOW}Restart the service for this to take effect:${NC}"
        echo "  ./stop_multi_gpu.sh && ./main.sh multi"
    else
        echo "Keeping current threshold."
    fi
}

# ─── Main ────────────────────────────────────────────────────────────────────
case "${1:-all}" in
    rebuild)
        rebuild_container
        ;;
    download)
        download_model
        ;;
    tune)
        tune_bm25
        ;;
    all)
        rebuild_container
        download_model
        echo ""
        echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${CYAN}SETUP COMPLETE${NC}"
        echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo ""
        echo "Next steps:"
        echo "  1. Start the service:      ./main.sh multi"
        echo "  2. Tune BM25 threshold:    ./setup_upgrade.sh tune"
        echo "  3. Run integration tests:  ./comprehensive_test.sh"
        echo ""
        echo "To revert the model change, edit config.yaml and rebuild."
        ;;
    *)
        echo "Usage: $0 [rebuild|download|tune|all]"
        exit 1
        ;;
esac
