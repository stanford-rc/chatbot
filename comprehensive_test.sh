#!/bin/bash
# Comprehensive test suite for the Stanford HPC documentation chatbot.
# Tests: connectivity, citation quality, grounding/hallucination guard,
#        cross-cluster isolation, semantic cache, and response latency.
#
# Usage:  ./comprehensive_test.sh [--port PORT] [--host HOST] [--no-cache-clear]

set -euo pipefail

# ── Config ─────────────────────────────────────────────────────────────────
# Default: worker1 direct port (8001). The load balancer sits on api_port (8000)
# but may not be running in single-worker / dev mode.
PORT=$(python3 -c "
import yaml
try:
    c = yaml.safe_load(open('config.yaml'))
    workers = c.get('workers', [])
    print(workers[0]['port'] if workers else c.get('server',{}).get('api_port', 8001))
except Exception:
    print(8001)
" 2>/dev/null || echo 8001)
HOST="localhost"

# Parse optional overrides
while [[ $# -gt 0 ]]; do
    case $1 in
        --port) PORT="$2"; shift 2 ;;
        --host) HOST="$2"; shift 2 ;;
        --no-cache-clear) NO_CACHE_CLEAR=1; shift ;;
        *) shift ;;
    esac
done

ENDPOINT="http://$HOST:$PORT"
RESULTS_DIR="/tmp/chatbot_test_$(date +%s)"
mkdir -p "$RESULTS_DIR"

PASS=0; FAIL=0; WARN=0

GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'

pass() { echo -e "${GREEN}✓${NC} $*"; ((PASS++)); }
fail() { echo -e "${RED}✗${NC} $*"; ((FAIL++)); }
warn() { echo -e "${YELLOW}⚠${NC}  $*"; ((WARN++)); }
section() { echo ""; echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"; echo -e "${CYAN}$*${NC}"; echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"; }

query() {
    # query <outfile> <cluster> <question>
    local out="$1" cluster="$2" question="$3"
    curl -s --max-time 120 -X POST "$ENDPOINT/query/" \
        -H "Content-Type: application/json" \
        -d "{\"query\": \"$question\", \"cluster\": \"$cluster\"}" \
        -o "$out" -w "%{http_code}" 2>/dev/null
}

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║      STANFORD HPC CHATBOT — COMPREHENSIVE TEST       ║"
echo "╚══════════════════════════════════════════════════════╝"
echo "  Endpoint : $ENDPOINT"
echo "  Results  : $RESULTS_DIR"
echo ""


# ── Test 0: Connectivity ───────────────────────────────────────────────────
section "TEST 0 — Connectivity & Health"

http_code=$(curl -s -o "$RESULTS_DIR/health.json" -w "%{http_code}" "$ENDPOINT/health")
if [[ "$http_code" == "200" ]]; then
    pass "Health endpoint returned 200"
    clusters=$(jq -r '.clusters | join(", ")' "$RESULTS_DIR/health.json" 2>/dev/null || echo "unknown")
    pass "Available clusters: $clusters"
else
    fail "Health endpoint returned $http_code — is the server up?"
    echo "  Run:  tail -f logs/worker1.log"
    exit 1
fi


# ── Test 1: Basic RAG — does it answer at all? ─────────────────────────────
section "TEST 1 — Basic RAG Response Quality"

http=$(query "$RESULTS_DIR/basic.json" "sherlock" "How do I submit a batch job on Sherlock?")
if [[ "$http" == "200" ]]; then
    pass "POST /query/ returned 200"
else
    fail "POST /query/ returned $http"; exit 1
fi

answer=$(jq -r '.answer' "$RESULTS_DIR/basic.json")
answer_len=${#answer}

if [[ $answer_len -gt 200 ]]; then
    pass "Answer length is substantive ($answer_len chars)"
else
    fail "Answer too short ($answer_len chars) — model may not have loaded"
fi

source_count=$(jq '.sources | length' "$RESULTS_DIR/basic.json")
if [[ $source_count -ge 1 ]]; then
    pass "At least one source returned ($source_count)"
else
    warn "No sources returned — retriever may not be finding documents"
fi

echo ""
echo "  Answer preview:"
echo "$answer" | fold -s -w 80 | head -5 | sed 's/^/    /'
echo "    ..."


# ── Test 2: Citation Format ────────────────────────────────────────────────
section "TEST 2 — Citation Quality"
# Inline markdown links: [Title](https://...)
# No raw .md filenames: [something.md]

query "$RESULTS_DIR/citation.json" "sherlock" \
    "What storage systems are available on Sherlock and how much space do I get?" \
    > /dev/null

answer=$(jq -r '.answer' "$RESULTS_DIR/citation.json")

if echo "$answer" | grep -qE '\[.+\]\(https?://[^)]+\)'; then
    pass "Inline citations with URLs present  [Title](https://...)"
else
    warn "No inline markdown URL citations found — check prompt template"
fi

if echo "$answer" | grep -qE '\[[a-zA-Z0-9_-]+\.md\]'; then
    fail "Raw .md filenames appear as citations — model citing filenames, not titles"
else
    pass "No raw .md filename citations"
fi

stanford_url=$(echo "$answer" | grep -oE 'https?://[^)]+stanford\.edu[^)]*' | head -1 || true)
if [[ -n "$stanford_url" ]]; then
    pass "Stanford URL found in citation: $stanford_url"
else
    warn "No *.stanford.edu URL found in inline citations"
fi

echo ""
echo "  Sources returned by API:"
jq -r '.sources[] | "    • \(.title)  \(.url // "(no url)")"' "$RESULTS_DIR/citation.json"


# ── Test 3: Grounding Guard ────────────────────────────────────────────────
section "TEST 3 — Grounding / Hallucination Guard"
# Send a vague or off-topic question that is unlikely to retrieve any
# matching docs.  If GROUNDING_CHECK_ENABLED, the answer should include
# the srcc-support disclaimer when the model goes off-script.

query "$RESULTS_DIR/grounding.json" "sherlock" \
    "What is the airspeed velocity of an unladen swallow?" \
    > /dev/null

answer_g=$(jq -r '.answer' "$RESULTS_DIR/grounding.json")
sources_g=$(jq '.sources | length' "$RESULTS_DIR/grounding.json")

echo "  Off-topic question sources retrieved: $sources_g"
echo "  Answer preview:"
echo "$answer_g" | fold -s -w 80 | head -4 | sed 's/^/    /'
echo ""

if echo "$answer_g" | grep -qi "srcc-support\|verify with\|may not reflect"; then
    pass "Grounding disclaimer present — model flagged ungrounded answer"
else
    warn "No grounding disclaimer — check GROUNDING_CHECK_ENABLED in config.yaml"
fi

# Now test a cluster-specific but retrievable question — should NOT have disclaimer
query "$RESULTS_DIR/grounding_ok.json" "sherlock" \
    "How do I run a GPU job on Sherlock?" \
    > /dev/null

answer_ok=$(jq -r '.answer' "$RESULTS_DIR/grounding_ok.json")
if echo "$answer_ok" | grep -qi "srcc-support\|may not reflect"; then
    warn "Grounding disclaimer appeared on a well-grounded answer — threshold may be too low"
else
    pass "No false-positive grounding disclaimer on a well-grounded answer"
fi


# ── Test 4: Cross-Cluster Isolation ───────────────────────────────────────
section "TEST 4 — Cross-Cluster Isolation"
# Ask the same question to two different clusters.
# Answers should reference the correct cluster name and not bleed across.

query "$RESULTS_DIR/sherlock_q.json" "sherlock" \
    "What is the default job time limit?" > /dev/null
query "$RESULTS_DIR/farmshare_q.json" "farmshare" \
    "What is the default job time limit?" > /dev/null

s_cluster=$(jq -r '.cluster' "$RESULTS_DIR/sherlock_q.json")
f_cluster=$(jq -r '.cluster' "$RESULTS_DIR/farmshare_q.json")

[[ "$s_cluster" == "sherlock"  ]] && pass "Sherlock response tagged cluster=sherlock"  || fail "Sherlock response cluster field = $s_cluster"
[[ "$f_cluster" == "farmshare" ]] && pass "Farmshare response tagged cluster=farmshare" || fail "Farmshare response cluster field = $f_cluster"

s_answer=$(jq -r '.answer' "$RESULTS_DIR/sherlock_q.json")
f_answer=$(jq -r '.answer' "$RESULTS_DIR/farmshare_q.json")

if [[ "$s_answer" == "$f_answer" ]]; then
    warn "Sherlock and Farmshare returned identical answers — retrievers may be sharing an index"
else
    pass "Sherlock and Farmshare returned distinct answers"
fi


# ── Test 5: Semantic Cache ─────────────────────────────────────────────────
section "TEST 5 — Semantic Cache Performance"

if [[ -z "${NO_CACHE_CLEAR:-}" ]]; then
    read -r -p "  Clear semantic cache before this test? [y/N] " resp
    if [[ "$resp" =~ ^[Yy]$ ]]; then
        rm -f .response_cache.db
        echo "  Cache cleared."
    fi
fi

echo "  Q1: fresh question..."
t1_start=$SECONDS
query "$RESULTS_DIR/cache1.json" "sherlock" \
    "How do I check my remaining CPU hours on Sherlock?" > /dev/null
t1=$(( SECONDS - t1_start ))

sleep 1

echo "  Q2: exact repeat (should be instant)..."
t2_start=$SECONDS
query "$RESULTS_DIR/cache2.json" "sherlock" \
    "How do I check my remaining CPU hours on Sherlock?" > /dev/null
t2=$(( SECONDS - t2_start ))

sleep 1

echo "  Q3: semantically similar wording..."
t3_start=$SECONDS
query "$RESULTS_DIR/cache3.json" "sherlock" \
    "How can I see how many CPU hours I have left on Sherlock?" > /dev/null
t3=$(( SECONDS - t3_start ))

echo ""
echo "  Latency — Q1 (fresh): ${t1}s   Q2 (exact cache): ${t2}s   Q3 (semantic cache): ${t3}s"
echo ""

if [[ $t2 -lt 3 ]]; then
    pass "Exact cache hit: ${t2}s (< 3s)"
else
    fail "Exact cache too slow: ${t2}s — cache may not be working"
fi

ans1=$(jq -r '.answer' "$RESULTS_DIR/cache1.json")
ans2=$(jq -r '.answer' "$RESULTS_DIR/cache2.json")
ans3=$(jq -r '.answer' "$RESULTS_DIR/cache3.json")

[[ "$ans1" == "$ans2" ]] && pass "Exact cache: answers identical" || fail "Exact cache: answers differ"
[[ "$ans1" == "$ans3" ]] && pass "Semantic cache: answers identical" || \
    warn "Semantic cache: answers differ (may be below similarity threshold)"

echo ""
echo "  Baseline latency for fresh query: ${t1}s"
if [[ $t1 -lt 10 ]]; then
    pass "First-token latency excellent (${t1}s)"
elif [[ $t1 -lt 30 ]]; then
    pass "First-token latency acceptable (${t1}s)"
else
    warn "First-token latency high (${t1}s) — consider reducing max_new_tokens or checking GPU util"
fi


# ── Test 6: Response latency / throughput ─────────────────────────────────
section "TEST 6 — Concurrent Request Throughput"
echo "  Sending 3 concurrent requests across clusters..."

t_par_start=$SECONDS
query "$RESULTS_DIR/par1.json" "sherlock"  "How do I use modules on Sherlock?"    &
query "$RESULTS_DIR/par2.json" "farmshare" "How do I connect to Farmshare?"       &
query "$RESULTS_DIR/par3.json" "oak"       "How do I transfer files to Oak?"      &
wait
t_par=$(( SECONDS - t_par_start ))

all_ok=true
for f in par1 par2 par3; do
    [[ -s "$RESULTS_DIR/$f.json" ]] || { fail "$f.json empty — request failed"; all_ok=false; }
done
$all_ok && pass "All 3 concurrent requests returned responses in ${t_par}s total"


# ── Summary ────────────────────────────────────────────────────────────────
section "SUMMARY"
total=$((PASS + FAIL + WARN))
echo "  Passed : $PASS / $total"
echo "  Failed : $FAIL"
echo "  Warnings: $WARN"
echo ""
echo "  Full results: $RESULTS_DIR/"
echo "  Inspect with:"
echo "    jq . $RESULTS_DIR/basic.json"
echo "    jq -r '.answer' $RESULTS_DIR/grounding.json"
echo ""

if [[ $FAIL -gt 0 ]]; then
    echo -e "${RED}RESULT: $FAIL test(s) FAILED${NC}"
    exit 1
elif [[ $WARN -gt 0 ]]; then
    echo -e "${YELLOW}RESULT: All tests passed with $WARN warning(s)${NC}"
else
    echo -e "${GREEN}RESULT: All tests passed ✓${NC}"
fi
