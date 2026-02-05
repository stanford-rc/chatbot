#!/bin/bash
# Comprehensive Test: Multi-GPU, Citations, and Semantic Caching

set -e
ENDPOINT="http://localhost:8000/query/"
RESULTS_DIR="/tmp/chatbot_test_$(date +%s)"
mkdir -p "$RESULTS_DIR"

echo "========================================="
echo "COMPREHENSIVE CHATBOT TEST"
echo "========================================="
echo "Results directory: $RESULTS_DIR"
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test 1: Multi-GPU Load Balancing
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST 1: Multi-GPU Load Balancing"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Sending 4 concurrent requests to different clusters..."

# Send 4 requests in parallel to test load balancing
(time curl -s -X POST "$ENDPOINT" \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I submit a job?", "cluster": "sherlock"}' \
  > "$RESULTS_DIR/gpu_test_1.json") 2> "$RESULTS_DIR/gpu_time_1.txt" &

(time curl -s -X POST "$ENDPOINT" \
  -H "Content-Type: application/json" \
  -d '{"query": "What storage is available?", "cluster": "oak"}' \
  > "$RESULTS_DIR/gpu_test_2.json") 2> "$RESULTS_DIR/gpu_time_2.txt" &

(time curl -s -X POST "$ENDPOINT" \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I connect?", "cluster": "farmshare"}' \
  > "$RESULTS_DIR/gpu_test_3.json") 2> "$RESULTS_DIR/gpu_time_3.txt" &

(time curl -s -X POST "$ENDPOINT" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Sherlock?", "cluster": "sherlock"}' \
  > "$RESULTS_DIR/gpu_test_4.json") 2> "$RESULTS_DIR/gpu_time_4.txt" &

# Wait for all parallel requests
wait

echo -e "${GREEN}✓${NC} All 4 concurrent requests completed"
echo ""
echo "Response times:"
for i in 1 2 3 4; do
  time=$(grep real "$RESULTS_DIR/gpu_time_$i.txt" | awk '{print $2}')
  echo "  Request $i: $time"
done
echo ""

# Test 2: Citation Accuracy
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST 2: Citation Accuracy"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

curl -s -X POST "$ENDPOINT" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Sherlock?", "cluster": "sherlock"}' \
  > "$RESULTS_DIR/citation_test.json"

answer=$(jq -r '.answer' "$RESULTS_DIR/citation_test.json")
sources=$(jq -r '.sources[] | .title' "$RESULTS_DIR/citation_test.json")

echo "Answer excerpt:"
echo "$answer" | head -c 200
echo "..."
echo ""

# Check for inline citations with URLs
if echo "$answer" | grep -q '\[.*\](https://.*\.stanford\.edu.*)'; then
  echo -e "${GREEN}✓${NC} Inline citations with URLs found"
else
  echo -e "${RED}✗${NC} WARNING: No inline citations with URLs found"
fi

# Check for document titles (not filenames)
if echo "$answer" | grep -q '\[.*\.md\]'; then
  echo -e "${RED}✗${NC} WARNING: Found .md filename citations (should be titles)"
else
  echo -e "${GREEN}✓${NC} No .md filename citations (good!)"
fi

# Count sources
source_count=$(jq '.sources | length' "$RESULTS_DIR/citation_test.json")
echo "Sources cited: $source_count"
if [ "$source_count" -le 3 ]; then
  echo -e "${GREEN}✓${NC} Source count reasonable ($source_count ≤ 3)"
else
  echo -e "${YELLOW}⚠${NC}  Many sources cited ($source_count) - might include irrelevant docs"
fi

echo ""
echo "Sources listed:"
echo "$sources" | sed 's/^/  - /'
echo ""

# Test 3: Semantic Caching Performance
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST 3: Semantic Caching Performance"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Query 1: Fresh (should be slow)
echo "Query 1: Fresh question (uncached)..."
time1=$( (time curl -s -X POST "$ENDPOINT" \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I log into Sherlock cluster?", "cluster": "sherlock"}' \
  > "$RESULTS_DIR/cache_test_1.json") 2>&1 | grep real | awk '{print $2}')

sleep 1

# Query 2: Exact match (should be very fast)
echo "Query 2: Exact same question (should hit cache)..."
time2=$( (time curl -s -X POST "$ENDPOINT" \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I log into Sherlock cluster?", "cluster": "sherlock"}' \
  > "$RESULTS_DIR/cache_test_2.json") 2>&1 | grep real | awk '{print $2}')

sleep 1

# Query 3: Semantic match (should be fast)
echo "Query 3: Similar question (should hit semantic cache)..."
time3=$( (time curl -s -X POST "$ENDPOINT" \
  -H "Content-Type: application/json" \
  -d '{"query": "How to connect to Sherlock?", "cluster": "sherlock"}' \
  > "$RESULTS_DIR/cache_test_3.json") 2>&1 | grep real | awk '{print $2}')

sleep 1

# Query 4: Different question (same cluster)
echo "Query 4: Similar wording (should hit semantic cache)..."
time4=$( (time curl -s -X POST "$ENDPOINT" \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I login to Sherlock", "cluster": "sherlock"}' \
  > "$RESULTS_DIR/cache_test_4.json") 2>&1 | grep real | awk '{print $2}')

echo ""
echo "Performance Results:"
echo "  Query 1 (fresh):           $time1"
echo "  Query 2 (exact cache):     $time2"
echo "  Query 3 (semantic cache):  $time3"
echo "  Query 4 (semantic cache):  $time4"
echo ""

# Extract seconds for comparison
sec1=$(echo "$time1" | sed 's/[^0-9.]*//g')
sec2=$(echo "$time2" | sed 's/[^0-9.]*//g')
sec3=$(echo "$time3" | sed 's/[^0-9.]*//g')
sec4=$(echo "$time4" | sed 's/[^0-9.]*//g')

# Check if caching is working (cached should be <10% of fresh)
if (( $(echo "$sec2 < $sec1 * 0.1" | bc -l) )); then
  echo -e "${GREEN}✓${NC} Exact cache working (${sec2}s < ${sec1}s * 0.1)"
else
  echo -e "${RED}✗${NC} Exact cache NOT working (${sec2}s vs ${sec1}s)"
fi

if (( $(echo "$sec3 < $sec1 * 0.1" | bc -l) )); then
  echo -e "${GREEN}✓${NC} Semantic cache working (${sec3}s < ${sec1}s * 0.1)"
else
  echo -e "${YELLOW}⚠${NC}  Semantic cache might not be working (${sec3}s vs ${sec1}s)"
fi

# Verify answers are identical for cached queries
ans1=$(jq -r '.answer' "$RESULTS_DIR/cache_test_1.json")
ans2=$(jq -r '.answer' "$RESULTS_DIR/cache_test_2.json")
ans3=$(jq -r '.answer' "$RESULTS_DIR/cache_test_3.json")
ans4=$(jq -r '.answer' "$RESULTS_DIR/cache_test_4.json")

echo ""
if [ "$ans1" == "$ans2" ]; then
  echo -e "${GREEN}✓${NC} Exact cache: Answers identical"
else
  echo -e "${RED}✗${NC} Exact cache: Answers differ!"
fi

if [ "$ans1" == "$ans3" ]; then
  echo -e "${GREEN}✓${NC} Semantic cache: Answers identical (Q1 vs Q3)"
else
  echo -e "${YELLOW}⚠${NC}  Semantic cache: Answers differ (might be below similarity threshold)"
fi

if [ "$ans1" == "$ans4" ]; then
  echo -e "${GREEN}✓${NC} Semantic cache: Answers identical (Q1 vs Q4)"
else
  echo -e "${YELLOW}⚠${NC}  Semantic cache: Answers differ (might be below similarity threshold)"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "SUMMARY"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "All test results saved to: $RESULTS_DIR"
echo ""
echo "Review individual results with:"
echo "  jq . $RESULTS_DIR/citation_test.json"
echo "  jq -r '.answer' $RESULTS_DIR/cache_test_1.json"
echo ""
echo "Test complete!"
