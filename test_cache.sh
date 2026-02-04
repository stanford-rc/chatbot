#!/bin/bash
echo "=== Semantic Cache Performance Test ==="
# Query 1: New question (will be slow, then cached)
echo "Query 1 (fresh):"
time curl -s -X POST "http://localhost:8000/query/" \
  -d '{"query": "What is Sherlock?", "cluster": "sherlock"}' \
  -H "Content-Type: application/json" > /tmp/test1.json
sleep 1
# Query 2: Exact same (should hit cache ~0.1s)
echo -e "\nQuery 2 (exact match):"
time curl -s -X POST "http://localhost:8000/query/" \
  -d '{"query": "What is Sherlock?", "cluster": "sherlock"}' \
  -H "Content-Type: application/json" > /tmp/test2.json
sleep 1
# Query 3: Similar wording (should hit semantic cache ~0.1s)
echo -e "\nQuery 3 (semantic match):"
time curl -s -X POST "http://localhost:8000/query/" \
  -d '{"query": "What is Sherlock", "cluster": "sherlock"}' \
  -H "Content-Type: application/json" > /tmp/test3.json
sleep 1
# Query 4: Different wording (should hit semantic cache ~0.1s)
echo -e "\nQuery 4 (semantic match):"
time curl -s -X POST "http://localhost:8000/query/" \
  -d '{"query": "Tell me about Sherlock", "cluster": "sherlock"}' \
  -H "Content-Type: application/json" > /tmp/test4.json
# Verify answers
echo -e "\n=== Answer Comparison ==="
if diff <(jq -S . /tmp/test1.json) <(jq -S . /tmp/test2.json) > /dev/null; then
  echo "✓ Q1 and Q2 identical (exact cache)"
fi
if diff <(jq -S . /tmp/test1.json) <(jq -S . /tmp/test3.json) > /dev/null; then
  echo "✓ Q1 and Q3 identical (semantic cache)"
fi
if diff <(jq -S . /tmp/test1.json) <(jq -S . /tmp/test4.json) > /dev/null; then
  echo "✓ Q1 and Q4 identical (semantic cache)"
fi
