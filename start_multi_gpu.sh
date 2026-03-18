#!/bin/bash
# Start multi-GPU chatbot with load balancer and queueing
# Handles unlimited concurrent requests across 2 GPU workers

cd "$(dirname "$0")"

echo "=== Multi-GPU Chatbot with Smart Queueing ==="
echo ""

# Make sure logs directory exists
mkdir -p logs

# Check if instance is running, start it if not

# Check if instance is running, start it if not
if ! apptainer instance list | grep -q chatapi; then
    echo "Starting chatapi instance..."
    apptainer instance start --nv \
      --bind "$PWD:$PWD" \
      --bind "$PWD/logs:logs" \
      chatbot.sif chatapi
    sleep 2
    echo "✓ Instance started"
else
    echo "✓ Instance already running"
fi

# Kill any existing workers
echo "Stopping any existing workers..."
pkill -f "uvicorn app.main:app.*8000" || true
pkill -f "uvicorn app.main:app.*800[12]" || true
pkill -f "load_balancer.py" || true
sleep 2

# Function to wait for worker health check
wait_for_worker() {
  local port=$1
  local worker_name=$2
  local max_attempts=60  # 5 minutes max (60 * 5 seconds)
  local attempt=0
  
  echo "Waiting for $worker_name to initialize..."
  
  while [ $attempt -lt $max_attempts ]; do
    if curl -s -f http://localhost:$port/health > /dev/null 2>&1; then
      echo "✓ $worker_name is ready!"
      return 0
    fi
    
    attempt=$((attempt + 1))
    if [ $((attempt % 6)) -eq 0 ]; then
      echo "  Still waiting for $worker_name... (${attempt}/60 checks, ~$((attempt * 5 / 60)) min elapsed)"
    fi
    sleep 5
  done
  
  echo "✗ $worker_name failed to start within timeout"
  echo "  Check logs/$worker_name.log for details"
  return 1
}

# Start Worker 1 on GPU 0 (port 8001)
echo ""
echo "Starting Worker 1 (GPU 0) on port 8001..."
APPTAINERENV_WORKER_GPU=cuda:0 apptainer exec --nv instance://chatapi \
  python3 -m uvicorn app.main:app \
  --host 127.0.0.1 \
  --port 8001 \
  --log-level info \
  > logs/worker1.log 2>&1 &

WORKER1_PID=$!
echo $WORKER1_PID > .worker1.pid
echo "  → Worker 1 PID: $WORKER1_PID"

# Wait for Worker 1 to be ready
if ! wait_for_worker 8001 "Worker 1"; then
  echo ""
  echo "ERROR: Worker 1 failed to start"
  echo "Tail of worker1.log:"
  tail -20 logs/worker1.log
  exit 1
fi

# Start Worker 2 on GPU 1 (port 8002)
echo ""
echo "Starting Worker 2 (GPU 1) on port 8002..."
APPTAINERENV_WORKER_GPU=cuda:1 apptainer exec --nv instance://chatapi \
  python3 -m uvicorn app.main:app \
  --host 127.0.0.1 \
  --port 8002 \
  --log-level info \
  > logs/worker2.log 2>&1 &

WORKER2_PID=$!
echo $WORKER2_PID > .worker2.pid
echo "  → Worker 2 PID: $WORKER2_PID"

# Wait for Worker 2 to be ready
if ! wait_for_worker 8002 "Worker 2"; then
  echo ""
  echo "ERROR: Worker 2 failed to start"
  echo "Tail of worker2.log:"
  tail -20 logs/worker2.log
  exit 1
fi

# Start load balancer inside container
echo "Starting load balancer on port 8000..."
apptainer exec instance://chatapi \
  python3 -m app.load_balancer > logs/loadbalancer.log 2>&1 &

LB_PID=$!
echo $LB_PID > .loadbalancer.pid
echo "  → Load Balancer PID: $LB_PID"

# Wait for Load Balancer to be ready
echo "Waiting for Load Balancer..."
sleep 3

if wait_for_worker 8000 "Load Balancer"; then
  echo ""
  echo "=========================================="
  echo "✓ Multi-GPU service with queueing started!"
  echo "=========================================="
  echo ""
  echo "Architecture:"
  echo "  Client → Load Balancer (port 8000)"
  echo "           ├─→ Worker 1 (GPU 0, port 8001)"
  echo "           └─→ Worker 2 (GPU 1, port 8002)"
  echo ""
  echo "API Endpoints:"
  echo "  • Query: http://localhost:8000/query/"
  echo "  • Health: http://localhost:8000/health"
  echo "  • Stats: http://localhost:8000/stats"
  echo ""
  echo "Features:"
  echo "  • Accepts unlimited concurrent requests"
  echo "  • Intelligent queue management"
  echo "  • Automatic load balancing"
  echo "  • ~24 tokens/sec total throughput"
  echo ""
  echo "Logs:"
  echo "  • logs/worker1.log - Worker 1 (GPU 0)"
  echo "  • logs/worker2.log - Worker 2 (GPU 1)"
  echo "  • logs/loadbalancer.log - Load Balancer"
  echo ""
  echo "To stop all services:"
  echo "  ./stop_multi_gpu.sh"
  echo ""
else
  echo ""
  echo "ERROR: Load Balancer failed to start"
  echo "Tail of loadbalancer.log:"
  tail -20 logs/loadbalancer.log
  exit 1
fi
