#!/bin/bash
# Start multi-GPU chatbot with load balancer and queueing
# Handles unlimited concurrent requests across 2 GPU workers

cd "$(dirname "$0")"

echo "=== Multi-GPU Chatbot with Smart Queueing ==="
echo ""

# Check if container exists
if [ ! -f chatbot.sif ]; then
    echo "ERROR: chatbot.sif not found. Run ./main.sh first to build."
    exit 1
fi

# Read settings from config.yaml
MODEL_PATH=$(python3 -c "import yaml; c=yaml.safe_load(open('config.yaml')); print(c['model']['path'])")
API_PORT=$(python3 -c "import yaml; c=yaml.safe_load(open('config.yaml')); print(c.get('server',{}).get('api_port',8000))")
LOG_DIR=$(python3 -c "import yaml; c=yaml.safe_load(open('config.yaml')); print(c.get('logging',{}).get('log_dir','logs'))")
WORKER1_PORT=$(python3 -c "import yaml; c=yaml.safe_load(open('config.yaml')); w=c.get('workers',[]); print(w[0]['port'] if w else 8001)")
WORKER2_PORT=$(python3 -c "import yaml; c=yaml.safe_load(open('config.yaml')); w=c.get('workers',[]); print(w[1]['port'] if len(w)>1 else 8002)")
WORKER1_GPU=$(python3 -c "import yaml; c=yaml.safe_load(open('config.yaml')); w=c.get('workers',[]); print(w[0]['gpu'] if w else 'cuda:0')")
WORKER2_GPU=$(python3 -c "import yaml; c=yaml.safe_load(open('config.yaml')); w=c.get('workers',[]); print(w[1]['gpu'] if len(w)>1 else 'cuda:1')")

# Make sure logs directory exists
mkdir -p "$LOG_DIR"

# Check if instance is running, start it if not
if ! apptainer instance list | grep -q chatapi; then
    echo "Starting chatapi instance..."
    
    BIND_MOUNTS="--bind $PWD:/workspace"
    if [ -d "$MODEL_PATH" ]; then
        BIND_MOUNTS="$BIND_MOUNTS --bind $MODEL_PATH:$MODEL_PATH:ro"
    fi
    
    apptainer instance start --nv $BIND_MOUNTS chatbot.sif chatapi
    sleep 2
    echo "✓ Instance started"
else
    echo "✓ Instance already running"
fi

# Kill any existing workers
echo "Stopping any existing workers..."
pkill -f "uvicorn app.main:app.*$API_PORT" || true
pkill -f "uvicorn app.main:app.*$WORKER1_PORT" || true
pkill -f "uvicorn app.main:app.*$WORKER2_PORT" || true
pkill -f "uvicorn app.load_balancer:app" || true
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
  echo "  Check $LOG_DIR/$worker_name.log for details"
  return 1
}

# Start Worker 1
echo ""
echo "Starting Worker 1 ($WORKER1_GPU) on port $WORKER1_PORT..."
APPTAINERENV_WORKER_GPU=$WORKER1_GPU \
APPTAINERENV_PYTHONPATH=/workspace \
apptainer exec --nv instance://chatapi \
  /opt/chatbot-env/bin/python -m uvicorn app.main:app \
  --host 127.0.0.1 \
  --port $WORKER1_PORT \
  --log-level info \
  > "$LOG_DIR/worker1.log" 2>&1 &

WORKER1_PID=$!
echo $WORKER1_PID > .worker1.pid
echo "  → Worker 1 PID: $WORKER1_PID"

# Wait for Worker 1 to be ready
if ! wait_for_worker $WORKER1_PORT "Worker 1"; then
  echo ""
  echo "ERROR: Worker 1 failed to start"
  echo "Tail of $LOG_DIR/worker1.log:"
  tail -20 "$LOG_DIR/worker1.log"
  exit 1
fi

# Start Worker 2
echo ""
echo "Starting Worker 2 ($WORKER2_GPU) on port $WORKER2_PORT..."
APPTAINERENV_WORKER_GPU=$WORKER2_GPU \
APPTAINERENV_PYTHONPATH=/workspace \
apptainer exec --nv instance://chatapi \
  /opt/chatbot-env/bin/python -m uvicorn app.main:app \
  --host 127.0.0.1 \
  --port $WORKER2_PORT \
  --log-level info \
  > "$LOG_DIR/worker2.log" 2>&1 &

WORKER2_PID=$!
echo $WORKER2_PID > .worker2.pid
echo "  → Worker 2 PID: $WORKER2_PID"

# Wait for Worker 2 to be ready
if ! wait_for_worker $WORKER2_PORT "Worker 2"; then
  echo ""
  echo "ERROR: Worker 2 failed to start"
  echo "Tail of $LOG_DIR/worker2.log:"
  tail -20 "$LOG_DIR/worker2.log"
  exit 1
fi

# Start load balancer inside container
echo ""
echo "Starting load balancer on port $API_PORT..."
apptainer exec instance://chatapi \
  /opt/chatbot-env/bin/python -m uvicorn app.load_balancer:app \
  --host 0.0.0.0 \
  --port $API_PORT \
  --log-level info \
  > "$LOG_DIR/loadbalancer.log" 2>&1 &

LB_PID=$!
echo $LB_PID > .loadbalancer.pid
echo "  → Load Balancer PID: $LB_PID"

# Wait for Load Balancer to be ready
echo "Waiting for Load Balancer..."
sleep 3

if wait_for_worker $API_PORT "Load Balancer"; then
  echo ""
  echo "=========================================="
  echo "✓ Multi-GPU service with queueing started!"
  echo "=========================================="
  echo ""
  echo "Architecture:"
  echo "  Client → Load Balancer (port $API_PORT)"
  echo "           ├─→ Worker 1 ($WORKER1_GPU, port $WORKER1_PORT)"
  echo "           └─→ Worker 2 ($WORKER2_GPU, port $WORKER2_PORT)"
  echo ""
  echo "API Endpoints:"
  echo "  • Query: http://localhost:$API_PORT/query/"
  echo "  • Health: http://localhost:$API_PORT/health"
  echo "  • Stats: http://localhost:$API_PORT/stats"
  echo ""
  echo "Features:"
  echo "  • Accepts unlimited concurrent requests"
  echo "  • Intelligent queue management"
  echo "  • Automatic load balancing"
  echo "  • ~24 tokens/sec total throughput"
  echo ""
  echo "Logs:"
  echo "  • $LOG_DIR/worker1.log - Worker 1 ($WORKER1_GPU)"
  echo "  • $LOG_DIR/worker2.log - Worker 2 ($WORKER2_GPU)"
  echo "  • $LOG_DIR/loadbalancer.log - Load Balancer"
  echo ""
  echo "To stop all services:"
  echo "  ./stop_multi_gpu.sh"
  echo ""
else
  echo ""
  echo "ERROR: Load Balancer failed to start"
  echo "Tail of $LOG_DIR/loadbalancer.log:"
  tail -20 "$LOG_DIR/loadbalancer.log"
  exit 1
fi
