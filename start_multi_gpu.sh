#!/bin/bash
# Start multi-GPU chatbot with load balancer and queueing
# Handles unlimited concurrent requests across 2 GPU workers

cd "$(dirname "$0")"

echo "=== Multi-GPU Chatbot with Smart Queueing ==="
echo ""

# Check if instance is running, start it if not
if ! apptainer instance list | grep -q chatapi; then
    echo "Starting chatapi instance..."
    apptainer instance start --nv --bind "$PWD:$PWD" chatbot.sif chatapi
    sleep 2
    echo "✓ Instance started"
else
    echo "✓ Instance already running"
fi

# Kill any existing single worker on port 8000
echo "Stopping any existing workers..."
pkill -f "uvicorn app.main:app.*8000" || true
pkill -f "uvicorn app.main:app.*800[12]" || true
pkill -f "load_balancer.py" || true
sleep 2


# Make sure logs directory exists
mkdir -p logs

# Start workers
echo "Starting Worker 1 (GPU 0) on port 8001..."
APPTAINERENV_WORKER_GPU=cuda:0 apptainer exec --nv instance://chatapi \
  python3 -m uvicorn app.main:app \
  --host 127.0.0.1 \
  --port 8001 \
  --log-level info > logs/worker1.log 2>&1 &

echo "Starting Worker 2 (GPU 1) on port 8002..."
APPTAINERENV_WORKER_GPU=cuda:1 apptainer exec --nv instance://chatapi \
  python3 -m uvicorn app.main:app \
  --host 127.0.0.1 \
  --port 8002 \
  --log-level info > logs/worker2.log 2>&1 &

WORKER1_PID=$!
echo $WORKER1_PID > .worker1.pid
echo "  → Worker 1 PID: $WORKER1_PID"

WORKER2_PID=$!
echo $WORKER2_PID > .worker2.pid
echo "  → Worker 2 PID: $WORKER2_PID"

# Wait for workers to be ready
echo "Waiting for workers to initialize..."
sleep 30  # Give workers time to load the model

# Start load balancer inside container
echo "Starting load balancer on port 8000..."
apptainer exec instance://chatapi \
  python3 /workspace/load_balancer.py > logs/loadbalancer.log 2>&1 &




LB_PID=$!
echo $LB_PID > .loadbalancer.pid
echo "  → Load Balancer PID: $LB_PID"

echo ""
echo "✓ Multi-GPU service with queueing started!"
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
echo "To stop all services:"
echo "  ./stop_multi_gpu.sh"
echo ""
echo "Monitoring:"
echo "  watch -n 1 'curl -s http://localhost:8000/stats | jq'"

echo "All services started!"
echo "Check logs/worker*.log and logs/loadbalancer.log for details"
