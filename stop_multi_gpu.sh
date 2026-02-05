#!/bin/bash
# Stop all multi-GPU services (workers + load balancer)

echo "Stopping multi-GPU services..."

if [ -f .loadbalancer.pid ]; then
    PID=$(cat .loadbalancer.pid)
    kill $PID 2>/dev/null && echo "✓ Stopped Load Balancer (PID $PID)" || echo "  Load Balancer already stopped"
    rm .loadbalancer.pid
fi

if [ -f .worker1.pid ]; then
    PID1=$(cat .worker1.pid)
    kill $PID1 2>/dev/null && echo "✓ Stopped Worker 1 (PID $PID1)" || echo "  Worker 1 already stopped"
    rm .worker1.pid
fi

if [ -f .worker2.pid ]; then
    PID2=$(cat .worker2.pid)
    kill $PID2 2>/dev/null && echo "✓ Stopped Worker 2 (PID $PID2)" || echo "  Worker 2 already stopped"
    rm .worker2.pid
fi

# Also kill by process name as backup
pkill -f "uvicorn app.main:app.*800[12]" 2>/dev/null
pkill -f "load_balancer.py" 2>/dev/null

echo "Done."
