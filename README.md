Here's a comprehensive README for your chatbot scripts:

```markdown
# SRCF Chatbot - Deployment Scripts

## Overview

This chatbot provides RAG (Retrieval-Augmented Generation) for Stanford Research Computing documentation. It can run in two modes:

- **Single-worker mode**: Development mode with auto-reload
- **Multi-GPU mode**: Production mode with load balancing across 2 GPUs

## Prerequisites

- Apptainer/Singularity installed
- Python 3.x with PyYAML on host machine
- NVIDIA GPUs available (for multi-GPU mode)
- `config.yaml` properly configured with model path

## Scripts

### `main.sh` - Primary startup script

**Usage:**
```bash
./main.sh           # Single-worker development mode (default)
./main.sh multi     # Multi-GPU production mode
```

**Single-worker mode** (`./main.sh`):
- Builds container if not present
- Starts single FastAPI worker on port 8000
- Enables `--reload` for automatic code reloading
- Exposes API on `0.0.0.0` (accessible from network)
- Best for: Development, testing, debugging

**Multi-GPU mode** (`./main.sh multi`):
- Builds container if not present
- Calls `start_multi_gpu.sh` to start load-balanced service
- Best for: Production deployment with high load

**What it does:**
1. Reads model path from `config.yaml`
2. Validates model exists on disk
3. Builds `chatbot.sif` container (if needed)
4. Starts Apptainer instance with GPU support
5. Binds necessary directories (workspace, model)
6. Launches worker(s) based on mode

---

### `start_multi_gpu.sh` - Multi-GPU load balancer

**Usage:**
```bash
./start_multi_gpu.sh
```

**What it does:**
1. Checks if container exists (must run `main.sh` first to build)
2. Starts Apptainer instance if not running
3. Launches 2 GPU workers:
   - Worker 1: GPU 0, port 8001
   - Worker 2: GPU 1, port 8002
4. Launches load balancer on port 8000
5. Waits for all services to become healthy
6. Routes incoming requests intelligently to available workers

**Architecture:**
```
Client Request → Load Balancer (port 8000)
                 ├─→ Worker 1 (GPU 0, port 8001)
                 └─→ Worker 2 (GPU 1, port 8002)
```

**Features:**
- Accepts unlimited concurrent requests
- Queues requests when all workers busy
- Automatic failover if worker errors
- Per-worker statistics tracking
- ~24 tokens/sec total throughput

**Logs:**
- `logs/worker1.log` - Worker 1 output
- `logs/worker2.log` - Worker 2 output
- `logs/loadbalancer.log` - Load balancer output

---

### `stop_multi_gpu.sh` - Stop all services

**Usage:**
```bash
./stop_multi_gpu.sh
```

**What it does:**
- Kills all worker processes
- Kills load balancer process
- Stops Apptainer instance
- Cleans up PID files

---

## Quick Start

### Development Workflow

```bash
# First time setup
./main.sh

# Access API
curl http://ada-lovelace.stanford.edu:8000/health
curl http://ada-lovelace.stanford.edu:8000/docs

# Make code changes - server auto-reloads
# Edit app/*.py files

# Stop when done
apptainer instance stop chatapi
```

### Production Deployment

```bash
# Start multi-GPU service
./main.sh multi

# Check status
curl http://localhost:8000/health
curl http://localhost:8000/stats

# Monitor logs
tail -f logs/worker1.log
tail -f logs/worker2.log
tail -f logs/loadbalancer.log

# Stop all services
./stop_multi_gpu.sh
```

---

## API Endpoints

### Single-worker mode (port 8000)
- `GET /` - Root endpoint
- `GET /health` - Health check
- `POST /query/` - Submit query
- `GET /stats` - Worker statistics

### Multi-GPU mode (port 8000)
- `GET /health` - Load balancer health + worker status
- `POST /query/` - Submit query (routed to available worker)
- `GET /stats` - Detailed worker statistics

### Individual workers (multi-GPU only, ports 8001-8002)
- `GET /health` - Worker health check
- `POST /query/` - Direct query to specific worker
- `GET /stats` - Worker-specific stats

---

## Troubleshooting

### Container won't build
```bash
# Check syntax
apptainer build --sandbox test_chatbot chatbot.def

# Check logs
cat chatbot.def
```

### Workers fail to start
```bash
# Check logs
tail -50 logs/worker1.log

# Verify instance is running
apptainer instance list

# Check GPU availability
nvidia-smi
```

### Load balancer can't connect to workers
```bash
# Verify workers are listening
netstat -tlnp | grep 800[12]

# Test worker directly
curl http://localhost:8001/health
curl http://localhost:8002/health

# Check firewall
# (Ports 8001-8002 only need to be accessible from localhost)
```

### Model not loading
```bash
# Verify model path in config.yaml
cat config.yaml

# Check if model files exist
ls -la ~/apichatbot/models/YOUR_MODEL/

# Ensure required files present
# - config.json
# - pytorch_model.bin or model.safetensors
# - tokenizer files
```

---

## Configuration

### `config.yaml`
```yaml
model:
  path: /path/to/your/model  # Must exist on host
  device: "cuda"             # or "cpu"
  type: "mistral"
  
api:
  cors_origins:
    - "http://localhost:3000"
    - "https://your-frontend.com"
```

### Environment Variables

Set via `APPTAINERENV_*` prefix:
- `APPTAINERENV_WORKER_GPU=cuda:0` - Assign specific GPU
- `APPTAINERENV_LOG_LEVEL=debug` - Set log verbosity

---

## File Structure

```
apichatbot/
├── main.sh                  # Primary startup script
├── start_multi_gpu.sh       # Multi-GPU launcher
├── stop_multi_gpu.sh        # Stop all services
├── chatbot.def             # Container definition
├── chatbot.sif             # Built container (generated)
├── config.yaml             # Configuration
├── requirements.txt        # Python dependencies
├── app/
│   ├── main.py            # Main FastAPI app (worker)
│   ├── load_balancer.py   # Load balancer
│   ├── config.py          # Config loader
│   ├── models.py          # Pydantic models
│   └── rag_service.py     # RAG implementation
├── logs/                   # Runtime logs (generated)
│   ├── worker1.log
│   ├── worker2.log
│   └── loadbalancer.log
└── models/                 # LLM models
    └── your-model/
```

---

## Performance

### Single-worker mode
- ~12 tokens/sec
- 1 GPU utilized
- Good for: <5 concurrent users

### Multi-GPU mode
- ~24 tokens/sec total
- 2 GPUs utilized
- Request queueing enabled
- Good for: 10+ concurrent users

---

## Notes

- Container must be rebuilt after changing `chatbot.def`
- Code changes in `app/` hot-reload in single-worker mode
- Code changes in `app/` require restart in multi-GPU mode
- Logs rotate automatically (managed by system)
- PID files stored as `.worker1.pid`, `.worker2.pid`, `.loadbalancer.pid`

---

## Support

For issues or questions, check:
1. Log files in `logs/`
2. Container build output
3. GPU availability with `nvidia-smi`
4. Model path in `config.yaml`
