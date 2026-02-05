#!/bin/bash
# GPU Diagnostic Script
# Verifies that GPU compute is actually happening, not just memory allocation

echo "=== GPU Diagnostic ==="

apptainer exec --nv instance://chatapi /opt/chatbot-env/bin/python -c "
import torch
import time

print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('CUDA version:', torch.version.cuda)
print('Number of GPUs:', torch.cuda.device_count())

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
        print(f'  Compute Capability: {torch.cuda.get_device_capability(i)}')

# Test actual GPU compute speed
print('\n=== GPU Compute Speed Test ===')
if torch.cuda.is_available():
    # Matrix multiplication on GPU
    size = 4096
    a = torch.randn(size, size, device='cuda', dtype=torch.float16)
    b = torch.randn(size, size, device='cuda', dtype=torch.float16)
    
    # Warmup
    _ = torch.matmul(a, b)
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.time()
    for _ in range(10):
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
    elapsed = time.time() - start
    
    tflops = (2 * size**3 * 10) / (elapsed * 1e12)
    print(f'GPU Matrix Multiply (FP16): {elapsed/10:.4f}s per 4096x4096')
    print(f'Performance: {tflops:.2f} TFLOPS')
    print('Expected on L4 GPU: ~15-20 TFLOPS')
    
    if tflops < 5:
        print('⚠️  WARNING: GPU performance is very low! Possible issues:')
        print('   - CPU fallback occurring')
        print('   - CUDA version mismatch')
        print('   - ARM architecture incompatibility')
else:
    print('❌ CUDA not available!')

# Test transformers library
print('\n=== Transformers Library Test ===')
from transformers import __version__ as transformers_version
print(f'Transformers version: {transformers_version}')

# Check if bitsandbytes has ARM support
try:
    import bitsandbytes as bnb
    print(f'bitsandbytes version: {bnb.__version__}')
    print('bitsandbytes available: ✓')
except Exception as e:
    print(f'bitsandbytes issue: {e}')
"
