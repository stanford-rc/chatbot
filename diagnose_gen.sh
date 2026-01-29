#!/bin/bash
# Deep Generation Pipeline Diagnostic
# Tests if the problem is in the generation loop itself

echo "=== Generation Pipeline Diagnostic ==="

apptainer exec --nv instance://chatapi /opt/chatbot-env/bin/python -c "
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

model_name = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'

print('Loading TinyLlama (1.1B - should be FAST)...')
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map='cuda:0'  # Force to single GPU
)
model.eval()

print(f'Model device: {model.device}')
print(f'Model dtype: {model.dtype}')

# Test 1: Single forward pass (no generation)
print('\n=== Test 1: Single Forward Pass (no generation loop) ===')
prompt = 'Hello world'
inputs = tokenizer(prompt, return_tensors='pt').to('cuda:0')

start = time.time()
with torch.no_grad():
    outputs = model(**inputs)
elapsed = time.time() - start
print(f'Single forward pass: {elapsed*1000:.2f}ms')
print('Expected on fast GPU: <10ms')

# Test 2: Generate with profiling
print('\n=== Test 2: Generation with GPU utilization ===')
inputs = tokenizer('Write a story', return_tensors='pt').to('cuda:0')

# Check GPU before
torch.cuda.synchronize()
print(f'GPU memory before: {torch.cuda.memory_allocated()/1e9:.2f}GB')

start = time.time()
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=10,  # Just 10 tokens
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id
    )
torch.cuda.synchronize()
elapsed = time.time() - start

tokens = outputs.shape[1] - inputs['input_ids'].shape[1]
print(f'Generated {tokens} tokens in {elapsed:.3f}s')
print(f'Speed: {tokens/elapsed:.2f} tokens/sec')
print(f'Time per token: {elapsed/tokens*1000:.1f}ms')
print('Expected per token on L4: ~10-50ms')

if elapsed/tokens > 1.0:
    print('⚠️  PROBLEM: >1 second per token is EXTREMELY slow!')
    print('   This suggests:')
    print('   - CPU fallback in generation loop')
    print('   - Device transfers happening')
    print('   - Transformers library issue on ARM')

# Test 3: Check what device tensors are on during generation
print('\n=== Test 3: Device Location Check ===')
for name, param in model.named_parameters():
    if 'lm_head' in name or 'embed' in name:
        print(f'{name}: {param.device}')
    if not param.is_cuda:
        print(f'⚠️  {name} is on CPU!')

# Test 4: Try with explicit device and no device_map
print('\n=== Test 4: Generation with model.to(cuda) ===')
model2 = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16
).to('cuda:0')
model2.eval()

inputs = tokenizer('Test', return_tensors='pt').to('cuda:0')
start = time.time()
with torch.no_grad():
    outputs = model2.generate(**inputs, max_new_tokens=10, do_sample=False, use_cache=True, pad_token_id=tokenizer.eos_token_id)
elapsed = time.time() - start
print(f'Speed with .to(cuda): {10/elapsed:.2f} tokens/sec')

# Test 5: Check PyTorch CUDA settings
print('\n=== Test 5: PyTorch CUDA Settings ===')
print(f'cudnn.enabled: {torch.backends.cudnn.enabled}')
print(f'cudnn.benchmark: {torch.backends.cudnn.benchmark}')
print(f'CUDA launch blocking: {torch.cuda.is_initialized()}')
"
