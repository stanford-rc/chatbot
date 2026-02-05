#!/bin/bash
# Benchmark script for TinyLlama 1.1B on ARM with L4 GPUs
# Run this inside your Apptainer container instance

echo "=== TinyLlama 1.1B Benchmark ==="
echo "Testing with 100 token generation..."

apptainer exec --nv instance://chatapi /opt/chatbot-env/bin/python -c "
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

model_name = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'

print('Loading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(model_name)

print('Loading model...')
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map='auto'
)

prompt = 'Write a detailed explanation of quantum computing'
print(f'Prompt: {prompt}')

inputs = tokenizer(prompt, return_tensors='pt').to('cuda')

print('Generating tokens...')
start = time.time()
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,
        num_beams=1,
        use_cache=True
    )
end = time.time()

tokens = outputs.shape[1] - inputs['input_ids'].shape[1]
elapsed = end - start
tokens_per_sec = tokens / elapsed

print(f'\n=== Results ===')
print(f'Generated {tokens} tokens in {elapsed:.2f}s')
print(f'Performance: {tokens_per_sec:.2f} tokens/sec')

# Memory usage
if torch.cuda.is_available():
    print(f'GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB')
    print(f'GPU Memory Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB')
"
