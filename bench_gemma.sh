#!/bin/bash
# Benchmark script for Gemma 2 9B on ARM with L4 GPUs
# Run this inside your Apptainer container instance
# Model path is read from config.yaml for consistency

echo "=== Gemma 2 9B Benchmark ==="
echo "Testing with 100 token generation..."
echo "Reading model configuration from config.yaml..."

apptainer exec --nv instance://chatapi /opt/chatbot-env/bin/python -c "
import torch
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

# Read config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

model_path = config['model']['path']
use_quantization = config['model']['use_quantization']
local_files_only = config['model']['local_files_only']
device = config['model']['device']

print(f'Model path: {model_path}')
print(f'Device: {device}')
print(f'Quantization: {use_quantization}')
print('Loading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=local_files_only)

print(f'Loading model in FP16 (quantization={use_quantization})...')
if use_quantization:
    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map='auto',
        local_files_only=local_files_only,
        torch_dtype=torch.float16
    )
else:
    # Load without device_map to avoid multi-GPU overhead
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        local_files_only=local_files_only
    )
    model = model.to(device)
    print(f'Model loaded to {device}')

prompt = 'Write a detailed explanation of quantum computing'

inputs = tokenizer(prompt, return_tensors='pt').to(device)

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
