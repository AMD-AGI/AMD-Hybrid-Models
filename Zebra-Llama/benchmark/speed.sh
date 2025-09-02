#!/bin/bash

# Define common parameters
MODEL='amd/Zebra-Llama-8B-8MLA-24Mamba-DPO'
REPEATS=1


# --- Generation Speed Benchmarks ---

# Scenario 1: Variable batch size, fixed generation length (8192)
echo "--- Running Generation Speed Benchmarks (Variable Batch Size) ---"
batches=(8 16 32 64 128 256 512 1024)

for batch in "${batches[@]}"; do
    echo "Running benchmark for model $MODEL with batchsize $batch"
    python benchmark_generation_speed.py --model-path $MODEL --gen-len 8192 --batch-size $batch --prompt-len 1 --repeats $REPEATS
done


# Scenario 2: Variable context length, fixed batch size (48)
echo "--- Running Context Length Benchmarks (Variable Prompt Length) ---"
contexts=(4096 8192 16384 32768 65536 131072 262144)

for context in "${contexts[@]}"; do
    echo "Running benchmark for model $model with batchsize $batch"
    python benchmark_generation_speed.py --model-path $MODE --gen-len 1024 --batch-size 48 --prompt-len $context --repeats $REPEATS 
done

