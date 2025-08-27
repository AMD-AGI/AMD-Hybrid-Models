#!/bin/bash

# A script to run LLM evaluation with lm-harness

# --- Configuration ---
# Set the visible devices for HIP and CUDA.
export HIP_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0

# Define variables for clarity and easy modification
MODEL_NAME="hybrid"
MODEL_PATH="amd/Zebra-Llama-1B-4MLA-12Mamba-DPO"
EVAL_TASKS="mmlu,hellaswag,piqa,arc_easy,arc_challenge,winogrande,openbookqa,pubmedqa,race"
NUM_FEWSHOT=0
DEVICE="cuda"
BATCH_SIZE=8
OUTPUT_LOG_FILE="llama1b_4MLA12Mamba.log"

# --- Execution ---
echo "Starting evaluation for model: ${MODEL_PATH}"
echo "Tasks: ${EVAL_TASKS}"
echo "Batch size: ${BATCH_SIZE}"
echo "---------------------------------------------------"

python benchmark/llm_eval/lm_harness_eval.py \
    --model "${MODEL_NAME}" \
    --model_args pretrained=${MODEL_PATH} \
    --tasks "${EVAL_TASKS}" \
    --num_fewshot "${NUM_FEWSHOT}" \
    --device "${DEVICE}" \
    --batch_size "${BATCH_SIZE}" \
    |& tee ${OUTPUT_LOG_FILE} 

echo "Evaluation started. Output is being logged to ${OUTPUT_LOG_FILE} and running in the background."
