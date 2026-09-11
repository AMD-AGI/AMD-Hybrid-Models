#!/bin/bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
#################################################################################

# Define a function to launch training with specified configurations
launch_training() {
    local fsdp_config=$1
    local model_config=$2
    local script=$3
    
    echo "Starting training with FSDP config: $fsdp_config and Model config: $model_config"
    
    # Run the training script
    ACCELERATE_LOG_LEVEL=info accelerate launch \
        --config_file "$fsdp_config" \
        "$script" \
        "$model_config"
    
    # Check if the training command was successful
    if [ $? -ne 0 ]; then
        echo "Error: Training failed for model config: $model_config"
        return 1
    fi
    
    # Extract the output directory from the YAML config file
    local checkpoint_dir
    checkpoint_dir=$(python3 -c "import yaml; import sys;
with open(sys.argv[1], 'r') as f:
    data = yaml.safe_load(f);
    path = sys.argv[2].split('.');
    value = data;
    for key in path:
        value = value.get(key, None);
        if value is None:
            break;
    if value is not None:
        print(value);" "$model_config" "output_dir")

    # Check if the output directory was found
    if [ -z "$checkpoint_dir" ]; then
        echo "Error: 'output_dir' not found in config file: $model_config"
        return 1
    fi
    
    echo "Found checkpoint directory: $checkpoint_dir"
    
    # Check if the checkpoint directory exists and is a directory
    if [ ! -d "$checkpoint_dir" ]; then
        echo "Error: Checkpoint directory '$checkpoint_dir' does not exist or is not a directory."
        return 1
    fi

    pushd "$checkpoint_dir" || { echo "Error: Failed to change directory to $checkpoint_dir"; return 1; }
    
    # Find the latest checkpoint
    local last_checkpoint
    last_checkpoint=$(find . -maxdepth 1 -type d -name "checkpoint-*" | sort -V | tail -n 1)
    
    if [ -z "$last_checkpoint" ]; then
        echo "No checkpoint directory found in $checkpoint_dir"
        popd
        return 1
    fi
    
    # Get the checkpoint number
    local checkpoint_num
    checkpoint_num=$(echo "$last_checkpoint" | sed 's/.*checkpoint-//')
    
    echo "Processing checkpoint $checkpoint_num in directory $checkpoint_dir"
    
    # Convert and merge the model weights
    accelerate merge-weights "$last_checkpoint/pytorch_model_fsdp_0/" ./
    
    popd
    echo "Finished processing checkpoint."
    echo "--------------------------------------------------"
}

# --- Main Script Execution ---

# End-to-end recipe for HyLo-Llama-4MLA12GDN-8K-SFT. Stage 1 runs twice, once per
# block type, because the SFT config initialises its MLA and linear layers from two
# separate ILD checkpoints.

# Stage 1a — Enhanced-ILD for the linear (Gated DeltaNet) layers.
launch_training \
    configs/fsdp_GDN_ILD.yaml \
    configs/llama3.2_1B/zebra_GDN_ILD.yaml \
    train_hybrid/train_distill.py

# Stage 1b — Enhanced-ILD for the MLA layers.
launch_training \
    configs/fsdp_MLA_ILD.yaml \
    configs/llama3.2_1B/zebra_MLA_ILD.yaml \
    train_hybrid/train_distill.py

# Stage 2 — Long-context SFT with teacher-guided distillation.
launch_training \
    configs/fsdp_GDN.yaml \
    configs/llama3.2_1B/zebra_4MLA12GDN_8bT_SFT_8k_combined.yaml \
    train_hybrid/train_distill.py

# Swap the stage-2 config for any other released variant, e.g.
#     configs/llama3.2_3B/zebra_6MLA22GDN_8bT_SFT_64k_combined.yaml
#     configs/qwen3_1.7B/zebra_7MLA21GDN_8bT_SFT_8k_combined.yaml
# Mamba-2 variants use configs/fsdp_M2_ILD.yaml for stage 1 and configs/fsdp.yaml
# for stage 2.

