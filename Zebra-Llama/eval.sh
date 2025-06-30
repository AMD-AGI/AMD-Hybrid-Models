ckpts=(
    "checkpoint_path"
)


# Get the number of elements in the ckpts array
num_ckpts=${#ckpts[@]}
# Loop through indexes from 0 to (num_ckpts-1)
for index in $(seq 0 $((num_ckpts-1))); do
    current_ckpt="${ckpts[$index]}"
    repo_path=`pwd`
    if [ ! -f "${current_ckpt}/model.safetensors" ]; then
        # Navigate to the checkpoint directory
        pushd "$current_ckpt"
        
        
        # Find the latest checkpoint directory (format: checkpoint-XXXX)
        latest_checkpoint=$(find . -maxdepth 1 -type d -name "checkpoint-*" | sort -V | tail -n 1)
        
        if [ -z "$latest_checkpoint" ]; then
            echo "No checkpoint directory found in $current_ckpt"
            popd
            continue
        fi
        
        # Extract checkpoint number for global_step directory
        checkpoint_num=$(echo "$latest_checkpoint" | sed 's/.*checkpoint-//')
        
        echo "Processing $current_ckpt with checkpoint $checkpoint_num"
        
        # Convert model
        accelerate merge-weights ${latest_checkpoint}/pytorch_model_fsdp_0/ ./
        if [ -f "${current_ckpt}/model.safetensors" ]; then
            # Clean up
            rm -rf "$latest_checkpoint" "global_step$checkpoint_num"
        fi
        popd
    fi
    # # Run evaluation
    echo "Running evaluation for $current_ckpt"
    HIP_VISIBLE_DEVICES=$index CUDA_VISIBLE_DEVICES=$index \
    python benchmark/llm_eval/lm_harness_eval.py \
    --model hybrid \
    --model_args pretrained="$current_ckpt" \
    --tasks mmlu,hellaswag,piqa,arc_easy,arc_challenge,winogrande,openbookqa,pubmedqa,race \
    --num_fewshot 0 --device cuda --batch_size 16 > "${current_ckpt}/lm_harness_eval.md" 2>&1 &
done
