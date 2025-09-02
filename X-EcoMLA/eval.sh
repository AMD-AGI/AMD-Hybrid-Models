ckpts=(
    "checkpoint_path"
)


# Get the number of elements in the ckpts array
num_ckpts=${#ckpts[@]}
# Loop through indexes from 0 to (num_ckpts-1)
for index in $(seq 0 $((num_ckpts-1))); do
    # # Run evaluation
    echo "Running evaluation for $current_ckpt"
    HIP_VISIBLE_DEVICES=$index CUDA_VISIBLE_DEVICES=$index \
    python benchmark/llm_eval/lm_harness_eval.py \
    --model mla_hybrid \
    --model_args pretrained="$current_ckpt" \
    --tasks mmlu,hellaswag,piqa,arc_easy,arc_challenge,winogrande,openbookqa,pubmedqa,race \
    --num_fewshot 0 --device cuda --batch_size 16 > "${current_ckpt}/lm_harness_eval.md" 2>&1 &
done

