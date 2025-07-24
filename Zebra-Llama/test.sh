


# HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 \
# python benchmark/llm_eval/lm_harness_eval_latest.py \
# --model hybrid \
# --model_args pretrained="amd/Zebra-Llama-1B-4MLA-12Mamba-SFT" \
# --tasks truthfulqa_mc1,truthfulqa_mc2,truthfulqa_gen   \
# --num_fewshot 0 --device cuda --batch_size 8 > logs/truthfulQA_hybrid_1B_4MLA_12M2_SFT.log 2>&1 &

# HIP_VISIBLE_DEVICES=1 CUDA_VISIBLE_DEVICES=1 \
# python benchmark/llm_eval/lm_harness_eval_latest.py \
# --model hybrid \
# --model_args pretrained="amd/Zebra-Llama-1B-8MLA-8Mamba-SFT" \
# --tasks truthfulqa_mc1,truthfulqa_mc2,truthfulqa_gen   \
# --num_fewshot 0 --device cuda --batch_size 8 > logs/truthfulQA_hybrid_1B_8MLA_8M2_SFT.log 2>&1 &


# HIP_VISIBLE_DEVICES=2 CUDA_VISIBLE_DEVICES=2 \
# python benchmark/llm_eval/lm_harness_eval_latest.py \
# --model hybrid \
# --model_args pretrained="amd/Zebra-Llama-3B-6MLA-22Mamba-SFT" \
# --tasks truthfulqa_mc1,truthfulqa_mc2,truthfulqa_gen   \
# --num_fewshot 0 --device cuda --batch_size 8 > logs/truthfulQA_hybrid_3B_6MLA_22M2_SFT.log 2>&1 &

# HIP_VISIBLE_DEVICES=3 CUDA_VISIBLE_DEVICES=3 \
# python benchmark/llm_eval/lm_harness_eval_latest.py \
# --model hybrid \
# --model_args pretrained="amd/Zebra-Llama-3B-14MLA-14Mamba-SFT" \
# --tasks truthfulqa_mc1,truthfulqa_mc2,truthfulqa_gen   \
# --num_fewshot 0 --device cuda --batch_size 8 > logs/truthfulQA_hybrid_3B_14MLA_14M2_SFT.log 2>&1 &


# HIP_VISIBLE_DEVICES=4 CUDA_VISIBLE_DEVICES=4 \
# python benchmark/llm_eval/lm_harness_eval_latest.py \
# --model hybrid \
# --model_args pretrained="amd/Zebra-Llama-8B-8MLA-24Mamba-SFT" \
# --tasks truthfulqa_mc1,truthfulqa_mc2,truthfulqa_gen   \
# --num_fewshot 0 --device cuda --batch_size 8 > logs/truthfulQA_hybrid_8B_8MLA_24M2_SFT.log 2>&1 &

# HIP_VISIBLE_DEVICES=5 CUDA_VISIBLE_DEVICES=5 \
# python benchmark/llm_eval/lm_harness_eval_latest.py \
# --model hybrid \
# --model_args pretrained="amd/Zebra-Llama-8B-16MLA-16Mamba-SFT" \
# --tasks truthfulqa_mc1,truthfulqa_mc2,truthfulqa_gen   \
# --num_fewshot 0 --device cuda --batch_size 8 > logs/truthfulQA_hybrid_8B_16MLA_16M2_SFT.log 2>&1 &





# HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 \
# python benchmark/llm_eval/lm_harness_eval_latest.py \
# --model hybrid \
# --model_args pretrained="amd/Zebra-Llama-1B-4MLA-12Mamba-SFT" \
# --tasks toxigen   \
# --num_fewshot 0 --device cuda --batch_size 8 > logs/toxigen_hybrid_1B_4MLA_12M2_SFT.log 2>&1 &

# HIP_VISIBLE_DEVICES=1 CUDA_VISIBLE_DEVICES=1 \
# python benchmark/llm_eval/lm_harness_eval_latest.py \
# --model hybrid \
# --model_args pretrained="amd/Zebra-Llama-1B-8MLA-8Mamba-SFT" \
# --tasks toxigen   \
# --num_fewshot 0 --device cuda --batch_size 8 > logs/toxigen_hybrid_1B_8MLA_8M2_SFT.log 2>&1 &


# HIP_VISIBLE_DEVICES=2 CUDA_VISIBLE_DEVICES=2 \
# python benchmark/llm_eval/lm_harness_eval_latest.py \
# --model hybrid \
# --model_args pretrained="amd/Zebra-Llama-3B-6MLA-22Mamba-SFT" \
# --tasks toxigen   \
# --num_fewshot 0 --device cuda --batch_size 8 > logs/toxigen_hybrid_3B_6MLA_22M2_SFT.log 2>&1 &

# HIP_VISIBLE_DEVICES=3 CUDA_VISIBLE_DEVICES=3 \
# python benchmark/llm_eval/lm_harness_eval_latest.py \
# --model hybrid \
# --model_args pretrained="amd/Zebra-Llama-3B-14MLA-14Mamba-SFT" \
# --tasks toxigen   \
# --num_fewshot 0 --device cuda --batch_size 8 > logs/toxigen_hybrid_3B_14MLA_14M2_SFT.log 2>&1 &


# HIP_VISIBLE_DEVICES=4 CUDA_VISIBLE_DEVICES=4 \
# python benchmark/llm_eval/lm_harness_eval_latest.py \
# --model hybrid \
# --model_args pretrained="amd/Zebra-Llama-8B-8MLA-24Mamba-SFT" \
# --tasks toxigen   \
# --num_fewshot 0 --device cuda --batch_size 8 > logs/toxigen_hybrid_8B_8MLA_24M2_SFT.log 2>&1 &

# HIP_VISIBLE_DEVICES=5 CUDA_VISIBLE_DEVICES=5 \
# python benchmark/llm_eval/lm_harness_eval_latest.py \
# --model hybrid \
# --model_args pretrained="amd/Zebra-Llama-8B-16MLA-16Mamba-SFT" \
# --tasks toxigen   \
# --num_fewshot 0 --device cuda --batch_size 8 > logs/toxigen_hybrid_8B_16MLA_16M2_SFT.log 2>&1 &



# HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 \
# python benchmark/llm_eval/lm_harness_eval_latest.py \
# --model hybrid \
# --model_args pretrained="amd/Zebra-Llama-1B-4MLA-12Mamba-SFT" \
# --tasks 2wikimqa   \
# --num_fewshot 0 --device cuda --batch_size 8 #--apply_chat_template #> logs/ifeval_hybrid_1B_4MLA_12M2_SFT.log 2>&1 &

# HIP_VISIBLE_DEVICES=1 CUDA_VISIBLE_DEVICES=1 \
# python benchmark/llm_eval/lm_harness_eval_latest.py \
# --model hybrid \
# --model_args pretrained="amd/Zebra-Llama-1B-8MLA-8Mamba-SFT" \
# --tasks ifeval   \
# --num_fewshot 0 --device cuda --batch_size 8 --apply_chat_template > logs/ifeval_hybrid_1B_8MLA_8M2_SFT.log 2>&1 &


# HIP_VISIBLE_DEVICES=2 CUDA_VISIBLE_DEVICES=2 \
# python benchmark/llm_eval/lm_harness_eval_latest.py \
# --model hybrid \
# --model_args pretrained="amd/Zebra-Llama-3B-6MLA-22Mamba-SFT" \
# --tasks ifeval   \
# --num_fewshot 0 --device cuda --batch_size 8 --apply_chat_template > logs/ifeval_hybrid_3B_6MLA_22M2_SFT.log 2>&1 &

# HIP_VISIBLE_DEVICES=3 CUDA_VISIBLE_DEVICES=3 \
# python benchmark/llm_eval/lm_harness_eval_latest.py \
# --model hybrid \
# --model_args pretrained="amd/Zebra-Llama-3B-14MLA-14Mamba-SFT" \
# --tasks ifeval   \
# --num_fewshot 0 --device cuda --batch_size 8 --apply_chat_template > logs/ifeval_hybrid_3B_14MLA_14M2_SFT.log 2>&1 &


# HIP_VISIBLE_DEVICES=4 CUDA_VISIBLE_DEVICES=4 \
# python benchmark/llm_eval/lm_harness_eval_latest.py \
# --model hybrid \
# --model_args pretrained="amd/Zebra-Llama-8B-8MLA-24Mamba-SFT" \
# --tasks ifeval   \
# --num_fewshot 0 --device cuda --batch_size 8 --apply_chat_template > logs/ifeval_hybrid_8B_8MLA_24M2_SFT.log 2>&1 &

# HIP_VISIBLE_DEVICES=5 CUDA_VISIBLE_DEVICES=5 \
# python benchmark/llm_eval/lm_harness_eval_latest.py \
# --model hybrid \
# --model_args pretrained="amd/Zebra-Llama-8B-16MLA-16Mamba-SFT" \
# --tasks ifeval   \
# --num_fewshot 0 --device cuda --batch_size 8 --apply_chat_template > logs/ifeval_hybrid_8B_16MLA_16M2_SFT.log 2>&1 &


# HIP_VISIBLE_DEVICES=6 CUDA_VISIBLE_DEVICES=6 \
# python benchmark/llm_eval/lm_harness_eval_latest.py \
# --model hf \
# --model_args pretrained="meta-llama/Llama-3.2-1B-Instruct" \
# --tasks ifeval   \
# --num_fewshot 0 --device cuda --batch_size 8 --apply_chat_template > logs/ifeval_hybrid_1B_llama.log 2>&1 &

# HIP_VISIBLE_DEVICES=7 CUDA_VISIBLE_DEVICES=7 \
# python benchmark/llm_eval/lm_harness_eval_latest.py \
# --model hf \
# --model_args pretrained="meta-llama/Llama-3.2-3B-Instruct" \
# --tasks ifeval   \
# --num_fewshot 0 --device cuda --batch_size 8 --apply_chat_template > logs/ifeval_hybrid_3B_llama.log 2>&1 &




# HIP_VISIBLE_DEVICES=5 CUDA_VISIBLE_DEVICES=5 \
# python benchmark/llm_eval/lm_harness_eval_latest.py \
# --model hybrid \
# --model_args pretrained="amd/Zebra-Llama-8B-16MLA-16Mamba-DPO" \
# --tasks gsm8k_cot_self_consistency \
# --num_fewshot 8 --device cuda --batch_size 8 --fewshot_as_multiturn --apply_chat_template > gsm8k_cot_self_consistency_hybrid_3B_14MLA_14M2.log 2>&1 &

# HF_ALLOW_CODE_EVAL=1 HIP_VISIBLE_DEVICES=5 CUDA_VISIBLE_DEVICES=5 \
# python benchmark/llm_eval/lm_harness_eval_latest.py \
# --model hybrid \
# --model_args pretrained="amd/Zebra-Llama-8B-8MLA-24Mamba-DPO" \
# --tasks humaneval_instruct \
# --num_fewshot 0 --device cuda --batch_size 8 --confirm_run_unsafe_code --apply_chat_template > humaneval_hybrid_8B_8MLA_24M2.log 2>&1 &

# HF_ALLOW_CODE_EVAL=1 HIP_VISIBLE_DEVICES=7 CUDA_VISIBLE_DEVICES=7 \
# python benchmark/llm_eval/lm_harness_eval_latest.py \
# --model hybrid \
# --model_args pretrained="amd/Zebra-Llama-8B-16MLA-16Mamba-DPO" \
# --tasks humaneval_instruct \
# --num_fewshot 0 --device cuda --batch_size 8 --confirm_run_unsafe_code > humaneval_hybrid_8B_16MLA_16M2.log 2>&1 &


# HF_ALLOW_CODE_EVAL=1 HIP_VISIBLE_DEVICES=6 CUDA_VISIBLE_DEVICES=6 \
# python benchmark/llm_eval/lm_harness_eval_latest.py \
# --model hf \
# --model_args pretrained="meta-llama/Llama-3.1-8B-Instruct" \
# --tasks humaneval_instruct \
# --num_fewshot 0 --device cuda --batch_size 8 --confirm_run_unsafe_code --apply_chat_template #> humaneval_llama8b.log 2>&1 &



# HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 \
# python benchmark/llm_eval/lm_harness_eval_latest.py \
# --model hybrid \
# --model_args pretrained="/home/mnt/mingyyan/checkpoints/hybrid_QWEN_7B_7B_mla_8_mamba20_Fix96_qr1536_qh64_stage2-dpo" \
# --tasks mmlu,hellaswag,piqa,arc_easy,arc_challenge,winogrande,openbookqa,pubmedqa,race \
# --num_fewshot 0 --device cuda --batch_size 16 > lm_eval_qwen.log 2>&1 &

# HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 \
# python benchmark/llm_eval/lm_harness_eval_latest.py \
# --model hf \
# --model_args pretrained="meta-llama/Llama-3.2-3B-Instruct" \
# --tasks longbench \
# --num_fewshot 0 --device cuda --batch_size 1 > logs/longbench_llama3b.log 2>&1 &

# HIP_VISIBLE_DEVICES=1 CUDA_VISIBLE_DEVICES=1 \
# python benchmark/llm_eval/lm_harness_eval_latest.py \
# --model hf \
# --model_args pretrained="meta-llama/Llama-3.1-8B-Instruct" \
# --tasks longbench \
# --num_fewshot 0 --device cuda --batch_size 1 > logs/longbench_llama8b.log 2>&1 &

# HIP_VISIBLE_DEVICES=3 CUDA_VISIBLE_DEVICES=3 \
# python benchmark/llm_eval/lm_harness_eval_latest.py \
# --model hybrid \
# --model_args pretrained="amd/Zebra-Llama-3B-14MLA-14Mamba-SFT" \
# --tasks longbench \
# --num_fewshot 0 --device cuda --batch_size 1 > logs/longbench_hybrid_3B_14MLA_14M2_sft.log 2>&1 &

# HIP_VISIBLE_DEVICES=4 CUDA_VISIBLE_DEVICES=4 \
# python benchmark/llm_eval/lm_harness_eval_latest.py \
# --model hybrid \
# --model_args pretrained="amd/Zebra-Llama-3B-6MLA-22Mamba-DPO" \
# --tasks longbench \
# --num_fewshot 0 --device cuda --batch_size 1 > logs/longbench_hybrid_3B_6MLA_22M2_dpo.log 2>&1 &

# HIP_VISIBLE_DEVICES=5 CUDA_VISIBLE_DEVICES=5 \
# python benchmark/llm_eval/lm_harness_eval_latest.py \
# --model hybrid \
# --model_args pretrained="amd/Zebra-Llama-3B-6MLA-22Mamba-SFT" \
# --tasks longbench \
# --num_fewshot 0 --device cuda --batch_size 1 > logs/longbench_hybrid_3B_6MLA_22M2_sft.log 2>&1 &

# HIP_VISIBLE_DEVICES=6 CUDA_VISIBLE_DEVICES=6 \
# python benchmark/llm_eval/lm_harness_eval_latest.py \
# --model hybrid \
# --model_args pretrained="/home/mnt/mingyyan/checkpoints/hybrid_QWEN_7B_7B_mla_8_mamba20_Fix96_qr1536_qh64_stage2-dpo" \
# --tasks longbench \
# --num_fewshot 0 --device cuda --batch_size 1 > logs/longbench_hybrid_qwen_7B_8MLA_20M2_dpo.log 2>&1 &

HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 \
python benchmark/llm_eval/lm_harness_eval_latest.py \
--model hf \
--model_args pretrained="meta-llama/Llama-3.2-1B-Instruct" \
--tasks longbench_e \
--num_fewshot 0 --device cuda --batch_size 1 --apply_chat_template \
--output_path logs/longbench_llama1b_chat \
--log_samples> logs/longbench_llama1b_chat.log 2>&1 &

# HIP_VISIBLE_DEVICES=1 CUDA_VISIBLE_DEVICES=1 \
# python benchmark/llm_eval/lm_harness_eval_latest.py \
# --model hybrid \
# --model_args pretrained="amd/Zebra-Llama-1B-4MLA-12Mamba-DPO" \
# --tasks longbench_e \
# --num_fewshot 0 --device cuda --batch_size 1 \
# --output_path logs/longbench_hybrid_1B_4MLA_12M2_dpo \
# --log_samples > logs/longbench_hybrid_1B_4MLA_12M2_dpo.log 2>&1 &

# HIP_VISIBLE_DEVICES=2 CUDA_VISIBLE_DEVICES=2 \
# python benchmark/llm_eval/lm_harness_eval_latest.py \
# --model hybrid \
# --model_args pretrained="amd/Zebra-Llama-1B-4MLA-12Mamba-SFT" \
# --tasks longbench_e \
# --num_fewshot 0 --device cuda --batch_size 1 \
# --output_path logs/longbench_hybrid_1B_4MLA_12M2_sft \
# --log_samples > logs/longbench_hybrid_1B_4MLA_12M2_sft.log 2>&1 &

# HIP_VISIBLE_DEVICES=3 CUDA_VISIBLE_DEVICES=3 \
# python benchmark/llm_eval/lm_harness_eval_latest.py \
# --model hybrid \
# --model_args pretrained="amd/Zebra-Llama-1B-8MLA-8Mamba-DPO" \
# --tasks longbench_e \
# --num_fewshot 0 --device cuda --batch_size 1 \
# --output_path logs/longbench_hybrid_1B_8MLA_8M2_dpo \
# --log_samples > logs/longbench_hybrid_1B_8MLA_8M2_dpo.log 2>&1 &

# HIP_VISIBLE_DEVICES=4 CUDA_VISIBLE_DEVICES=4 \
# python benchmark/llm_eval/lm_harness_eval_latest.py \
# --model hybrid \
# --model_args pretrained="amd/Zebra-Llama-1B-8MLA-8Mamba-SFT" \
# --tasks longbench_e \
# --num_fewshot 0 --device cuda --batch_size 1 \
# --output_path logs/longbench_hybrid_1B_8MLA_8M2_sft \
# --log_samples > logs/longbench_hybrid_1B_8MLA_8M2_sft.log 2>&1 &

# HIP_VISIBLE_DEVICES=5 CUDA_VISIBLE_DEVICES=5 \
# python benchmark/llm_eval/lm_harness_eval_latest.py \
# --model hf \
# --model_args pretrained="meta-llama/Llama-3.2-3B-Instruct" \
# --tasks longbench_e \
# --num_fewshot 0 --device cuda --batch_size 1 \
# --output_path logs/longbench_llama3b \
# --log_samples > logs/longbench_llama3b.log 2>&1 &

# HIP_VISIBLE_DEVICES=6 CUDA_VISIBLE_DEVICES=6 \
# python benchmark/llm_eval/lm_harness_eval_latest.py \
# --model hf \
# --model_args pretrained="meta-llama/Llama-3.1-8B-Instruct" \
# --tasks longbench_e \
# --num_fewshot 0 --device cuda --batch_size 1 \
# --output_path logs/longbench_llama8b \
# --log_samples > logs/longbench_llama8b.log 2>&1 &

# HIP_VISIBLE_DEVICES=7 CUDA_VISIBLE_DEVICES=7 \
# python benchmark/llm_eval/lm_harness_eval_latest.py \
# --model hybrid \
# --model_args pretrained="/home/mnt/mingyyan/checkpoints/hybrid_QWEN_7B_7B_mla_8_mamba20_Fix96_qr1536_qh64_stage2-dpo" \
# --tasks longbench_e \
# --num_fewshot 0 --device cuda --batch_size 1 \
# --output_path logs/longbench_hybrid_qwen_7B_8MLA_20M2_dpo \
# --log_samples > logs/longbench_hybrid_qwen_7B_8MLA_20M2_dpo.log 2>&1 &


# HIP_VISIBLE_DEVICES=7 CUDA_VISIBLE_DEVICES=7 \
# python benchmark/llm_eval/lm_harness_eval_latest.py \
# --model hybrid \
# --model_args pretrained="amd/Zebra-Llama-8B-8MLA-24Mamba-DPO" \
# --tasks toxigen \
# --num_fewshot 0 --device cuda --batch_size 8 > toxigen_hybrid_8B_8MLA_24M2.log 2>&1 &


# HIP_VISIBLE_DEVICES=3 CUDA_VISIBLE_DEVICES=3 \
# python benchmark/llm_eval/lm_harness_eval_latest.py \
# --model hybrid \
# --model_args pretrained="amd/Zebra-Llama-1B-8MLA-8Mamba-DPO" \
# --tasks gsm8k_cot_llama \
# --num_fewshot 8 --device cuda --batch_size 8 --fewshot_as_multiturn --apply_chat_template > gsm8k_hybrid_1B_8MLA_8M2.log 2>&1 &


# HIP_VISIBLE_DEVICES=1 CUDA_VISIBLE_DEVICES=1 \
# python benchmark/llm_eval/lm_harness_eval_latest.py \
# --model hybrid \
# --model_args pretrained="amd/Zebra-Llama-8B-16MLA-16Mamba-DPO" \
# --tasks gsm8k_cot_llama \
# --num_fewshot 8 --device cuda --batch_size 8 --fewshot_as_multiturn --apply_chat_template > gsm8k_hybrid_8B_16MLA_16M2.log 2>&1 &


# HIP_VISIBLE_DEVICES=4 CUDA_VISIBLE_DEVICES=4 \
# python benchmark/llm_eval/lm_harness_eval_latest.py \
# --model hf \
# --model_args pretrained="meta-llama/Llama-3.2-1B-Instruct" \
# --tasks toxigen \
# --num_fewshot 0 --device cuda --batch_size 8 > toxigen_llama_1b.log 2>&1 &


# HIP_VISIBLE_DEVICES=5 CUDA_VISIBLE_DEVICES=5 \
# python benchmark/llm_eval/lm_harness_eval_latest.py \
# --model hf \
# --model_args pretrained="meta-llama/Llama-3.2-3B-Instruct" \
# --tasks toxigen \
# --num_fewshot 0 --device cuda --batch_size 8 > toxigen_llama_3b.log 2>&1 &

# HIP_VISIBLE_DEVICES=6 CUDA_VISIBLE_DEVICES=6 \
# python benchmark/llm_eval/lm_harness_eval_latest.py \
# --model hf \
# --model_args pretrained="meta-llama/Llama-3.1-8B-Instruct" \
# --tasks toxigen \
# --num_fewshot 0 --device cuda --batch_size 8 > toxigen_llama_8b.log 2>&1 &

# HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 \
# python benchmark/llm_eval/lm_harness_eval_latest.py \
# --model hf \
# --model_args pretrained="meta-llama/Llama-3.2-1B-Instruct" \
# --tasks truthfulqa_mc1,truthfulqa_mc2 \
# --num_fewshot 0 --device cuda --batch_size 8 > truthfulQA_llama_1b.log 2>&1 &

# HIP_VISIBLE_DEVICES=2 CUDA_VISIBLE_DEVICES=2 \
# python benchmark/llm_eval/lm_harness_eval_latest.py \
# --model hf \
# --model_args pretrained="meta-llama/Llama-3.2-3B-Instruct" \
# --tasks truthfulqa_mc1,truthfulqa_mc2 \
# --num_fewshot 0 --device cuda --batch_size 8 > truthfulQA_llama_3b.log 2>&1 &

# HIP_VISIBLE_DEVICES=3 CUDA_VISIBLE_DEVICES=3 \
# python benchmark/llm_eval/lm_harness_eval_latest.py \
# --model hf \
# --model_args pretrained="meta-llama/Llama-3.1-8B-Instruct" \
# --tasks truthfulqa_mc1,truthfulqa_mc2 \
# --num_fewshot 0 --device cuda --batch_size 8 > truthfulQA_llama_8b.log 2>&1 &

# HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 \
# python benchmark/llm_eval/lm_harness_eval_latest.py \
# --model hybrid \
# --model_args pretrained="amd/Zebra-Llama-1B-4MLA-12Mamba-DPO" \
# --tasks toxigen   \
# --num_fewshot 0 --device cuda --batch_size 8 > toxigen_hybrid_1B_4MLA_12M2.log 2>&1 &