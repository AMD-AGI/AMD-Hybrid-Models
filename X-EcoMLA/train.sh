



ACCELERATE_LOG_LEVEL=info accelerate launch --config_file deepspeed_zero3.yaml train_mla/train_distill.py configs/SmolLM_135M/mla_kv_rank_16_1.7bt.yaml