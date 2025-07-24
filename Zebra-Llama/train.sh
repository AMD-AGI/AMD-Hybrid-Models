





# 
# ACCELERATE_LOG_LEVEL=info accelerate launch --config_file configs/fsdp.yaml train_hybrid/train_distill.py configs/llama3.2_1B/mla_1bt_SFT.yaml
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file configs/fsdp_MLA_ILD.yaml train_hybrid/train_distill.py configs/debug_MLA_stage1.yaml
# ACCELERATE_LOG_LEVEL=info accelerate launch --config_file configs/fsdp.yaml train_hybrid/train_distill.py configs/llama3.2_1B/zebra_8MLA8M2_8bt_SFT.yaml
# ACCELERATE_LOG_LEVEL=info accelerate launch --config_file configs/fsdp.yaml train_hybrid/train_dpo.py configs/llama3.2_1B/zebra_8MLA8M2_8bt_DPO.yaml


# ACCELERATE_LOG_LEVEL=info accelerate launch --config_file configs/multi_gpu.yaml train_hybrid/train_distill.py configs/llama3q.2_1B/zebra_MLA_stage1.yaml

# ACCELERATE_LOG_LEVEL=info accelerate launch --config_file configs/multi_gpu.yaml train_hybrid/train_distill.py configs/llama3.1_8B/zebra_MLA_ILD.yaml

# ACCELERATE_LOG_LEVEL=info accelerate launch --config_file configs/fsdp.yaml train_hybrid/train_distill.py configs/llama3.2_1B/zebra_8MLA8M2_8bt_SFT.yaml