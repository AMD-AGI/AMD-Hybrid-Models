#!/bin/bash

##########################################################################
FSDP_CONFIG=configs/fsdp.yaml

#--- Distilled SFT Example
MODEL_CONFIG=configs/llama3.2_1B/mla_kv_rank_64_1bt.yaml
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file ${FSDP_CONFIG} train_mla/train_distill.py ${MODEL_CONFIG}

#--- DPO Example
# MODEL_CONFIG=configs/dpo.yaml
# ACCELERATE_LOG_LEVEL=info accelerate launch --config_file ${FSDP_CONFIG} train_mla/train_dpo.py ${MODEL_CONFIG}

#--- Distilled Pretraining Example
# MODEL_CONFIG=configs/SmolLM_1.7B/mla_kv_rank_480_pretrain.yaml.yaml
# ACCELERATE_LOG_LEVEL=info accelerate launch --config_file ${FSDP_CONFIG} train_mla/train.py ${MODEL_CONFIG}
##########################################################################


# Extract the output dir from the config file
CHECKPOINT_DIR=$(python3 -c "import yaml; import sys;
with open(sys.argv[1], 'r') as f:
    data = yaml.safe_load(f);
    path = sys.argv[2].split('.');
    value = data;
    for key in path:
        value = value.get(key, None);
        if value is None:
            break;
    if value is not None:
        print(value);" "$MODEL_CONFIG" "output_dir")

echo $CHECKPOINT_DIR
pushd $CHECKPOINT_DIR

LAST_CHECKPOINT=$(find . -maxdepth 1 -type d -name "checkpoint-*" | sort -V | tail -n 1)
if [ -z "$LAST_CHECKPOINT" ]; then
    echo "No checkpoint directory found in $CHECKPOINT_DIR"
    popd
    continue
fi

# Extract checkpoint number for global_step directory
CHECKPOINT_NUM=$(echo "$LAST_CHECKPOINT" | sed 's/.*checkpoint-//')

echo "Processing $CHECKPOINT_DIR with checkpoint $CHECKPOINT_NUM"
# Convert model
accelerate merge-weights ${LAST_CHECKPOINT}/pytorch_model_fsdp_0/ ./
if [ -f "${CHECKPOINT_DIR}/model.safetensors" ]; then
    # Clean up
    rm -rf "$LAST_CHECKPOINT" "global_step$CHECKPOINT_NUM"
fi
popd