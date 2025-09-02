#!/bin/bash
###############################################################################
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
#################################################################################
#set -x

# parsing input arguments
for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)

   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"

   export "$KEY"="$VALUE"
done

FLASH_ATTN="${FLASH_ATTN:-0}"
UPDATE_PKG="${UPDATE_PKG:-1}"
MAMBA="${MAMBA:-0}"
REPO_PATH=$(pwd)

if [ "$UPDATE_PKG" -eq 1 ]; then
    cd $REPO_PATH

    git clone https://github.com/huggingface/alignment-handbook.git
    cd alignment-handbook/
    git checkout 606d2e9
    pip install .
    cd ..
    rm -rf alignment-handbook

    # setup lm-eval
    git clone https://github.com/EleutherAI/lm-evaluation-harness.git
    cd lm-evaluation-harness
    git checkout c9bbec6e7de418b9082379da82797522eb173054
    pip install .
    cd ..
    rm -rf lm-evaluation-harness

    pip install accelerate==0.34.1
    pip install huggingface-hub==0.24.5
    pip install trl==0.8.6
    pip install peft==0.12.0
    pip install transformers==4.43.1
    pip install triton==3.3.0
    pip install numpy==1.26.4
    pip install datasets==2.20.0

    pip install --upgrade 'optree>=0.13.0'

    pip install wandb

    cd $REPO_PATH
    trainsformer_path=$(python -c "import transformers; print(transformers.__path__[0])")
    deepspeed_path="${trainsformer_path//transformers/deepspeed}"

    cp patch/transformer_trainer.py $trainsformer_path/trainer.py
    cp patch/elastic_agent.py $deepspeed_path/elasticity/elastic_agent.py
fi

if [ "$FLASH_ATTN" -eq 1 ]; then
    pip uninstall flash_attn -y
    rm -rf flash-attention
    git clone https://github.com/Dao-AILab/flash-attention.git
    cd flash-attention/
    git checkout v2.7.4.post1
    MAX_JOBS=128 python setup.py install
fi

if [ "$MAMBA" -eq 1 ]; then
    #install for mamba environments
    cd $REPO_PATH
    git clone https://github.com/Dao-AILab/causal-conv1d.git
    cd causal-conv1d/
    python setup.py install
    cd ..
    rm -rf causal-conv1d

    git clone https://github.com/state-spaces/mamba.git mamba-pip
    cd mamba-pip/
    python setup.py install
    cd ..
    rm -rf mamba-pip
fi
