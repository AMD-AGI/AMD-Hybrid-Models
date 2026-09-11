#!/bin/bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
#################################################################################
#SBATCH --job-name=hybrid-multinode
#SBATCH --output=logs/slurm/multinode.%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node 8
#SBATCH --time=72:00:00
#
# Usage:
#   sbatch multinode.slurm <train_config.yaml>
#   TRAIN_CONFIG=configs/llama3.2_1B/zebra_4MLA12GDN_8bT_SFT_8k_combined.yaml sbatch multinode.slurm
#   For Mamba-2 variants (linear_type=mamba):
#     ACCELERATE_CONFIG=configs/fsdp.yaml TRAIN_CONFIG=configs/qwen3_1.7B/zebra_14MLA14M2_8bT_SFT_8k_combined.yaml sbatch multinode.slurm
#   For stage-1 ILD: ACCELERATE_CONFIG=configs/fsdp_GDN_ILD.yaml TRAIN_CONFIG=configs/llama3.2_1B/zebra_GDN_ILD.yaml sbatch multinode.slurm

# -----------------------------------------------------------------------------
# Environment (override before sbatch or export in shell)
# -----------------------------------------------------------------------------
export IMAGE_NAME="${IMAGE_NAME:=rocm/pytorch-training:v26.1}"
export CONTAINER_NAME="${CONTAINER_NAME:=hybrid-train-${SLURM_JOB_ID}}"
export TRAIN_SCRIPT="${TRAIN_SCRIPT:=train_hybrid/train_distill.py}"
export ACCELERATE_CONFIG="${ACCELERATE_CONFIG:=configs/fsdp_GDN.yaml}"
export NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:=8}"
export TRAIN_EXTRA_ARGS="${TRAIN_EXTRA_ARGS:=}"
# LOCAL_MOUNT: node-dependent path mounted as /mnt in the container. Set this to wherever
# your datasets and checkpoints live; it differs per cluster.
export LOCAL_MOUNT="${LOCAL_MOUNT:=$HOME}"
export WORKDIR="${WORKDIR:=/workspace}"

# NCCL fabric settings are site-specific: the interface and HCA names below must match
# your nodes (check `ip link` and `ibv_devinfo`). Override them for your cluster.
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-}"
export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
export NCCL_PROTO=Simple
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=NET,INIT,COLL
# NCCL_DEBUG_FILE is set per-node inside the container (see docker exec below) so each node writes its own log

export TRAIN_CONFIG="${TRAIN_CONFIG:=configs/llama3.2_1B/zebra_4MLA12GDN_8bT_SFT_8k_combined.yaml}"
# Repo directory (absolute path of directory containing this script)
export REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
echo "REPO_DIR: $REPO_DIR"
mkdir -p "$REPO_DIR/logs/slurm"
mkdir -p "$REPO_DIR/logs/nccl/$SLURM_JOB_ID"
export MASTER_ADDR=$(srun --ntasks=1 hostname | head -n 1)
export MASTER_PORT="${MASTER_PORT:-29440}"
export NUM_PROCESSES=$((SLURM_JOB_NUM_NODES * NUM_GPUS_PER_NODE))

: "${HF_TOKEN:?Set HF_TOKEN in your environment (do not hardcode tokens in scripts)}"
: "${WANDB_API_KEY:?Set WANDB_API_KEY in your environment (do not hardcode tokens in scripts)}"
export HF_TOKEN WANDB_API_KEY
# -----------------------------------------------------------------------------
# Pull image on all nodes so each node has the image before we start containers
# -----------------------------------------------------------------------------
echo "Pulling Docker image on all nodes: $IMAGE_NAME"
srun --nodes=$SLURM_JOB_NUM_NODES --ntasks=$SLURM_JOB_NUM_NODES docker pull "$IMAGE_NAME"

# -----------------------------------------------------------------------------
# Stop any existing container with the same name (e.g. from a previous run)
# -----------------------------------------------------------------------------
echo "Stopping any existing containers"
srun bash -c 'docker ps -aq | xargs -r docker stop 2>/dev/null || true; docker ps -aq | xargs -r docker rm -f 2>/dev/null || true'

# -----------------------------------------------------------------------------
# Start one container per node (LOCAL_MOUNT is node-dependent)
# -----------------------------------------------------------------------------
echo "Starting Docker containers on all nodes"
srun bash -c '
  LOCAL_MOUNT_NODE="${LOCAL_MOUNT:-$HOME}"
  docker run -d \
    --rm \
    --cap-add=SYS_PTRACE \
    --cap-add=CAP_SYS_ADMIN \
    --security-opt seccomp=unconfined \
    --user root \
    --device=/dev/kfd \
    --device=/dev/dri \
    --device=/dev/infiniband \
    --group-add video \
    --ipc=host \
    --privileged \
    --network=host \
    -v "$REPO_DIR:$WORKDIR" \
    -v "$LOCAL_MOUNT_NODE:/mnt" \
    --workdir "$WORKDIR" \
    --name "$CONTAINER_NAME" \
    --entrypoint "" \
    "$IMAGE_NAME" \
    tail -f /dev/null
'

# -----------------------------------------------------------------------------
# Run training inside container: accelerate launch with per-node args from env
# -----------------------------------------------------------------------------
echo "Launching training (accelerate) on all nodes"
srun bash -c "
  docker exec \
    -e MASTER_ADDR='$MASTER_ADDR' \
    -e MASTER_PORT='$MASTER_PORT' \
    -e MACHINE_RANK=\$SLURM_NODEID \
    -e NNODES=$SLURM_JOB_NUM_NODES \
    -e NCCL_SOCKET_IFNAME='${NCCL_SOCKET_IFNAME}' \
    -e NCCL_IB_HCA='${NCCL_IB_HCA}' \
    -e NCCL_IB_GID_INDEX='${NCCL_IB_GID_INDEX}' \
    -e NCCL_PROTO='${NCCL_PROTO}' \
    -e NCCL_DEBUG='${NCCL_DEBUG}' \
    -e NCCL_DEBUG_SUBSYS='${NCCL_DEBUG_SUBSYS}' \
    -e NUM_PROCESSES=$NUM_PROCESSES \
    -e TRAIN_SCRIPT='$TRAIN_SCRIPT' \
    -e TRAIN_CONFIG='$TRAIN_CONFIG' \
    -e ACCELERATE_CONFIG='$ACCELERATE_CONFIG' \
    -e REPO_DIR=/workspace \
    -e TRAIN_EXTRA_ARGS='$TRAIN_EXTRA_ARGS' \
    -e HF_TOKEN='$HF_TOKEN' \
    -e HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}" \
    -e WANDB_API_KEY='$WANDB_API_KEY' \
    -e SLURM_JOB_ID='$SLURM_JOB_ID' \
    -e PYTORCH_HIP_ALLOC_CONF=expandable_segments:True \
    $CONTAINER_NAME \
    bash -c '
      huggingface-cli login --token \$HF_TOKEN
      pip install -e 3rdparty/flash-linear-attention
      sleep 5
      cd \"\$REPO_DIR\"
      mkdir -p \"logs/nccl/\$SLURM_JOB_ID\"
      echo \"envs for rank \$MACHINE_RANK: \\n$(env) \\n\"
      export NCCL_DEBUG_FILE=\"logs/nccl/\$SLURM_JOB_ID/rank_\${MACHINE_RANK}_\$(hostname).log\"
      echo \"NCCL_DEBUG_FILE=\$NCCL_DEBUG_FILE\"
      ACCELERATE_LOG_LEVEL=info ACCELERATE_DEBUG_MODE=1 accelerate launch \
        --config_file \"\$ACCELERATE_CONFIG\" \
        --num_processes \"\$NUM_PROCESSES\" \
        --num_machines \"\$NNODES\" \
        --machine_rank \"\$MACHINE_RANK\" \
        --main_process_ip \"\$MASTER_ADDR\" \
        --main_process_port \"\$MASTER_PORT\" \
        \"\$TRAIN_SCRIPT\" \"\$TRAIN_CONFIG\" \$TRAIN_EXTRA_ARGS
    '
"

EXIT_CODE=$?

# -----------------------------------------------------------------------------
# Stop and remove containers on all nodes
# -----------------------------------------------------------------------------
echo "Stopping containers on all nodes"
srun --nodes=$SLURM_JOB_NUM_NODES --ntasks=$SLURM_JOB_NUM_NODES docker stop "$CONTAINER_NAME" 2>/dev/null || true
srun --nodes=$SLURM_JOB_NUM_NODES --ntasks=$SLURM_JOB_NUM_NODES docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

exit $EXIT_CODE
