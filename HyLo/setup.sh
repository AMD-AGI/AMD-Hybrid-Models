###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
#################################################################################

git submodule update --init 3rdparty/flash-linear-attention

pip install --no-deps -e 3rdparty/flash-linear-attention
pip install 'trl==0.8.6' 'peft==0.12.0' 'transformers==4.52.4'

_align_loc="$(pip show alignment-handbook 2>/dev/null | awk '/^Location:/{print $2}')"
if [ -n "$_align_loc" ] && [ -f "$_align_loc/alignment/model_utils.py" ]; then
    # Patch every file under the alignment/ package (model_utils.py is the
    # only known offender today, but the grep-and-sed is cheap insurance).
    find "$_align_loc/alignment" -type f -name '*.py' -print0 \
        | xargs -0 sed -i 's|from huggingface_hub\.utils\._errors import|from huggingface_hub.errors import|g'
    echo "Patched alignment-handbook imports in $_align_loc/alignment"
fi

export HSA_NO_SCRATCH_RECLAIM=1
export PATH="/opt/venv/bin:$PATH"
# ----------------------------------------------------------------------------
# Hugging Face authentication.
#
# HF_TOKEN is expected to be forwarded into the container by the launch
# command, e.g. `docker run ... -e HF_TOKEN="$HF_TOKEN"`. We
# `unset HF_TOKEN` AFTER capturing it so that downstream calls to
# huggingface_hub use the on-disk credentials written by `hf auth login`
# (avoids accidental token leakage via env in subprocess logs).
# ----------------------------------------------------------------------------
if [ -n "${HF_TOKEN:-}" ]; then
    _hf_token_local="$HF_TOKEN"
    unset HF_TOKEN
    hf auth login --token "$_hf_token_local"
    unset _hf_token_local
else
    echo "[setup.sh] WARNING: HF_TOKEN is not set; skipping hf auth login."
    echo "[setup.sh] Private/gated dataset and model downloads will fail."
fi
# ----------------------------------------------------------------------------
# W&B authentication.
#
# transformers integration_utils.WandbCallback.on_train_begin() calls
# wandb.init(); without valid creds inside the container this fails with
#     wandb.errors.errors.CommError: Error uploading run: returned error 401:
#     {"message":"user is not logged in","extensions":{"code":"PERMISSION_ERROR"}}
# and the training loop crashes (since report_to: [wandb] is set in the
# training YAML). The container does not mount the host ~/.netrc, so we
# must explicitly log in here.
#
# WANDB_API_KEY is expected to be forwarded into the container by the
# launch command, e.g.:
#     docker run ... -e WANDB_API_KEY="$WANDB_API_KEY" ...
# (mirroring how HF_TOKEN is already forwarded). For an already-running
# container, you can also set it inside the shell before sourcing setup.sh:
#     export WANDB_API_KEY=<your-key> && source setup.sh
# ----------------------------------------------------------------------------
if [ -n "${WANDB_API_KEY:-}" ]; then
    wandb login --relogin "$WANDB_API_KEY"
else
    echo "[setup.sh] WARNING: WANDB_API_KEY is not set; skipping wandb login."
    echo "[setup.sh] If your training YAML has report_to: [wandb], training"
    echo "[setup.sh] will crash at trainer.train() with a 401 PERMISSION_ERROR."
    echo "[setup.sh] Either export WANDB_API_KEY before sourcing setup.sh,"
    echo "[setup.sh] forward it from the host via docker -e WANDB_API_KEY,"
    echo "[setup.sh] or set report_to: [] / WANDB_MODE=offline to disable wandb."
fi
