#!/bin/bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# install.sh - build the HyLo training environment on top of a public base image.
#
# Reference environment
# ---------------------
# The released checkpoints were trained in a container built from the public
#
#     rocm/pytorch-training:v26.1
#
# plus exactly the additions in this script. The base image already provides
# everything GPU-facing, and this script never modifies it:
#
#     torch       2.10.0.dev20251112+rocm7.1   (ROCm 7.1)
#     triton      built from ROCm/triton
#     flash_attn  2.8.3, built from ROCm/flash-attention for gfx942/gfx950
#     transformers 4.55.0, accelerate 1.9.0, trl 0.21.0, datasets 3.6.0,
#     numpy 1.26.4, wandb, huggingface_hub[cli]
#
# The HF stack is re-pinned below because the released runs were done against
# transformers 4.52.4 / trl 0.8.6 / peft 0.12.0, not the base image versions.
#
# Every pip install runs against a constraints file generated from the torch and
# triton already installed, so the resolver can never replace the ROCm build
# with a PyPI (CUDA) wheel. That substitution is the most common way to silently
# break one of these containers: it succeeds, then fails at the first kernel
# launch. The script is therefore also usable on a CUDA base image
# (e.g. nvcr.io/nvidia/pytorch:24.10-py3) - only the kernel builds below are
# ROCm-aware, via autodetection.
#
# Usage, inside the container and from this directory:
#
#     bash install.sh                # full environment (recommended)
#     bash install.sh MAMBA=0        # skip Mamba-2 kernels
#     bash install.sh FLA=0          # skip Gated DeltaNet kernels
#     bash install.sh FLASH_ATTN=1   # rebuild FlashAttention (not needed on v26.1)
#
###############################################################################

set -uo pipefail

# ---------------------------------------------------------------------------
# Pinned versions. These are the versions present in the reference container;
# changing one here is a deliberate change to the training environment.
# ---------------------------------------------------------------------------
TRANSFORMERS_VERSION="4.52.4"
TRL_VERSION="0.8.6"
PEFT_VERSION="0.12.0"
LIGER_VERSION="0.7.0"
ACCELERATE_COMMIT="13640d3242e73f01295f8a323351b2ad45d07d89"
ALIGNMENT_HANDBOOK_COMMIT="606d2e9"
MAMBA_TAG="v2.3.1"
CAUSAL_CONV1D_TAG="v1.6.1"
FLASH_ATTN_TAG="v2.7.4.post1"

# ---------------------------------------------------------------------------
# KEY=VALUE arguments
# ---------------------------------------------------------------------------
for ARGUMENT in "$@"; do
    KEY=$(echo "$ARGUMENT" | cut -f1 -d=)
    KEY_LENGTH=${#KEY}
    VALUE="${ARGUMENT:$KEY_LENGTH+1}"
    export "$KEY"="$VALUE"
done

UPDATE_PKG="${UPDATE_PKG:-1}"   # Python packages (HF stack, alignment-handbook)
MAMBA="${MAMBA:-1}"             # mamba-ssm + causal-conv1d, for the *M2* models
FLA="${FLA:-1}"                 # flash-linear-attention, for the *GDN* models
FLASH_ATTN="${FLASH_ATTN:-0}"   # off: v26.1 already ships FlashAttention 2.8.3
REPO_PATH="$(pwd)"
# Compile fan-out for the mamba-ssm / causal-conv1d kernels. Each hipcc job peaks
# at a couple of GB, so an unbounded $(nproc) on a many-core node can exhaust RAM
# and fail the build with a bare "Error compiling objects for extension". Raise it
# with MAX_JOBS=<n> if the machine has memory to spare.
_NPROC="$(nproc)"
MAX_JOBS="${MAX_JOBS:-$(( _NPROC > 32 ? 32 : _NPROC ))}"
export MAX_JOBS

PY="${PY:-python3}"

die() { echo "ERROR: $*" >&2; exit 1; }
step() { echo; echo "=== $* ==="; }

# ---------------------------------------------------------------------------
# Freeze the torch/triton ecosystem. Anything installed later is resolved
# against these pins, so torch is never up- or downgraded.
# ---------------------------------------------------------------------------
step "Detecting base image"

CONSTRAINTS="$(mktemp /tmp/hylo-torch-constraints-XXXXXX.txt)"
trap 'rm -f "$CONSTRAINTS"' EXIT

"$PY" - <<'PYEOF' > "$CONSTRAINTS" 2>/dev/null
import re, subprocess, sys

protect = {
    "torch", "torchvision", "torchaudio",
    "triton", "pytorch-triton", "pytorch-triton-rocm",
}
try:
    frozen = subprocess.check_output(
        [sys.executable, "-m", "pip", "list", "--format=freeze"],
        text=True, stderr=subprocess.DEVNULL,
    )
except Exception:
    sys.exit(0)
for line in frozen.splitlines():
    m = re.match(r"^([A-Za-z0-9_.-]+)==(.+)$", line)
    if m and m.group(1).lower().replace("_", "-") in protect:
        print(line)
PYEOF

[ -s "$CONSTRAINTS" ] || die "no torch found in this environment; are you inside the container?"

echo "Frozen (will not be modified):"
sed 's/^/    /' "$CONSTRAINTS"

PIP=("$PY" -m pip install -c "$CONSTRAINTS")

# ROCm vs CUDA: only affects how the fused kernels below are compiled.
if "$PY" -c 'import torch, sys; sys.exit(0 if torch.version.hip else 1)' 2>/dev/null; then
    # gfx942 = MI300X/MI325X, gfx950 = MI355X. Match the reference image.
    export PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-gfx942;gfx950}"
    export HIP_ARCHITECTURES="${HIP_ARCHITECTURES:-gfx942,gfx950}"
    echo "    platform: ROCm (PYTORCH_ROCM_ARCH=$PYTORCH_ROCM_ARCH)"
else
    echo "    platform: CUDA"
fi

# ---------------------------------------------------------------------------
# Python packages
# ---------------------------------------------------------------------------
if [ "$UPDATE_PKG" -eq 1 ]; then
    step "alignment-handbook @ $ALIGNMENT_HANDBOOK_COMMIT"

    # --no-deps is required: the handbook pins an old transformers/accelerate
    # generation that would be resolved over the versions pinned below.
    rm -rf "$REPO_PATH/alignment-handbook"
    git clone --quiet https://github.com/huggingface/alignment-handbook.git "$REPO_PATH/alignment-handbook"
    git -C "$REPO_PATH/alignment-handbook" checkout --quiet "$ALIGNMENT_HANDBOOK_COMMIT"
    "${PIP[@]}" --no-deps "$REPO_PATH/alignment-handbook" || die "alignment-handbook install failed"
    rm -rf "$REPO_PATH/alignment-handbook"

    # huggingface_hub >= 0.30 removed utils._errors, which the handbook imports.
    _align_dir="$("$PY" -c 'import alignment, os; print(os.path.dirname(alignment.__file__))' 2>/dev/null)"
    if [ -n "$_align_dir" ] && [ -d "$_align_dir" ]; then
        find "$_align_dir" -type f -name '*.py' -print0 \
            | xargs -0 sed -i 's|from huggingface_hub\.utils\._errors import|from huggingface_hub.errors import|g'
        echo "Patched huggingface_hub imports in $_align_dir"
    fi
fi

# ---------------------------------------------------------------------------
# Gated DeltaNet kernels
# ---------------------------------------------------------------------------
if [ "$FLA" = "1" ] || [ "$FLA" = "submodule" ]; then
    step "flash-linear-attention (submodule + AMD patch)"

    FLA_DIR="$REPO_PATH/3rdparty/flash-linear-attention"
    git -C "$REPO_PATH" submodule update --init 3rdparty/flash-linear-attention \
        || echo "NOTE: submodule update failed; using 3rdparty/flash-linear-attention as-is."

    [ -d "$FLA_DIR" ] || die "$FLA_DIR is missing"

    # 'git apply --directory=' from the outer repo silently no-ops when the
    # target subdir is itself a git repo, so apply from inside the FLA repo.
    for _patch in flash-linear-attention-amd.patch; do
        [ -f "$REPO_PATH/patches/$_patch" ] || continue
        if git -C "$FLA_DIR" apply --check -p1 "$REPO_PATH/patches/$_patch" 2>/dev/null; then
            git -C "$FLA_DIR" apply -p1 "$REPO_PATH/patches/$_patch"
            echo "  $_patch applied."
        elif git -C "$FLA_DIR" apply --reverse --check -p1 "$REPO_PATH/patches/$_patch" 2>/dev/null; then
            echo "  $_patch already applied."
        else
            die "$_patch does not apply to $FLA_DIR and is not already applied. Reset with 'git submodule update --force 3rdparty/flash-linear-attention'."
        fi
    done

    # --no-deps: FLA's requirements would pull a newer transformers.
    "${PIP[@]}" --no-deps "$FLA_DIR" || die "flash-linear-attention install failed"
elif [ "$FLA" = "pypi" ]; then
    step "flash-linear-attention (PyPI, unpatched)"
    "${PIP[@]}" --no-deps flash-linear-attention || die "flash-linear-attention install failed"
fi

# ---------------------------------------------------------------------------
# Mamba-2 kernels
#
# --no-build-isolation is essential: with isolation, pip builds these against a
# freshly downloaded torch from PyPI instead of the ROCm torch in the image.
# ---------------------------------------------------------------------------
if [ "$MAMBA" -eq 1 ]; then
    step "causal-conv1d $CAUSAL_CONV1D_TAG and mamba-ssm $MAMBA_TAG (source build, MAX_JOBS=$MAX_JOBS)"

    export CAUSAL_CONV1D_FORCE_BUILD=TRUE
    export MAMBA_FORCE_BUILD=TRUE

    rm -rf "$REPO_PATH/causal-conv1d" "$REPO_PATH/mamba"

    git clone --quiet --branch "$CAUSAL_CONV1D_TAG" --depth 1 \
        https://github.com/Dao-AILab/causal-conv1d.git "$REPO_PATH/causal-conv1d"
    "${PIP[@]}" --no-build-isolation "$REPO_PATH/causal-conv1d" || die "causal-conv1d build failed"
    rm -rf "$REPO_PATH/causal-conv1d"

    git clone --quiet --branch "$MAMBA_TAG" --depth 1 \
        https://github.com/state-spaces/mamba.git "$REPO_PATH/mamba"
    # Required on ROCm: without it the Mamba-2 backward pass dies in Triton's
    # AMD backend. See patches/README.md.
    git -C "$REPO_PATH/mamba" apply -p1 "$REPO_PATH/patches/mamba-ssm-rocm-num-stages.patch" \
        || die "mamba-ssm-rocm-num-stages.patch does not apply to mamba $MAMBA_TAG"
    echo "  mamba-ssm-rocm-num-stages.patch applied."
    "${PIP[@]}" --no-build-isolation --no-deps "$REPO_PATH/mamba" || die "mamba-ssm build failed"
    rm -rf "$REPO_PATH/mamba"
fi

# ---------------------------------------------------------------------------
# FlashAttention. v26.1 already ships 2.8.3 built for gfx942/gfx950, so this is
# opt-in and only useful on a base image without it.
# ---------------------------------------------------------------------------
if [ "$FLASH_ATTN" -eq 1 ]; then
    step "FlashAttention $FLASH_ATTN_TAG (source build)"
    "$PY" -m pip uninstall -y flash_attn
    rm -rf "$REPO_PATH/flash-attention"
    # --recurse-submodules: the ROCm backend needs csrc/composable_kernel.
    git clone --quiet --branch "$FLASH_ATTN_TAG" --depth 1 --recurse-submodules \
        https://github.com/Dao-AILab/flash-attention.git "$REPO_PATH/flash-attention"
    "${PIP[@]}" --no-build-isolation "$REPO_PATH/flash-attention" || die "FlashAttention build failed"
    rm -rf "$REPO_PATH/flash-attention"
fi

# ---------------------------------------------------------------------------
# HF stack, pinned last so it wins over any transitive requirement pulled in
# above. This is the authoritative pin for the training environment.
# ---------------------------------------------------------------------------
if [ "$UPDATE_PKG" -eq 1 ]; then
    step "HF stack (authoritative pins)"

    "${PIP[@]}" \
        "transformers==$TRANSFORMERS_VERSION" \
        "trl==$TRL_VERSION" \
        "peft==$PEFT_VERSION" \
        "liger-kernel==$LIGER_VERSION" \
        || die "HF stack install failed"

    # Pinned to the commit used for the released runs; the 1.9.0 release in the
    # base image predates the FSDP fixes these runs depend on.
    "${PIP[@]}" "accelerate @ git+https://github.com/huggingface/accelerate.git@$ACCELERATE_COMMIT" \
        || die "accelerate install failed"

    # DeepSpeed elastic-agent replacement, only if DeepSpeed is present.
    # v26.1 does not ship DeepSpeed; the FSDP configs in configs/ do not need it.
    _ds_dir="$("$PY" -c 'import deepspeed, os; print(os.path.dirname(deepspeed.__file__))' 2>/dev/null)"
    if [ -n "$_ds_dir" ] && [ -d "$_ds_dir/elasticity" ]; then
        cp "$REPO_PATH/patch/elastic_agent.py" "$_ds_dir/elasticity/elastic_agent.py"
        echo "Applied patch/elastic_agent.py to $_ds_dir/elasticity/"
    else
        echo "DeepSpeed not installed; skipping patch/elastic_agent.py (not needed for the FSDP configs)."
    fi
fi

# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
step "Installed environment"

"$PY" - <<'PYEOF'
import importlib.metadata as md

# name -> version in the reference container, or None if base-image supplied
expected = {
    "torch": None,
    "triton": None,
    "flash_attn": None,
    "transformers": "4.52.4",
    "accelerate": "1.14.0.dev0",
    "trl": "0.8.6",
    "peft": "0.12.0",
    "liger_kernel": "0.7.0",
    "alignment-handbook": "0.4.0.dev0",
    "flash-linear-attention": "0.4.2",
    "mamba_ssm": "2.3.1",
    "causal_conv1d": "1.6.1",
}

problems = []
for name, want in expected.items():
    try:
        got = md.version(name)
    except md.PackageNotFoundError:
        got = None

    if got is None:
        status = "MISSING" if want else "not installed"
        if want:
            problems.append(f"{name}: missing (expected {want})")
    elif want is None or got == want:
        status = "ok"
    else:
        status = f"expected {want}"
        problems.append(f"{name}: {got}, expected {want}")

    print(f"  {name:<24} {str(got or '-'):<28} {status}")

print()
if problems:
    print("Mismatches against the reference environment:")
    for p in problems:
        print(f"  - {p}")
    print("\nThese are not necessarily fatal, but the released numbers were")
    print("produced with the versions above.")
else:
    print("Environment matches the reference container.")
PYEOF

echo
echo "Next: source setup.sh (HF/W&B login and runtime env), then bash train.sh"
