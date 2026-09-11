# Patches for the linear-block kernels

Both linear-block families need a small AMD/ROCm fix on top of their upstream release. `install.sh`
applies both automatically; this file records what they change and how to apply them by hand.

| Patch | Target | Needed for |
|---|---|---|
| `flash-linear-attention-amd.patch` | `3rdparty/flash-linear-attention` submodule | Gated DeltaNet (`linear_type: gdn`) on AMD |
| `mamba-ssm-rocm-num-stages.patch` | `state-spaces/mamba` v2.3.1 source clone | Mamba-2 (`linear_type: mamba`) on AMD |

## flash-linear-attention-amd.patch

Gated DeltaNet runs on [flash-linear-attention](https://github.com/fla-org/flash-linear-attention)
(FLA), vendored as a submodule under `3rdparty/flash-linear-attention`. The pinned revision is
recorded in exactly one place — the submodule entry in this repository — so there is no separate
commit hash to keep in sync.

AMD/ROCm compatibility fixes so GDN training works on MI300-class GPUs:

- **`fla/ops/gated_delta_rule/wy_fast.py`**: avoids a `b_kt * b_b[None, :]` encoding mismatch on AMD
  Triton by using `tl.trans(b_k * b_b[:, None])` instead.
- **`fla/modules/conv/triton/kernels.py`**: uses `NUM_WARPS_AUTOTUNE` (capped at 16 warps on AMD) for
  `causal_conv1d_bwd_kernel`, so the HIP kernel launch does not fail with "invalid argument".

Optional on NVIDIA; the patched code is equivalent there.

To apply by hand, from the repository root:

```bash
git submodule update --init 3rdparty/flash-linear-attention

# Apply from INSIDE the FLA repo. `git apply --directory=3rdparty/flash-linear-attention` from the
# outer repo silently no-ops, because that subdirectory is itself a git repository.
git -C 3rdparty/flash-linear-attention apply -p1 ../../patches/flash-linear-attention-amd.patch

pip install --no-deps -e 3rdparty/flash-linear-attention
```

To start over, reset the submodule with `git submodule update --force 3rdparty/flash-linear-attention`.

`FLA=pypi bash install.sh` installs unpatched FLA from PyPI instead. That is fine on NVIDIA; on AMD
GPUs it hits the kernel issues above.

## mamba-ssm-rocm-num-stages.patch

One hunk in `mamba_ssm/utils/determinism.py`: `autotune_configs()` rewrites every Triton autotune
config to `num_stages=1` when running on ROCm.

Without it, the Mamba-2 backward pass aborts on gfx950 under Triton 3.4.0 —
`_chunk_state_bwd_db_kernel` fails to lower to TTGIR and raises
`RuntimeError: PassManager::run failed` inside `make_ttgir`. `num_stages=1` disables the AMD
software pipeliner, which is what fails to lower, and leaves block sizes and warp counts untouched.
Setting `MAMBA_DETERMINISTIC=1` is *not* a workaround: it narrows autotuning to the single cheapest
config, which still fails.

The patch is a no-op on NVIDIA, where `torch.version.hip` is `None`, and can be turned off with
`MAMBA_ROCM_NUM_STAGES=0`.

To apply by hand, against a fresh clone of the version `install.sh` builds:

```bash
git clone --branch v2.3.1 --depth 1 https://github.com/state-spaces/mamba.git mamba
git -C mamba apply -p1 ../patches/mamba-ssm-rocm-num-stages.patch
MAMBA_FORCE_BUILD=TRUE pip install --no-build-isolation --no-deps ./mamba
```
