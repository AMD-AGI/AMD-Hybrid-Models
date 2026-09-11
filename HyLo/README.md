# HyLo: Long-Context Aware Upcycling for Hybrid LLMs

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![arXiv](https://img.shields.io/badge/arXiv-2604.24715-b31b1b.svg)][paper]
[![Models](https://img.shields.io/badge/%F0%9F%A4%97%20models-HyLo%20collection-yellow.svg)][models]

Convert a pretrained Transformer LLM into a long-context hybrid model, without pretraining from scratch.

HyLo (**Hy**brid **Lo**ng-context) is an upcycling recipe: it takes an existing instruction-tuned
Transformer and converts every layer into one of two cheaper types. Attention layers become
**Multi-head Latent Attention (MLA)**, which caches a low-rank latent instead of full keys and
values. The remaining layers become **linear blocks** — **Gated DeltaNet** or **Mamba-2** — which
carry a fixed-size recurrent state and no KV cache at all. The converted model is then trained in
two staged passes with teacher-guided distillation, which extends usable context by up to 32x and
cuts KV-cache memory by more than 90% while holding short-context accuracy.

This directory contains the upcycling and training code for the models reported in our paper
([paper][paper]). All 14 released checkpoints are in the [HyLo model collection][models].
The sibling directories in this repository hold the predecessor projects the recipe builds on:
[`../X-EcoMLA`](../X-EcoMLA) (upcycling attention into MLA) and
[`../Zebra-Llama`](../Zebra-Llama) (hybrid MLA + Mamba-2 models).

## Highlights

- **KV cache down to 2–7.8% of the backbone.** MLA caches a compressed latent; linear layers cache nothing.
- **Context extended up to 32x** over the backbone's original window through post-training alone (8K and 64K variants released).
- **Two backbones, two linear blocks.** Llama-3.2-1B/3B and Qwen3-1.7B, each with Gated DeltaNet or Mamba-2.
- **No pretraining required.** ~10B tokens of upcycling reproduces the released checkpoints on 8 AMD Instinct™ MI300X GPUs.
- **Competitive at small scale.** HyLo-Qwen-1.7B, upcycled on ~10B tokens, outperforms JetNemotron at similar scale on GSM8K, lm-harness commonsense reasoning and RULER-64K despite JetNemotron's far larger token budget.

## Released checkpoints

All checkpoints are hybrid conversions of an instruction-tuned backbone, distilled from a larger
teacher in the same family (Llama-3.1-8B-Instruct for the Llama models, Qwen3-8B for the Qwen models).
Names encode the layer layout: `4MLA12GDN` is 4 MLA layers and 12 Gated DeltaNet layers.

| Model | Backbone | Layer layout | Linear block | Trained context | KV cache |
|---|---|---|---|---|---|
| [`HyLo-Llama-4MLA12GDN-8K-SFT`][models] | Llama-3.2-1B-Instruct | 4 MLA + 12 GDN | Gated DeltaNet | 8K | 3.9% |
| [`HyLo-Llama-4MLA12GDN-64K-SFT`][models] | Llama-3.2-1B-Instruct | 4 MLA + 12 GDN | Gated DeltaNet | 64K | 3.9% |
| [`HyLo-Llama-8MLA8GDN-8K-SFT`][models] | Llama-3.2-1B-Instruct | 8 MLA + 8 GDN | Gated DeltaNet | 8K | 7.8% |
| [`HyLo-Llama-8MLA8GDN-64K-SFT`][models] | Llama-3.2-1B-Instruct | 8 MLA + 8 GDN | Gated DeltaNet | 64K | 7.8% |
| [`HyLo-Llama-6MLA22GDN-8K-SFT`][models] | Llama-3.2-3B-Instruct | 6 MLA + 22 GDN | Gated DeltaNet | 8K | 2.0% |
| [`HyLo-Llama-6MLA22GDN-64K-SFT`][models] | Llama-3.2-3B-Instruct | 6 MLA + 22 GDN | Gated DeltaNet | 64K | 2.0% |
| [`HyLo-Llama-14MLA14GDN-8K-SFT`][models] | Llama-3.2-3B-Instruct | 14 MLA + 14 GDN | Gated DeltaNet | 8K | 4.7% |
| [`HyLo-Llama-14MLA14GDN-64K-SFT`][models] | Llama-3.2-3B-Instruct | 14 MLA + 14 GDN | Gated DeltaNet | 64K | 4.7% |
| [`HyLo-Qwen-7MLA21GDN-8K-SFT`][models] | Qwen3-1.7B | 7 MLA + 21 GDN | Gated DeltaNet | 8K | 3.9% |
| [`HyLo-Qwen-7MLA21GDN-64K-SFT`][models] | Qwen3-1.7B | 7 MLA + 21 GDN | Gated DeltaNet | 64K | 3.9% |
| [`HyLo-Qwen-14MLA14GDN-8K-SFT`][models] | Qwen3-1.7B | 14 MLA + 14 GDN | Gated DeltaNet | 8K | 7.8% |
| [`HyLo-Qwen-14MLA14GDN-64K-SFT`][models] | Qwen3-1.7B | 14 MLA + 14 GDN | Gated DeltaNet | 64K | 7.8% |
| [`HyLo-Qwen-14MLA14M2-8K-SFT`][models] | Qwen3-1.7B | 14 MLA + 14 M2 | Mamba-2 | 8K | 7.8% |
| [`HyLo-Qwen-14MLA14M2-64K-SFT`][models] | Qwen3-1.7B | 14 MLA + 14 M2 | Mamba-2 | 64K | 7.8% |

"KV cache" is the fraction of the backbone's KV-cache footprint at the same sequence length. Each
model card in the [collection][models] carries the full per-task numbers, the training recipe and
the exact data mixture for that checkpoint.

## Results

Commonsense is the 0-shot average over ARC-Challenge, ARC-Easy, HellaSwag, OpenBookQA, PIQA, RACE
and WinoGrande. RULER is the average over all 13 tasks at each sequence length. GSM8K is 0-shot.
All numbers are from the paper.

| Model | Commonsense avg | GSM8K | RULER 8K | 16K | 32K | 64K |
|---|---|---|---|---|---|---|
| `HyLo-Llama-4MLA12GDN-8K-SFT` | 53.1 | 51.9 | 55.1 | 11.9 | 2.4 | 0.8 |
| `HyLo-Llama-4MLA12GDN-64K-SFT` | 51.2 | 37.5 | 52.5 | 48.3 | 44.5 | 40.8 |
| `HyLo-Llama-8MLA8GDN-8K-SFT` | 53.4 | 54.6 | 60.3 | 0.5 | 0.1 | 0.1 |
| `HyLo-Llama-8MLA8GDN-64K-SFT` | 51.8 | 39.4 | 61.5 | 53.7 | 48.1 | 41.6 |
| `HyLo-Llama-6MLA22GDN-8K-SFT` | 59.0 | 68.1 | 71.2 | 45.0 | 27.1 | 14.1 |
| `HyLo-Llama-6MLA22GDN-64K-SFT` | 57.2 | 56.0 | 68.2 | 62.1 | 55.7 | 46.3 |
| `HyLo-Llama-14MLA14GDN-8K-SFT` | 59.2 | 68.2 | 71.1 | 45.6 | 19.2 | 0.2 |
| `HyLo-Llama-14MLA14GDN-64K-SFT` | 57.9 | 58.9 | 73.2 | 69.7 | 62.9 | 52.0 |
| `HyLo-Qwen-7MLA21GDN-8K-SFT` | 56.1 | 76.0 | 63.5 | 43.6 | 30.3 | 17.4 |
| `HyLo-Qwen-7MLA21GDN-64K-SFT` | 55.4 | 73.3 | 59.8 | 53.8 | 42.5 | 30.5 |
| `HyLo-Qwen-14MLA14GDN-8K-SFT` | 56.5 | 76.1 | 71.1 | 45.6 | 19.2 | 0.2 |
| `HyLo-Qwen-14MLA14GDN-64K-SFT` | 56.7 | 73.8 | 66.9 | 53.2 | 41.4 | 31.6 |
| `HyLo-Qwen-14MLA14M2-8K-SFT` | 56.1 | 75.8 | 74.2 | 58.6 | 33.5 | 10.7 |
| `HyLo-Qwen-14MLA14M2-64K-SFT` | 55.7 | 73.5 | 73.9 | 62.6 | 46.2 | 33.1 |

The pattern to read here: the 8K models are the stronger short-context models, the 64K models are
the ones that hold up as the context grows. Training at 8K leaves RULER collapsing past 16K;
training at 64K trades a little commonsense and GSM8K accuracy for usable long-context behaviour.
Pick the variant that matches your sequence length.

## Installation

We strongly recommend running inside a container. The released checkpoints were trained on AMD
Instinct™ MI300X GPUs in a container built from the public `rocm/pytorch-training:v26.1` image plus
`install.sh`, and the same code has been run on NVIDIA H100/H200 from
`nvcr.io/nvidia/pytorch:24.10-py3`.

`rocm/pytorch-training:v26.1` supplies the entire GPU stack, so `install.sh` never touches it:
ROCm 7.1, `torch` 2.10.0 nightly, Triton built from `ROCm/triton`, and FlashAttention 2.8.3 built
from `ROCm/flash-attention` for gfx942/gfx950.

```bash
git clone https://github.com/AMD-AGI/AMD-Hybrid-Models.git
cd AMD-Hybrid-Models/HyLo
```

On AMD GPUs:

```bash
docker run -it \
  --device /dev/dri --device /dev/kfd --device /dev/infiniband \
  --network host --ipc host --group-add video --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined --privileged \
  -v $HOME:$HOME --shm-size 64G --name hylo \
  rocm/pytorch-training:v26.1

# inside the container, from this directory
bash install.sh
```

The defaults install the full environment, including both linear-block kernel families, and the
script ends with a table comparing what you got against the reference environment. `install.sh`
takes `KEY=VALUE` arguments if you want less than that:

| Flag | Default | Effect |
|---|---|---|
| `MAMBA` | `1` | Build `mamba-ssm` 2.3.1 (with the ROCm patch in `patches/`) and `causal-conv1d` 1.6.1 from source; needed for the Mamba-2 models (`*M2*`) |
| `FLA` | `1` | Install `3rdparty/flash-linear-attention` at the pinned revision with the patch in `patches/`; needed for the Gated DeltaNet models (`*GDN*`), and required for GDN on ROCm |
| `FLA` | `pypi` | Install `flash-linear-attention` from PyPI, unpatched (NVIDIA only) |
| `FLA` | `0` | Skip the Gated DeltaNet kernels |
| `FLASH_ATTN` | `0` | Build FlashAttention 2.7.4 from source. Not needed on `v26.1`, which already ships 2.8.3 |
| `UPDATE_PKG` | `1` | Install and pin the Python packages (`transformers==4.52.4`, `trl==0.8.6`, `peft==0.12.0`, `liger-kernel==0.7.0`, `accelerate` from a pinned commit, and `alignment-handbook`) |

Two details in `install.sh` are load-bearing rather than stylistic, and are worth preserving if you
adapt it. Every `pip install` is resolved against a constraints file generated from the `torch` and
Triton already in the image, so the resolver cannot quietly replace the ROCm build with a PyPI CUDA
wheel; and the fused kernels are built with `--no-build-isolation` for the same reason, since an
isolated build environment would fetch its own `torch` to compile against. Both failure modes look
like a successful install and then break at the first kernel launch.

Both linear-block kernel families also need a one-hunk AMD fix, applied by `install.sh` and
described in [patches/README.md](patches/README.md): one for the flash-linear-attention Gated
DeltaNet kernels, and one that disables Triton's AMD software pipeliner in `mamba-ssm`, without
which the Mamba-2 backward pass fails to compile on gfx950.

## Loading a checkpoint

Checkpoints load through `HybridModelWrapper`, which reads the hybrid layer plan from
`hybrid_config.json` and builds the matching MLA and linear layers. This is the same loader the
training code uses to initialise a stage-2 run from its stage-1 checkpoints.

```python
import torch
from transformers import AutoTokenizer
from hybrid.hybrid_wrapper import HybridModelWrapper

model_id = "amd/HyLo-Llama-4MLA12GDN-8K-SFT"  # any checkpoint from the table above

model = HybridModelWrapper.from_pretrained(model_id, torch_dtype=torch.bfloat16)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_id)

messages = [{"role": "user", "content": "Explain KV-cache compression in two sentences."}]
inputs = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt"
).cuda()

out = model.generate(inputs, max_new_tokens=256, do_sample=False)
print(tokenizer.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True))
```

Load in `bfloat16`: that is the precision every number in this README was measured at. The
published checkpoints store `float32` weights because that is what training wrote out, so loading
them as-is doubles the memory footprint for no accuracy benefit.

## Training

HyLo upcycles in two stages. Both are driven by `train_hybrid/train_distill.py` with `accelerate`
and FSDP; the YAML files under `configs/` hold the model plan, data mixture and hyperparameters.

**Stage 1 — Enhanced-ILD.** Initialise the new MLA and linear layers by matching the teacher's
intermediate hidden states (L2 loss) at 2K context on 20% of the mixture, LR 2e-4.

```bash
ACCELERATE_LOG_LEVEL=info accelerate launch \
  --config_file configs/fsdp_GDN_ILD.yaml \
  train_hybrid/train_distill.py \
  configs/llama3.2_1B/zebra_GDN_ILD.yaml
```

**Stage 2 — Long-context SFT with teacher-guided distillation.** Train the whole hybrid on the full
mixture at the target context length against the teacher's next-token distribution (pure KL,
`kl_weight 1.0`, `ce_weight 0.0`), with YaRN scaling over the backbone's original window.

```bash
ACCELERATE_LOG_LEVEL=info accelerate launch \
  --config_file configs/fsdp_GDN.yaml \
  train_hybrid/train_distill.py \
  configs/llama3.2_1B/zebra_4MLA12GDN_8bT_SFT_8k_combined.yaml
```

Swap the config for the variant you want: `configs/llama3.2_1B/`, `configs/llama3.2_3B/` and
`configs/qwen3_1.7B/` hold the ILD and SFT recipes, at both 8K and 64K.
`train.sh` and `multinode.slurm` are worked examples for single-node and multi-node launches.
`multinode.slurm` carries site-specific defaults (container image, mounts, NCCL fabric settings)
that you will need to point at your own cluster.

Stage 2 starts from the stage-1 output. Every config uses repo-relative checkpoint paths: a stage-1
recipe writes `output_dir: checkpoints/<name>`, and the stage-2 recipes read that same path through
`linear_ILD_path` and `mla_ILD_path`. Running the two stages in order therefore needs no path
edits — for example `configs/llama3.2_1B/zebra_GDN_ILD.yaml` and
`configs/llama3.2_1B/zebra_MLA_ILD.yaml` produce `checkpoints/Zebra-1B-ILD-GDN` and
`checkpoints/Zebra-1B-ILD-MLA`, which are exactly what the 1B SFT configs consume. Point
`output_dir` somewhere else if you prefer, and update the corresponding `*_ILD_path` to match.

Both stages ran on 8 AMD Instinct™ MI300X GPUs in bfloat16 mixed precision with FSDP full sharding.
After training, run `accelerate merge-weights` to produce a single `model.safetensors`.

## Training data

The stage-1 ILD configs train on [`JunxiongWang/sftdatasetv3`](https://huggingface.co/datasets/JunxiongWang/sftdatasetv3)
alone, and the shipped `dataset_mixer` blocks run as-is.

The stage-2 SFT mixture used for the released checkpoints also drew on four datasets that AMD
preprocessed internally. **Those four are not part of this release**, so the stage-2 configs ship with
only the public general-SFT set enabled. Each removed dataset was a reformatted variant of a public
upstream, listed here with the licence you are bound by if you rebuild it:

| Upstream | Licence | Role in the mixture | AMD variant |
|---|---|---|---|
| [`nvidia/OpenMathInstruct-2`](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2) | cc-by-4.0 | Math instruction following | 2M-example subset |
| [`open-thoughts/OpenThoughts-114k`](https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k) | apache-2.0 | Math reasoning traces | math split only |
| [`open-r1/OpenR1-Math-220k`](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k) | apache-2.0 | Math reasoning traces | full set |
| [`nvidia/ChatQA2-Long-SFT-data`](https://huggingface.co/datasets/nvidia/ChatQA2-Long-SFT-data) | **cc-by-nc-2.0** | Long-context QA; this is the data that teaches the 64K behaviour | `long_sft_QA` split only |

The preprocessing applied to each was the same in shape: take the upstream split, normalise every
example into the `messages` role/content form that the `chat_template` in each config expects, drop
examples that fail to convert or that exceed the run's `max_seq_length` after templating, and
subsample where a row budget is noted above. No new supervision was synthesised and no examples were
rewritten — the transformation is format and length filtering, not content.

To reproduce the published mixture, prepare the four upstreams that way and add them back to
`dataset_mixer` alongside the public set, each at weight `1.0`; every stage-2 config carries the same
list as a comment next to the mixer block. Expect small differences from the paper numbers unless your
conversion and length filtering match ours exactly.

Note that `nvidia/ChatQA2-Long-SFT-data` is non-commercial. That constrains any model you train with
the full mixture, regardless of this repository's Apache-2.0 licence.

## Repository layout

| Path | Contents |
|---|---|
| `hybrid/` | Hybrid model definition: MLA, Gated DeltaNet and Mamba-2 layers, cache, loader |
| `train_hybrid/` | Distillation training entry point |
| `trainer/` | Trainer subclass used by the distillation runs |
| `configs/` | FSDP/accelerate configs and the training recipes for the released models |
| `patches/` | AMD/ROCm patches for the flash-linear-attention submodule and `mamba-ssm` |
| `patch/` | DeepSpeed elastic-agent replacement, applied by `install.sh` only when DeepSpeed is present; the FSDP configs do not need it |

`configs/` holds exactly the recipes behind the release: a stage-1 ILD recipe per backbone and block
type, and the stage-2 SFT recipe for each of the 14 published checkpoints.

## Scope of this release

This directory covers upcycling and training only. The evaluation harness integration, the
needle-in-a-haystack sweep, the inference and speed benchmarks and the vLLM serving stack used for the
long-context prefill measurements are not included.

Reported numbers were produced with the [EleutherAI lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
at 0-shot for the commonsense tasks and GSM8K, and with RULER for long context. A trained checkpoint
loads through `HybridModelWrapper` as shown above, so it can be wired into the harness directly.

## Citation

```bibtex
@article{fashi2026hylo,
  title={Long-Context Aware Upcycling: A New Frontier for Hybrid LLM Scaling},
  author={Ashrafi Fashi, Parsa and Saxena, Utkarsh and Rezagholizadeh, Mehdi and Jafari, Aref
          and Haridas, Akash and Yang, Mingyu and Bhatia, Vansh and Li, Guihong
          and Appia, Vikram and Barsoum, Emad},
  journal={arXiv preprint arXiv:2604.24715},
  year={2026}
}

@article{yang2025zebra,
  title={Zebra-Llama: Towards Extremely Efficient Hybrid Models},
  author={Yang, Mingyu and Rezagholizadeh, Mehdi and Li, Guihong and Appia, Vikram and Barsoum, Emad},
  journal={arXiv preprint arXiv:2505.17272},
  year={2025}
}

@article{li2025x_ecomla,
  title={{X-EcoMLA}: Upcycling Pre-Trained Attention into {MLA} for Efficient and Extreme {KV} Compression},
  author={Li, Guihong and Rezagholizadeh, Mehdi and Yang, Mingyu and Appia, Vikram and Barsoum, Emad},
  journal={arXiv preprint arXiv:2503.11132},
  year={2025}
}
```

## Acknowledgements

HyLo builds on [MambaInLlama](https://github.com/jxiw/MambaInLlama), the
[flash-linear-attention](https://github.com/fla-org/flash-linear-attention) Gated DeltaNet kernels,
[mamba](https://github.com/state-spaces/mamba), and the
[EleutherAI lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness). Thanks to
those authors for making this work possible.

## Contributing, security and contact

Contributions are welcome — see [CONTRIBUTING.md](../CONTRIBUTING.md) for the workflow and
[SECURITY.md](../SECURITY.md) for how to report a suspected vulnerability. Maintainers are listed in
[.github/CODEOWNERS](../.github/CODEOWNERS); please open an issue for questions about the code or the
released checkpoints.

## License

This repository is released under the Apache License 2.0 — see [LICENSE](../LICENSE). The released
checkpoints carry their own licence terms, stated on each model card; check them before use, and
note that some training datasets are non-commercial.

[paper]: https://arxiv.org/abs/2604.24715
[models]: https://huggingface.co/collections/amd/hylo-long-context-aware-upcycling-for-hybrid-llms-6aa40540e1398545571f476a
