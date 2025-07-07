# Zebra-Llama

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) [![arXiv](https://img.shields.io/badge/arXiv-2505.17272-b31b1b.svg)](https://arxiv.org/abs/2505.17272)

**Official repository for Zebra-Llama: Towards Extremely Efficient Hybrid Models**

Large Language Models (LLMs) often face significant memory bottlenecks due to the large Key-Value (KV) cache required during inference. In **X-EcoMLA**, we address this challenge by proposing a novel method to "upcycle" the attention mechanisms of pre-trained models into Multi-head Latent Attention (MLA), which substantially reduces the KV cache with minimal impact on model performance. In the followup work **Zebra-Llama**, we achieve further KV cache compression and inference efficiency by proposing a hybrid model that mixs MLA and Mamba. 

This repository provides the code necessary to reproduce the results, train new Zebra-Llama models, and evaluate their performance.

## Features

* **Extreme KV Cache Compression:** Leverages MLA and Mamba to significantly reduce the memory footprint of the KV cache.
* **Efficient Upcycling:** Modifies pre-trained attention layers rather than training from scratch.
* **Two-Stage Training Pipeline:** Employs end-to-end distillation followed by Direct Preference Optimization (DPO) for optimal performance and alignment.
* **Hardware Support:** Verified training procedures for both AMD Instinct™ MI300 and MI325 GPUs and NVIDIA H100/H200 GPUs.
* **Example Implementations:** Provides configurations and scripts for Llama-family models (1B, 3B, and 8B parameters).

## Table of Contents

* [Installation](#installation)
* [Training](#training)
    * [1. Intermediate Layer Distillation](#1-intermediate-layer-distillation-ild)
    * [2: End-to-End SFT Distillation](#2-end-to-end-sft-distillation)
    * [3: Instruction Tuning with DPO](#3-instruction-tuning-with-dpo)
    * [Configuration](#configuration)
* [Evaluation](#evaluation)
* [Acknowledgements](#acknowledgements)
* [Citation](#citation)
* [License](#license)
<!-- * [Contributing](#contributing) -->


## Installation
We strongly recommend using Docker to ensure a consistent and reproducible environment.

**1. Clone the Repository:**
```bash
git clone https://github.com/AMD-AIG-AIMA/AMD-Hybrid-Models.git 
cd AMD-Hybrid-Models/Zebra-Llama
```

**2. Build the docker:**

Choose the instructions based on your GPU hardware:

  * **For AMD Instinct™ MI300 and MI325 GPUs:**

      * We verified training using the `rocm/pytorch-training:v25.4` image.

    <!-- end list -->

    ```bash
    # Launch the Docker container with ROCm device access
    docker run -it \
      --device /dev/dri --device /dev/kfd \
      --device=/dev/infiniband --network host --ipc host \
      --group-add video --cap-add SYS_PTRACE \
      --security-opt seccomp=unconfined --privileged \
      -v $HOME:$HOME --shm-size 64G --name mla_training \
      rocm/pytorch-training:v25.4
    # Note: Adjust --shm-size and permissions based on your system configuration.

    # Inside the container, navigate to the cloned repo and install dependencies
    # Use FLASH_ATTN=1 for potential optimizations on AMD hardware
    cd /path/to/your/cloned/repo # e.g., cd /home/user/AMD-Hybrid-Models
    bash install.sh FLASH_ATTN=1 MAMBA=1
    ```


  * **For NVIDIA H100/H200 GPUs:**

      * We verified training using the `nvcr.io/nvidia/pytorch:24.10-py3` image.

    ```bash
    # Launch the Docker container
    docker run --gpus all -it -v $HOME:$HOME --shm-size 64G --rm nvcr.io/nvidia/pytorch:24.10-py3
    # Note: Adjust --shm-size based on your system capabilities.

    # Inside the container, navigate to the cloned repo and install dependencies
    cd /path/to/your/cloned/repo # e.g., cd /home/user/Zebra-Llama
    bash install.sh MAMBA=1
    ```

**Note:** The `install.sh` script handles the installation of required Python packages and dependencies within the containerized environment.

## Pre-trained Checkpoints

We released our model checkpoints in the [huggingface](https://huggingface.co/collections/amd/amd-hybrid-models-67be591b09a4524abf65bcee). 

**Chat with the Model**

```bash
import torch
from transformers import AutoTokenizer
from hybrid.hybrid_wrapper import HybridModelWrapper

checkpoint_path = "amd/Zebra-Llama-1B-4MLA-12Mamba-DPO"
model = HybridModelWrapper.from_pretrained(checkpoint_path, torch_dtype=torch.bfloat16, absorb=True).cuda()
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
model.eval()

# Format the prompt using the chat template
prompt = [{"role": "user", "content": "What are the benefits of hybrid language models?"}]
input_ids = tokenizer.apply_chat_template(
    prompt,
    add_generation_prompt=True,
    return_tensors='pt'
).cuda()

# Generate a response
tokens = model.generate(
    inputs.to(model.device),
    max_new_tokens=256,
    temperature=0.7,
    do_sample=True
)

print(tokenizer.decode(tokens[0], skip_special_tokens=False))
```


## Training

### 1. Intermediate Layer Distillation (ILD)

  * **Goal:** Align the intermediate hidden states for better weight initialization.
  * **Method:** Minimize the Mean Square Error (MSE) between the outputs of the student and teacher layers given the same inputs.
  * **Framework:** Uses `accelerate` (configured via `configs/fsdp_M2_ILD.yaml` and `configs/fsdp_MLA_ILD.yaml`) for FSDP training.

**Example Commands:**

  * **Llama3.2-1B MLA blocks:**
    ```bash
    ACCELERATE_LOG_LEVEL=info accelerate launch --config_file configs/fsdp_MLA_ILD.yaml train_hybrid/train_distill.py configs/llama3.2_1B/zebra_MLA_ILD.yaml # <-- Update path if needed
    ```
  * **Llama3.2-1B Mamba blocks**
    ```bash
    ACCELERATE_LOG_LEVEL=info accelerate launch --config_file configs/fsdp_M2_ILD.yaml train_hybrid/train_distill.py configs/llama3.2_1B/zebra_M2_ILD.yaml # <-- Update path if needed
    ```

### 2. End-to-End SFT Distillation

  * **Goal:** Transfer knowledge from a larger, pre-trained teacher model to the smaller student model (with the MLA/Mamba architecture).
  * **Method:** Minimize the Kullback–Leibler (KL) divergence loss between the output distributions of the student and teacher models. We generally observe better results when using a larger teacher model.
  * **Framework:** Uses `accelerate` and `fsdp` (configured via `configs/fsdp.yaml`) for distributed training.

**Example Commands:**

  * **Zebra-Llama-1B-4MLA-12Mamba**
    ```bash
    ACCELERATE_LOG_LEVEL=info accelerate launch --config_file configs/fsdp.yaml train_hybrid/train_distill.py configs/llama3.2_1B/zebra_4MLA12M2_8bt_SFT.yaml # <-- Update path if needed
    ```
**Note:** After training, need to get the `model.safetensors` using `accelerate merge-weights` for the following DPO training.

### 3: Instruction Tuning with DPO

  * **Goal:** Further align the distilled model with desired behaviors and instruction-following capabilities.
  * **Method:** Apply Direct Preference Optimization (DPO) using a preference dataset.
  * **Framework:** Also uses `accelerate` and `fsdp`.

**Example Commands:**
```bash
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file configs/fsdp.yaml train_hybrid/train_dpo.py configs/dpo.yaml # <-- Update path if needed
```

### Configuration

  * Training hyperparameters, model paths, dataset details, model configurations (e.g., KV rank, quantization bits), and FSDP settings are controlled by `.yaml` configuration files (e.g., `configs/llama3.2_1B/zebra_4MLA12M2_8bt_SFT.yaml`, `configs/fsdp.yaml`).
  * Please inspect these files and modify them according to your specific needs (e.g., dataset paths, teacher/student model identifiers, compute resources).

## Evaluation

We provide several checkpoints of Zebra-Llama [here](https://huggingface.co/collections/amd/amd-hybrid-models-67be591b09a4524abf65bcee). The test these checkpoints or your trained hybrid model, please run:
```bash
# Run lm-eval
python benchmark/llm_eval/lm_harness_eval.py \
 --model hybrid \
 --model_args pretrained=amd/Zebra-Llama-8B-16MLA-16Mamba-DPO \
 --tasks mmlu,hellaswag,piqa,arc_easy,arc_challenge,winogrande,openbookqa,race \
 --num_fewshot 0 --device cuda --batch_size 16
```

Besides, we provide a script to perform batched evaluation for the trained hybrid models

1.  **Update Checkpoint Path:** Modify the `eval.sh` script to point to the directory containing your final model checkpoint (saved after the DPO stage).
    ```bash
    # Example modification within eval.sh
    ckpts=(
      "PATH_TO_YOUR_CKPT1"
      "PATH_TO_YOUR_CKPT2"
    )
    # ... rest of the script
    ```
2.  **Run Evaluation Script:** Execute the script from the root of the repository.
    ```bash
    bash eval.sh
    ```
    This script will typically run the model on standard benchmark datasets and report relevant metrics (e.g., perplexity, task-specific accuracy, memory usage). Check the script for details on the specific evaluation tasks performed.

## Results

### Extreme KV Cache Compression with Larger Teacher (X-EcoMLA/Zebra-Llama)
Here we study the impact of KV-cache compression and teacher model size on performance. Reducing the KV-cache size lowers accuracy, but larger teacher models help recover performance. 

**First)** Target model: **Llama3.2-1B-Inst**,  Teacher model: **Llama3.1-8B-Inst**
| Model & Setting  | KV Size | Param | Tokens | ARC | ARE | HS | MMLU | OBQA | PIQA | RA | WG | Avg. |
|-----------------|-------|-------|--------|----:|----:|---:|-----:|-----:|-----:|---:|---:|----:|
| **Llama3.2-1B-Inst** | 100% |1.24 B | –  | 37.97 | 63.30 | 60.65 | 46.05 | 34.80 | 74.32 |  38.18 | 59.67 | 51.87 |
|  X-EcoMLA (r<sub>kv</sub>=64)  | 9.37% | 1.23 B | 7 B | 40.02 | 67.17 | 58.40 | 38.53 | 37.80 | 73.83 | 39.43 | 60.93 | 52.01 |
|  Zebra-Llama-8MLA-8M2 (r<sub>kv</sub>=128) ([ckpt](https://huggingface.co/amd/Zebra-Llama-1B-8MLA-8Mamba-DPO))| 7.81% | 1.27 B | 7 B | 42.49 | 67.38 | 60.54 | 38.94 | 41.6 | 72.91 |  38.37 | 61.25 | 52.94 |
|  Zebra-Llama-4MLA-12M2 (r<sub>kv</sub>=128) ([ckpt](https://huggingface.co/amd/Zebra-Llama-1B-4MLA-12Mamba-DPO))| **3.91%** | 1.28 B | 7 B | 42.32 | 66.96 | 58.93 | 37.91 | 40.6 | 72.96 | 37.7 | 58.88 | 52.03 |

**Second)** Target model: **Llama3.2-3B-Inst**,  Teacher model: **Llama3.1-8B-Inst**
| Model & Setting  | KV Size | Param | Tokens | ARC | ARE | HS | MMLU | OBQA | PIQA | RA | WG | Avg. |
|-----------------|-------|-------|--------|----:|----:|---:|-----:|-----:|-----:|---:|---:|----:|
| **Llama3.2-3B-Inst** | 100% |3.21 B | –  | 46.08 | 67.93 | 70.38 | 60.34 | 36.4 | 75.79 |  40.86 | 67.25 | 58.13 |
|  X-EcoMLA (r<sub>kv</sub>=128)  | 9.37% | 3.21 B | 7 B | 52.05 | 75.38 | 70.95 | 53.2 | 40.8 | 77.09 | 44.69 | 66.85 | 60.13 |
|  Zebra-Llama-14MLA-14M2 (r<sub>kv</sub>=128) ([ckpt](https://huggingface.co/amd/Zebra-Llama-3B-14MLA-14Mamba-DPO)) | 4.69% | 3.27 B | 9 B | 51.28 | 76.14 | 72.57 | 52.1 | 42.4 | 77.53 |  45.93 | 67.56 | 60.69 |
|  Zebra-Llama-6MLA-22M2 (r<sub>kv</sub>=128) ([ckpt](https://huggingface.co/amd/Zebra-Llama-3B-6MLA-22Mamba-DPO)) | **2.01%** | 3.39 B | 9 B | 50.77 | 76.09 | 71.46 | 50.06 | 43.4 | 77.26 | 42.49 | 66.46 | 59.75 |

**Third)** Target model: **Llama3.1-8B-Inst**,  Teacher model: **Llama3.1-8B-Inst**
| Model & Setting  | KV Size | Param | Tokens | ARC | ARE | HS | MMLU | OBQA | PIQA | RA | WG | Avg. |
|-----------------|-------|-------|--------|----:|----:|---:|-----:|-----:|-----:|---:|---:|----:|
| **Llama3.1-8B-Inst** | 100% |8.03 B | –  | 54.86 | 79.55 | 79.23 | 68.13 | 43 | 80.9 |  44.69 | 73.88 | 65.53 |
|  X-EcoMLA (r<sub>kv</sub>=128)  | 9.37% | 8.03 B | 7 B | 56.57 | 79.04 | 77.38 | 58.6 | 42.8 | 79.6 | 48.33 | 70.96 | 64.16 |
|  Zebra-Llama-16MLA-16M2 (r<sub>kv</sub>=160) ([ckpt](https://huggingface.co/amd/Zebra-Llama-8B-16MLA-16Mamba-DPO)) | 5.47% | 8.19 B | 11 B | 58.62 | 78.37 | 79.27 | 58.17 | 43.4 | 80.03 |  49.28 | 72.61 | 64.97 |
|  Zebra-Llama-8MLA-24M2 (r<sub>kv</sub>=128) ([ckpt](https://huggingface.co/amd/Zebra-Llama-8B-8MLA-24Mamba-DPO)) | **2.73%** | 8.38 B | 11 B | 58.87 | 79.17 | 78.6 | 54.6 | 43.6 | 79.43 | 46.22 | 72.45 | 64.12 |

## Acknowledgements

This work builds upon the foundations laid by the [MambaInLlama](https://github.com/jxiw/MambaInLlama) project. We thank the authors for their contribution to the community.

## Citation

If you find Zebra-Llama useful in your research or application, please cite our paper:

```bibtex
@article{li2025x_ecomla,
  title={{X-EcoMLA}: Upcycling Pre-Trained Attention into {MLA} for Efficient and Extreme {KV} Compression},
  author={Li, Guihong and Rezagholizadeh, Mehdi and Yang, Mingyu and Appia, Vikram and Barsoum, Emad},
  journal={arXiv preprint arXiv:2503.11132},
  year={2025},
  url={https://arxiv.org/abs/2503.11132}
}

@article{yang2025zebra,
  title={Zebra-Llama: Towards Extremely Efficient Hybrid Models},
  author={Yang, Mingyu and Rezagholizadeh, Mehdi and Li, Guihong and Appia, Vikram and Barsoum, Emad},
  journal={arXiv preprint arXiv:2505.17272},
  year={2025}
}
```

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.
