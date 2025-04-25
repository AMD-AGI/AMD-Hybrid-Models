# X-EcoMLA: Efficient and Extreme KV Compression via Upcycled Attention

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) [![arXiv](https://img.shields.io/badge/arXiv-2503.11132-b31b1b.svg)](https://arxiv.org/abs/2503.11132)

**Official repository for X-EcoMLA: Upcycling Pre-Trained Attention into Multi-Layer Attention (MLA) for Efficient and Extreme Key-Value (KV) Cache Compression in Large Language Models.**

Large Language Models (LLMs) often face significant memory bottlenecks due to the large Key-Value (KV) cache required during inference. X-EcoMLA addresses this challenge by proposing a novel method to "upcycle" the existing attention mechanisms of pre-trained models into Multi-Layer Attention (MLA) structures. This approach achieves substantial KV cache compression with minimal impact on model performance, enabling more efficient deployment and inference.

This repository provides the code necessary to reproduce the results, train new X-EcoMLA models, and evaluate their performance.

## Features

* **Extreme KV Cache Compression:** Leverages MLA to significantly reduce the memory footprint of the KV cache.
* **Efficient Upcycling:** Modifies pre-trained attention layers rather than training from scratch.
* **Two-Stage Training Pipeline:** Employs end-to-end distillation followed by Direct Preference Optimization (DPO) for optimal performance and alignment.
* **Hardware Support:** Verified training procedures for both NVIDIA (H100/H200) and AMD (MI300/MI325) GPUs.
* **Example Implementations:** Provides configurations and scripts for Llama3.2 models (1B and 3B parameters).

## Table of Contents

* [Installation](#installation)
* [Training](#training)
    * [Stage 1: End-to-End Distillation](#stage-1-end-to-end-distillation)
    * [Stage 2: Instruction Tuning with DPO](#stage-2-instruction-tuning-with-dpo)
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
git clone https://github.com/AMD-AIG-AIMA/X-EcoMLA.git # <-- UPDATE THIS URL
cd X-EcoMLA # <-- UPDATE THIS DIR NAME
```

**2. Build the docker:**

Choose the instructions based on your GPU hardware:

  * **For NVIDIA H100/H200 GPUs:**

      * We verified training using the `nvcr.io/nvidia/pytorch:25.01-py3` image.

    ```bash
    # Launch the Docker container
    docker run --gpus all -it -v $HOME:$HOME --shm-size 64G --rm nvcr.io/nvidia/pytorch:25.01-py3
    # Note: Adjust --shm-size based on your system capabilities.

    # Inside the container, navigate to the cloned repo and install dependencies
    cd /path/to/your/cloned/repo # e.g., cd /home/user/X-EcoMLA
    bash install.sh
    ```

  * **For AMD MI300/MI325 GPUs:**

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
    cd /path/to/your/cloned/repo # e.g., cd /home/user/X-EcoMLA
    bash install.sh FLASH_ATTN=1
    ```

**Note:** The `install.sh` script handles the installation of required Python packages and dependencies within the containerized environment.

## Training

X-EcoMLA utilizes a two-stage training strategy for optimal results:

### Stage 1: End-to-End Distillation

  * **Goal:** Transfer knowledge from a larger, pre-trained teacher model to the smaller student model (with the MLA architecture).
  * **Method:** Minimize the Kullback–Leibler (KL) divergence loss between the output distributions of the student and teacher models. We generally observe better results when using a larger teacher model.
  * **Framework:** Uses `accelerate` and `deepspeed` (configured via `deepspeed_zero3.yaml`) for distributed training.

**Example Commands:**

  * **Llama3.2-1B Student Model:**
    ```bash
    ACCELERATE_LOG_LEVEL=info accelerate launch --config_file deepspeed_zero3.yaml train_mla/train_distill.py llama3.2_1B/mla_kv_rank_64_8bt.yaml
    ```
  * **Llama3.2-3B Student Model:**
    ```bash
    ACCELERATE_LOG_LEVEL=info accelerate launch --config_file deepspeed_zero3.yaml train_mla/train_distill.py llama3.2_3B/mla_kv_rank_96_8bt.yaml
    ```

### Stage 2: Instruction Tuning with DPO

  * **Goal:** Further align the distilled model with desired behaviors and instruction-following capabilities.
  * **Method:** Apply Direct Preference Optimization (DPO) using a preference dataset.
  * **Framework:** Also uses `accelerate` and `deepspeed`.

**Example Commands:**

  * **Llama3.2-1B Model (after Stage 1):**
    ```bash
    ACCELERATE_LOG_LEVEL=info accelerate launch --config_file deepspeed_zero3.yaml train_mla/train_dpo.py llama3.2_1B/dpo.yaml
    ```
  * **Llama3.2-3B Model (after Stage 1):**
    ```bash
    ACCELERATE_LOG_LEVEL=info accelerate launch --config_file deepspeed_zero3.yaml train_mla/train_dpo.py llama3.2_3B/dpo.yaml # <-- Update path if needed
    ```

### Configuration

  * Training hyperparameters, model paths, dataset details, MLA configurations (e.g., KV rank, quantization bits), and DeepSpeed settings are controlled by `.yaml` configuration files (e.g., `llama3.2_1B/mla_kv_rank_64_8bt.yaml`, `deepspeed_zero3.yaml`).
  * Please inspect these files and modify them according to your specific needs (e.g., dataset paths, teacher/student model identifiers, compute resources).

## Evaluation

To evaluate the performance of your trained X-EcoMLA model:

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

### Self-distillation Evaluation
The table below reports zero-shot performance when X-EcoMLA learns from its own target model as the teacher (_self-distillation_).
We evaluate two SVD-based initialization strategies—fixed rank and dynamic rank—across nine LM-Harness tasks: ARC-Challenge (ARC), ARC-Easy (ARE), HellaSwag (HS), MMLU, OpenBookQA (OBQA), PIQA, PubMedQA (PBMD), RACE (RA), and WinoGrande (WG). 

| Model & Setting            | Init. Method                            | KV-Size | ARC  | ARE  | HS   | MMLU | OBQA | PIQA | PBMD | RA   | WG   | Avg. |
|----------------------------|-----------------------------------------|--------:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|
| **Llama3.2-1B-Inst**       | Target Model                            | 100 %   | 37.97| 63.30| 60.65| 46.05| 34.80| 74.32| 60.00| 38.18| 59.67| **52.77** |
| X-EcoMLA               | Fixed (r<sub>kv</sub>=512)              | 53 %  | 40.19| 63.93| 60.67| 42.31| 37.60| 75.03| 59.20| 40.86| 61.01| 53.42 |
| X-EcoMLA              | Dynamic (δ<sub>kv</sub>=0.95)           | 55 %  | 40.36| 64.31| 60.88| 42.54| 36.80| 74.16| 61.40| 40.77| 60.69| 53.54 |
| **Llama3.2-3B-Inst**       | Target Model                            | 100 %   | 46.08| 67.93| 70.38| 60.34| 36.40| 75.79| 69.60| 40.86| 67.25| **59.40** |
| X-EcoMLA               | Fixed (r<sub>kv</sub>=816)              | 43 %    | 48.38| 70.37| 72.41| 57.51| 38.20| 76.28| 66.80| 46.41| 68.11| 60.50 |
| X-EcoMLA               | Dynamic (δ<sub>kv</sub>=0.95)           | 43 %    | 48.55| 70.12| 72.25| 57.70| 39.60| 75.84| 68.40| 46.12| 66.14| 60.52 |

### Extreme KV Cache Compression with Larger Teacher 
Here we study the impact of KV-cache compression and teacher model size on performance. Reducing the KV-cache size lowers accuracy, but larger teacher models help recover performance. We trained Llama3.2-1B-Inst on three different ranks using three different teachers.

**First)** **KV Size 15.6 %** (r<sub>kv</sub>=128, r<sub>q</sub>=1344, d<sub>qk</sub>=32) 
| Model & Setting | Teacher | Param | Tokens | ARC | ARE | HS | MMLU | OBQA | PIQA | PBMD | RA | WG | Avg. |
|-----------------|---------|-------|--------|----:|----:|---:|-----:|-----:|-----:|-----:|---:|---:|----:|
| **Llama3.2-1B-Inst** | – | 1.24 B | –  | 37.97 | 63.30 | 60.65 | 46.05 | 34.80 | 74.32 | 60.00 | 38.18 | 59.67 | 52.77 |
|  X-EcoMLA  | Llama3.2-1B-Inst | 1.23 B | 7 B | 40.10 | 62.88 | 58.17 | 39.70 | 37.80 | 73.50 | 56.60 | 39.33 | 59.67 | 51.97 |
|  X-EcoMLA  | Llama3.2-3B-Inst | 1.23 B | 7 B | 39.33 | 64.86 | 58.92 | 41.86 | 37.40 | 73.83 | 58.80 | 39.71 | 59.59 | 52.70 |
|  X-EcoMLA  | Llama3.1-8B-Inst | 1.23 B | 7 B | 42.49 | 67.13 | 60.58 | 42.51 | 36.60 | 73.99 | 59.40 | 40.38 | 59.43 | **53.61** |

**Second)** **KV Size 9.4 %** (r<sub>kv</sub>=64, r<sub>q</sub>=1424, d<sub>qk</sub>=32) 
| Model & Setting | Teacher | Param | Tokens | ARC | ARE | HS | MMLU | OBQA | PIQA | PBMD | RA | WG | Avg. |
|-----------------|---------|-------|--------|----:|----:|---:|-----:|-----:|-----:|-----:|---:|---:|----:|
| **Llama3.2-1B-Inst** | – | 1.24 B | –  | 37.97 | 63.30 | 60.65 | 46.05 | 34.80 | 74.32 | 60.00 | 38.18 | 59.67 | 52.77 |
|  X-EcoMLA  | Llama3.2-1B-Inst | 1.23 B | 7 B | 39.16 | 62.63 | 56.04 | 34.90 | 36.40 | 72.85 | 56.40 | 37.70 | 58.33 | 50.49 |
|  X-EcoMLA  | Llama3.2-3B-Inst | 1.23 B | 7 B | 37.97 | 63.55 | 56.95 | 37.54 | 35.40 | 72.74 | 57.00 | 38.66 | 59.27 | 51.01 |
|  X-EcoMLA  | Llama3.1-8B-Inst | 1.23 B | 7 B | 40.02 | 67.17 | 58.40 | 38.53 | 37.80 | 73.83 | 58.00 | 39.43 | 60.93 | **52.68** |

**Third)** **KV Size 7.8 %** (r<sub>kv</sub>=48, r<sub>q</sub>=1440, d<sub>qk</sub>=32)
| Model & Setting | Teacher | Param | Tokens | ARC | ARE | HS | MMLU | OBQA | PIQA | PBMD | RA | WG | Avg. |
|-----------------|---------|-------|--------|----:|----:|---:|-----:|-----:|-----:|-----:|---:|---:|----:|
| **Llama3.2-1B-Inst** | – | 1.24 B | –  | 37.97 | 63.30 | 60.65 | 46.05 | 34.80 | 74.32 | 60.00 | 38.18 | 59.67 | 52.77 |
|  X-EcoMLA  | Llama3.2-1B-Inst | 1.23 B | 7 B | 38.48 | 61.66 | 55.32 | 30.62 | 35.20 | 72.36 | 56.60 | 37.99 | 59.43 | 49.74 |
|  X-EcoMLA  | Llama3.2-3B-Inst | 1.23 B | 7 B | 36.18 | 62.21 | 55.82 | 36.41 | 35.60 | 72.03 | 57.00 | 38.09 | 60.06 | 50.38 |
|  X-EcoMLA  | Llama3.1-8B-Inst | 1.23 B | 7 B | 37.71 | 65.32 | 57.32 | 36.27 | 36.80 | 72.96 | 58.20 | 38.76 | 58.80 | **51.35** |

<p align="center">
  <img src="patch/download.png" alt="Average accuracy vs. KV-cache compression for X-EcoMLA" width="600">
  <br>
  <em>Figure&nbsp;1 – Average accuracy of X-EcoMLA (trained on ≈7 B tokens) as a function of KV-cache compression ratio, using 1 B, 3 B, and 8 B teacher models. The dashed line marks the base-model score (52.77).</em>
</p>

## Acknowledgements

This work builds upon the foundations laid by the [MambaInLlama](https://github.com/jxiw/MambaInLlama) project. We thank the authors for their contribution to the community.

## Citation

If you find X-EcoMLA useful in your research or application, please cite our paper:

```bibtex
@article{li2025x_ecomla,
  title={{X-EcoMLA}: Upcycling Pre-Trained Attention into {MLA} for Efficient and Extreme {KV} Compression},
  author={Li, Guihong and Rezagholizadeh, Mehdi and Yang, Mingyu and Appia, Vikram and Barsoum, Emad},
  journal={arXiv preprint arXiv:2503.11132},
  year={2025},
  url={https://arxiv.org/abs/2503.11132}
}
```

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.
