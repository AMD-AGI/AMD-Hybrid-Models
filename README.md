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
git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git) # <-- UPDATE THIS URL
cd YOUR_REPO_NAME # <-- UPDATE THIS DIR NAME
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

## Acknowledgements

This work builds upon the foundations laid by the [MambaInLlama](https://github.com/jxiw/MambaInLlama) project. We thank the authors for their contribution to the community.

## Citation

If you find X-EcoMLA useful in your research or application, please cite our paper:

```bibtex
@article{li2025x_ecomla,
  title={{X-EcoMLA}: Upcycling Pre-Trained Attention into {MLA} for Efficient and Extreme {KV} Compression},
  author={Li, Guihong and Rezagholizadeh, Mehdi and Yang, Mingyu and Appia, Vikram and Barsoum, Emad},
  journal={arXiv preprint arXiv:2503.11132},
  year={2025}
}
```

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.
