# AMD-Hybrid-Models

## 🔍 Overview: Efficient Hybrid Language Models on AMD GPUs  
**Official Repository for _X-EcoMLA_ and _Zebra-Llama_**

Welcome! This repo hosts two complementary projects that focus on memory-efficient and high-performance large language models (LLMs). 
Large Language Models (LLMs) often face major memory bottlenecks due to large key-value (KV) caches during inference. This repository introduces two solutions:

| Folder           | Description |
|------------------|-------------|
| `x-eco-mla/`     | Implements **X-EcoMLA**: a method for upcycling attention into Multi-head Latent Attention (MLA) for extreme KV cache compression. |
| `zebra-llama/`   | Implements **Zebra-Llama**: a family of hybrid MLA + Mamba2 models with minimal retraining and maximum efficiency. |


---

## 🧪 Quick Start

```bash
git clone 
cd efficient-hybrids
conda env create -f env.yml
conda activate hybrids


## Repository Structure
.
├── env.yml              # Conda env for PyTorch, ROCm, and Hugging Face
├── x-eco-mla/           # Codebase for X-EcoMLA
│   ├── README.md
│   ├── scripts/
│   └── configs/
└── zebra-llama/         # Codebase for Zebra-Llama
    ├── README.md
    ├── scripts/
    └── configs/
## Citation

@inproceedings{x-eco-mla,
  title     = {X-EcoMLA: Upcycling Pre-Trained Attention into Multi-Layer Attention
               for Extreme KV Cache Compression},
  author    = {...},
  year      = 2025
}

@inproceedings{zebra-llama,
  title     = {Zebra-Llama: Practical Hybrid Models with MLA and Mamba2},
  author    = {...},
  year      = 2025
}

## 🤝 Contributing
We welcome contributions! Please open an issue to discuss questions and major changes. 
