# AMD-Hybrid-Models
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) [![arXiv](https://img.shields.io/badge/arXiv-2604.24715-b31b1b.svg)](https://arxiv.org/abs/2604.24715) [![arXiv](https://img.shields.io/badge/arXiv-2505.17272-b31b1b.svg)](https://arxiv.org/abs/2505.17272) [![arXiv](https://img.shields.io/badge/arXiv-2503.11132-b31b1b.svg)](https://arxiv.org/abs/2503.11132)

## 🔍 Overview: Efficient Hybrid Language Models on AMD GPUs  
**Official Repository for _X-EcoMLA_, _Zebra-Llama_ and _HyLo_**

Welcome! This repo hosts three complementary projects that focus on memory-efficient and high-performance large language models (LLMs). 
Large Language Models (LLMs) often face major memory bottlenecks due to large key-value (KV) caches during inference. This repository introduces three solutions:

| Folder           | Description |
|------------------|-------------|
| [`X-EcoMLA/`](X-EcoMLA)     | Implements **X-EcoMLA**: a method for upcycling attention into Multi-head Latent Attention (MLA) for extreme KV cache compression. |
| [`Zebra-Llama/`](Zebra-Llama)   | Implements **Zebra-Llama**: a family of hybrid MLA + Mamba2 models with minimal retraining and maximum efficiency. |
| [`HyLo/`](HyLo)          | Implements **HyLo**: long-context aware upcycling into hybrid MLA + Gated DeltaNet / Mamba-2 models, extending usable context by up to 32x while cutting KV cache by over 90%. |


## Citation
If you find this repository useful in your research or application, please cite our paper:

```bibtex
@article{fashi2026hylo,
  title={Long-Context Aware Upcycling: A New Frontier for Hybrid LLM Scaling},
  author={Ashrafi Fashi, Parsa and Saxena, Utkarsh and Rezagholizadeh, Mehdi and Jafari, Aref
          and Haridas, Akash and Yang, Mingyu and Bhatia, Vansh and Li, Guihong
          and Appia, Vikram and Barsoum, Emad},
  journal={arXiv preprint arXiv:2604.24715},
  year={2026}
}

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


 

## 🤝 Contributing
We welcome contributions! Please open an issue to discuss questions and major changes. 

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
