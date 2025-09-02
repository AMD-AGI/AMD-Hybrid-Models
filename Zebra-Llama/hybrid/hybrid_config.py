from dataclasses import dataclass, field
from typing import List

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

@dataclass
class HybridConfig:
    
    # Common parameters
    hidden_size: int = 4096
    intermediate_size: int = 18432
    hidden_act: str = "silu"
    n_layer: int = 32
    mla_layers: List[int] = field(default_factory=list)
    rms_norm_eps: float = 1e-5

    # MLA parameters
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    kv_lora_rank: int = 128
    q_lora_rank: int = 1536
    use_lora_layer_norm: bool = False
    use_full_kv_head: bool = False
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    qk_nope_head_dim: int = 128
    qkv_rank_divisor: int = 8
    max_position_embeddings: int = 4096
    rope_theta: float = 10000.0
    rope_scaling: dict = None
    attention_bias: bool = False
    attention_dropout: float = 0.0
    rope_type: str = "yarn"

    # Mamba parameters
    d_model: int = 2560
    ssm_cfg: dict = field(default_factory=dict)
    d_inner: int = None
    d_xb: int = 2560

    def __post_init__(self):
        assert self.qkv_rank_divisor % 8 == 0, "qkv_rank_divisor must be divisible by 8"
        assert self.kv_lora_rank % self.qkv_rank_divisor == 0, "kv_lora_rank must be divisible by qkv_rank_divisor"
        assert self.q_lora_rank % self.qkv_rank_divisor == 0, "q_lora_rank must be divisible by qkv_rank_divisor"
        assert self.num_attention_heads % self.num_key_value_heads == 0, "num_attention_heads must be divisible by num_key_value_heads"


