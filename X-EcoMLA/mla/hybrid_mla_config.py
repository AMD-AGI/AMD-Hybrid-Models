from dataclasses import dataclass, field
from typing import List

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

DEEPSEEK_PRETRAINED_CONFIG_ARCHIVE_MAP = {}
class DeepseekV3Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`DeepseekV3Model`]. It is used to instantiate an DeepSeek
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the DeepSeek-V3.
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        vocab_size (`int`, *optional*, defaults to 129280):
            Vocabulary size of the Deep model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`DeepseekV3Model`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 11008):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer decoder.
        num_nextn_predict_layers (`int`, *optional*, defaults to 1):
            Number of nextn predict layers in the DeepSeekV3 Model.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer decoder.
        routed_scaling_factor (`float`, *optional*, defaults to 1.0):
            Scaling factor or routed experts.
        topk_method (`str`, *optional*, defaults to `gready`):
            Topk method used in routed gate.
        n_group (`int`, *optional*, defaults to None):
            Number of groups for routed experts.
        topk_group (`int`, *optional*, defaults to None):
            Number of selected groups for each token(for each token, ensuring the selected experts is only within `topk_group` groups).
        first_k_dense_replace (`int`, *optional*, defaults to 0):
            Number of dense layers in shallow layers(embed->dense->dense->...->dense->moe->moe...->lm_head).
                                                            \--k dense layers--/
        scoring_func (`str`, *optional*, defaults to 'softmax'):
            Method of computing expert weights.
        aux_loss_alpha (`float`, *optional*, defaults to 0.001):
            Auxiliary loss weight coefficient.
        seq_aux = (`bool`, *optional*, defaults to True):
            Whether to compute the auxiliary loss for each individual sample.
        llama_num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `llama_num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `llama_num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to
            `num_attention_heads`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 1):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 2):
            End of stream token id.
        pretraining_tp (`int`, *optional*, defaults to 1):
            Experimental feature. Tensor parallelism rank used during pretraining. Please refer to [this
            document](https://huggingface.co/docs/transformers/parallelism) to understand more about it. This value is
            necessary to ensure exact reproducibility of the pretraining results. Please refer to [this
            issue](https://github.com/pytorch/pytorch/issues/76232).
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. Currently supports two scaling
            strategies: linear and dynamic. Their scaling factor must be a float greater than 1. The expected format is
            `{"type": strategy name, "factor": scaling factor}`. When using this flag, don't update
            `max_position_embeddings` to the expected new maximum.
        attention_bias (`bool`, defaults to `False`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
    ```python
    >>> from transformers import DeepseekV3Model, DeepseekV3Config
    >>> # Initializing a Deepseek-V3 style configuration
    >>> configuration = DeepseekV3Config()
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "deepseek_v3"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=129280,
        hidden_size=7168,
        intermediate_size=18432,
        num_hidden_layers=61,
        num_nextn_predict_layers=1,
        num_attention_heads=128,
        llama_num_key_value_heads=128,
        mla_num_key_value_heads=16,
        ep_size = 1,
        routed_scaling_factor = 2.5,
        kv_lora_rank = 512,
        q_lora_rank = 1536,
        use_lora_layer_norm = False,
        use_fixed_rank_for_first_and_last_block = False,
        layer_rank_list = {},
        qk_rope_head_dim = 64,
        v_head_dim = 128,
        qk_nope_head_dim = 128,
        q_energy_ratio = 0.99,
        q_svd_norm = 2.0,
        kv_energy_ratio = None,
        kv_svd_norm = 2.0,
        qkv_rank_divisor = 8,
        topk_method = 'noaux_tc',
        n_group = 8,
        topk_group = 4,
        first_k_dense_replace = 3,
        norm_topk_prob = True,
        scoring_func = 'sigmoid',
        aux_loss_alpha = 0.001,
        seq_aux = True,
        hidden_act="silu",
        max_position_embeddings=4096,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=0,
        eos_token_id=1,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        rope_type="yarn",
        **kwargs,
    ):
        assert q_energy_ratio is None or 1.0 >= q_energy_ratio > 0.0, "q_energy_ratio must be (0, 1]"
        assert kv_energy_ratio is None or 1.0 >= kv_energy_ratio > 0.0, "q_energy_ratio must be (0, 1]"

        assert qkv_rank_divisor % 8 ==0, "qkv_rank_divisor must be divisible by 8"
        assert kv_lora_rank % qkv_rank_divisor ==0, "kv_lora_rank must be divisible by qkv_rank_divisor"
        assert q_lora_rank % qkv_rank_divisor ==0, "q_lora_rank must be divisible by qkv_rank_divisor"
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.num_attention_heads = num_attention_heads
        self.ep_size = ep_size
        self.routed_scaling_factor = routed_scaling_factor
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.use_lora_layer_norm = use_lora_layer_norm
        self.use_fixed_rank_for_first_and_last_block = use_fixed_rank_for_first_and_last_block
        self.layer_rank_list = layer_rank_list
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.q_energy_ratio = q_energy_ratio
        self.q_svd_norm = q_svd_norm
        self.kv_energy_ratio = kv_energy_ratio
        self.kv_svd_norm = kv_svd_norm
        self.qkv_rank_divisor = qkv_rank_divisor
        self.topk_method = topk_method
        self.n_group = n_group
        self.topk_group = topk_group
        self.first_k_dense_replace = first_k_dense_replace
        self.norm_topk_prob = norm_topk_prob
        self.scoring_func = scoring_func
        self.aux_loss_alpha = aux_loss_alpha
        self.seq_aux = seq_aux

        # for backward compatibility
        if mla_num_key_value_heads is None:
            mla_num_key_value_heads = num_attention_heads

        self.llama_num_key_value_heads = llama_num_key_value_heads
        self.mla_num_key_value_heads = mla_num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.rope_type = rope_type

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


@dataclass
class MLAConfig:
    d_model: int = 2560
    deepseek_cfg: DeepseekV3Config = DeepseekV3Config()
    rms_norm_eps: float = 1e-5
    vocab_size: int = None
    d_inner: int = None
    d_xb: int = 2560
    intermediate_size: int = 10240
    hidden_act: str = "silu"
    n_layer: int = 32
    attn_layers: List[int] = field(default_factory=list)