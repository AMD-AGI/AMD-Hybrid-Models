from typing import Optional, Tuple
from torch import Tensor
from transformers.activations import ACT2FN

from hybrid.hybrid_config import HybridConfig
from hybrid.mla.hybrid_mla_layer import DeepseekV3RMSNorm, DeepseekV3FlashAttention2, DeepseekV3Attention
from transformers.cache_utils import Cache, DynamicCache
from hybrid.hybrid_modeling import HybridDynamicCache

import torch
import torch.nn as nn
import logging
logger = logging.getLogger(__name__)



class MLP(nn.Module):
    def __init__(self, config: HybridConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class MLADecoderLayer(nn.Module):
    def __init__(
        self,
        config: HybridConfig,
        layer_idx: int,
        device=None,
        dtype=None,
        residual_in_fp32=True,
    ):
        super(MLADecoderLayer, self).__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.layer_idx = layer_idx
        self.mla = DeepseekV3FlashAttention2(
            config, layer_idx=layer_idx
        )
        self.mlp = MLP(config=config)
        self.input_layernorm = DeepseekV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = DeepseekV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.residual_in_fp32 = True

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[HybridDynamicCache] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            *args,
            **kwargs,            
        ):

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights, present_key_value = self.mla(
            hidden_states=hidden_states,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
