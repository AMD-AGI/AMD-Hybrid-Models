from typing import Optional, Tuple
from torch import Tensor
from transformers.activations import ACT2FN

from hybrid_inference.mla.hybrid_mla_layer import DeepseekV3RMSNorm, DeepseekV3FlashAttention2
from transformers.cache_utils import Cache, DynamicCache

import torch
import torch.nn as nn
import logging
logger = logging.getLogger(__name__)



class MLP(nn.Module):
    def __init__(self, config):
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
        config,
        layer_idx: int,
        device=None,
        absorb=False,
        dtype=None,
        residual_in_fp32=True,
    ):
        super(MLADecoderLayer, self).__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.layer_idx = layer_idx
        self.mla = DeepseekV3FlashAttention2(
            config, layer_idx=layer_idx, absorb=absorb
        )
        self.mlp = MLP(config=config)
        self.input_layernorm = DeepseekV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = DeepseekV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.residual_in_fp32 = True

    def forward(
            self,
            hidden_states: torch.Tensor,
            inference_params = None, 
            *args,
            **kwargs,            
        ):

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.mla(
            hidden_states=hidden_states,
            inference_params=inference_params,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states        
        
        outputs = (hidden_states,)
        
        return outputs
    
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mla.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs) 