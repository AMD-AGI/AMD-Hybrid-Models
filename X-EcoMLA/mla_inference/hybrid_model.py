from typing import Optional, Tuple
from torch import Tensor
from transformers.activations import ACT2FN

from mla.hybrid_mla_config import MLAConfig
from mla_inference.hybrid_mla_layer import DeepseekV3RMSNorm, DeepseekV3FlashAttention2
from transformers.cache_utils import Cache, DynamicCache

import torch
import torch.nn as nn
import logging
logger = logging.getLogger(__name__)



class MLP(nn.Module):
    def __init__(self, config: MLAConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.d_model
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
        config: MLAConfig,
        layer_idx: int,
        device=None,
        dtype=None,
        residual_in_fp32=True,
    ):
        super(MLADecoderLayer, self).__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.layer_idx = layer_idx
        self.mla = DeepseekV3FlashAttention2(
            config.deepseek_cfg, layer_idx=layer_idx
        )
        self.mlp = MLP(config=config)
        self.input_layernorm = DeepseekV3RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.post_attention_layernorm = DeepseekV3RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.residual_in_fp32 = True

    def forward(
            self,
            hidden_states: torch.Tensor,
            inference_params=None,
            position_ids: Optional[torch.LongTensor] = None,
            *args,
            **kwargs,            
        ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.mla(
            hidden_states=hidden_states,
            inference_params=inference_params,
            position_ids=position_ids,
            **kwargs,
        )
        # hidden_states = self.mla(hidden_states, inference_params=inference_params)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states
    
    def _init_merge(self):
        self.mla._init_merge()

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mla.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs) 
    
    # Break the forward pass into three parts
    def forward_static_1(self, hidden_states: Tensor, inference_params=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        q_nope, q_pe, k_pe, cos, sin, compressed_kv = self.mla.forward_static_1(hidden_states, inference_params=inference_params)
        return q_nope, q_pe, k_pe, cos, sin, compressed_kv, residual
    
    def forward_dynamic(self, q_nope, q_pe, k_pe, cos, sin, compressed_kv, inference_params=None):
        return self.mla.forward_dynamic(q_nope, q_pe, k_pe, cos, sin, compressed_kv, inference_params=inference_params)

    def forward_static_2(self, x: Tensor, residual: Tensor):
        hidden_states = self.mla.forward_static_2(x)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states    
        return hidden_states
