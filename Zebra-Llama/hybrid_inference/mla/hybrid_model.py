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
        self.config=config
        self.device=device
        self.dtype=dtype
    
    def _init_merge(self):
        self.mla._init_merge()

    def forward(
            self,
            hidden_states: torch.Tensor,
            inference_params = None, 
            position_ids=None,
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
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states        
        
        # outputs = (hidden_states,)
        
        return hidden_states
    
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mla.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs) 
    
    def forward_static_1(self, hidden_states: Tensor, inference_params=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # The breaked MLA module
        q_nope, q_pe, k_pe, cos, sin, compressed_kv = self.mla.forward_static_1(hidden_states, inference_params=inference_params)
        return (q_nope, q_pe, k_pe, cos, sin, compressed_kv, residual)

    def forward_static_2(self, x: Tensor, residual: Tensor):
        hidden_states = self.mla.forward_static_2(x)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states    
        return hidden_states

    def forward_graph(self, hidden_states: Tensor, inference_params=None, position_ids=None):
        
        self.residual.copy_(hidden_states)
        self.hidden_states.copy_(hidden_states)
        # q_nope, q_pe, compressed_kv_comb, _ = self.forward_static_1(hidden_states, inference_params=inference_params)
        self.graph.replay()
        q_nope, q_pe, k_pe, cos, sin, compressed_kv, _ = self.outputs

        x = self.mla.forward_dynamic(q_nope, q_pe, k_pe, cos, sin, compressed_kv, inference_params=inference_params, position_ids=position_ids)
        # return self.forward_static_2(x, self.residual)    

        self.x.copy_(x)
        self.graph2.replay()
        return self.outputs_2

        # ref = self.forward(hidden_states, inference_params=inference_params)
        # return self.forward(hidden_states, inference_params=inference_params)

    def capture_graph(self, inference_params):
        # conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
        self.hidden_states = torch.randn((inference_params.max_batch_size, 1, self.config.hidden_size), device=self.device, dtype=self.dtype)
        self.x = torch.randn((inference_params.max_batch_size, self.config.num_attention_heads, 1, self.config.kv_lora_rank), device=self.device, dtype=self.dtype)
        self.residual = torch.randn((inference_params.max_batch_size, 1, self.config.hidden_size), device=self.device, dtype=self.dtype)
        # self.conv_state = torch.randn_like(conv_state)
        # self.ssm_state = torch.randn_like(ssm_state)

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                self.outputs = self.forward_static_1(
                    self.hidden_states,
                    inference_params=inference_params
                    # self.conv_state,
                    # self.ssm_state,
                )
        s.synchronize()
        # This might be needed for correctness if we run with NCCL_GRAPH_MIXING_SUPPORT=0,
        # which requires that graph launch and non-captured launch to not overlap (I think,
        # that's how I interpret the documentation). I'm not sure if this is required.
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        torch.cuda.current_stream().wait_stream(s)
        # Captures the graph
        # To allow capture, automatically sets a side stream as the current stream in the context
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=torch.cuda.graphs.graph_pool_handle()):
            self.outputs = self.forward_static_1(
                    self.hidden_states,
                    inference_params=inference_params
                    # self.conv_state,
                    # self.ssm_state,
                )
        self.graph = graph 


        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                self.outputs_2 = self.forward_static_2(
                    self.x,
                    self.residual
                )
        s.synchronize()
        # This might be needed for correctness if we run with NCCL_GRAPH_MIXING_SUPPORT=0,
        # which requires that graph launch and non-captured launch to not overlap (I think,
        # that's how I interpret the documentation). I'm not sure if this is required.
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        torch.cuda.current_stream().wait_stream(s)
        # Captures the graph
        # To allow capture, automatically sets a side stream as the current stream in the context
        graph2 = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph2, pool=torch.cuda.graphs.graph_pool_handle()):
            self.outputs_2 = self.forward_static_2(
                    self.x,
                    self.residual
                )
        self.graph2 = graph2