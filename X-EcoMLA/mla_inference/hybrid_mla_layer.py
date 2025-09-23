# coding=utf-8
# Copyright 2023 DeepSeek-AI and The HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch DeepSeek model."""
import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import (
    AttentionMaskConverter,
    _prepare_4d_attention_mask,
    _prepare_4d_causal_attention_mask,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import (
    ALL_LAYERNORM_LAYERS,
    is_torch_greater_or_equal_than_1_13,
)
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from transformers.utils.import_utils import is_torch_fx_available
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

import torch.distributed as dist
import numpy as np
from einops import rearrange

from mla.hybrid_mla_layer import (
    DeepseekV3RMSNorm,
    DeepseekV3RotaryEmbedding,
    DeepseekV3LinearScalingRotaryEmbedding,
    DeepseekV3YarnRotaryEmbedding,
    DeepseekV3DynamicNTKScalingRotaryEmbedding,
    DeepseekV3YarnRotaryEmbedding
    )
from huggingface_hub import PyTorchModelHubMixin

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa


# This makes `_prepare_4d_causal_attention_mask` a leaf function in the FX graph.
# It means that the function will not be traced through and simply appear as a node in the graph.
if is_torch_fx_available():
    if not is_torch_greater_or_equal_than_1_13:
        import torch.fx

    _prepare_4d_causal_attention_mask = torch.fx.wrap(_prepare_4d_causal_attention_mask)


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "DeepseekV3Config"


def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(
        torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0)
    )
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )

ALL_LAYERNORM_LAYERS.append(DeepseekV3RMSNorm)

def _update_kv_cache(kv, inference_params, layer_idx):
    """kv: (batch_size, seqlen, head_dim) or (batch_size, 1, head_dim)"""
    # Pre-allocate memory for key-values for inference.
    assert layer_idx in inference_params.key_value_memory_dict
    kv_cache, _ = inference_params.key_value_memory_dict[layer_idx]
    # Adjust key and value for inference
    batch_start = inference_params.batch_size_offset
    batch_end = batch_start + kv.shape[0]
    sequence_start = inference_params.seqlen_offset
    sequence_end = sequence_start + kv.shape[1]
    assert batch_end <= kv_cache.shape[0]
    assert sequence_end <= kv_cache.shape[1]
    assert kv_cache is not None
    kv_cache[batch_start:batch_end, sequence_start:sequence_end, ...] = kv
    return kv_cache[batch_start:batch_end, :sequence_end, ...]

# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors."""

    if position_ids is not None:
        cos = cos[position_ids:position_ids+1].unsqueeze(0).unsqueeze(unsqueeze_dim)
        sin = sin[position_ids:position_ids+1].unsqueeze(0).unsqueeze(unsqueeze_dim)
    else:
        # If position_ids are not provided, we use the default positions
        seq_len = q.shape[-2]
        cos = cos[:seq_len].unsqueeze(0).unsqueeze(unsqueeze_dim)
        sin = sin[:seq_len].unsqueeze(0).unsqueeze(unsqueeze_dim)
    
    b, h, s, d = q.shape
    q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    b, h, s, d = k.shape
    k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
    

# Copied from transformers.models.llama.modeling_llama.LlamaAttention with Llama->DeepseekV3
class DeepseekV3FlashAttention2(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config, layer_idx: Optional[int] = None, *args, **kwargs):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        if str(layer_idx) in config.layer_rank_list:
            config.kv_lora_rank = config.layer_rank_list[str(layer_idx)]["kv_rank"]
            config.q_lora_rank =  config.layer_rank_list[str(layer_idx)]["q_rank"]

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = self.num_heads

        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim

        self.is_causal = True

        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(
                self.hidden_size, self.num_heads * self.q_head_dim, bias=False
            )
        else:
            self.q_a_proj = nn.Linear(
                self.hidden_size, config.q_lora_rank, bias=config.attention_bias
            )
            if config.use_lora_layer_norm:
                self.q_a_layernorm = DeepseekV3RMSNorm(config.q_lora_rank)
            else:
                self.q_a_layernorm = nn.Identity()
            self.q_b_proj = nn.Linear(
                config.q_lora_rank, self.num_heads * self.q_head_dim, bias=False
            )

        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size,
            config.kv_lora_rank + config.qk_rope_head_dim,
            bias=config.attention_bias,
        )
        if config.use_lora_layer_norm:
            self.kv_a_layernorm = DeepseekV3RMSNorm(config.kv_lora_rank)
        else:
            self.kv_a_layernorm = nn.Identity()
        self.kv_b_proj = nn.Linear(
            config.kv_lora_rank,
            self.num_kv_heads
            * (self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim),
            bias=False,
        )

        self.out_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=config.attention_bias,
        )
        self._init_rope()

        self.softmax_scale = self.q_head_dim ** (-0.5)
        if self.config.rope_scaling is not None:
            mscale_all_dim = self.config.rope_scaling.get("mscale_all_dim", 0)
            scaling_factor = self.config.rope_scaling["factor"]
            if mscale_all_dim:
                mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.softmax_scale = self.softmax_scale * mscale * mscale
    
    def _init_merge(self):
        wkv_b = self.kv_b_proj.weight.view(self.num_kv_heads, self.qk_nope_head_dim+self.v_head_dim, -1)
        wkv_b = torch.repeat_interleave(wkv_b, repeats=self.num_heads//self.num_kv_heads, dim=0)
        self.register_buffer("wk_b", wkv_b[:, :self.qk_nope_head_dim, :], persistent=True)
        self.register_buffer("wv_b", wkv_b[:, self.qk_nope_head_dim:, :], persistent=True)

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = DeepseekV3RotaryEmbedding(
                self.qk_rope_head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            rope_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if rope_type == "linear":
                self.rotary_emb = DeepseekV3LinearScalingRotaryEmbedding(
                    self.qk_rope_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif rope_type == "dynamic":
                self.rotary_emb = DeepseekV3DynamicNTKScalingRotaryEmbedding(
                    self.qk_rope_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif rope_type == "yarn":
                kwargs = {
                    key: self.config.rope_scaling[key]
                    for key in [
                        "original_max_position_embeddings",
                        "beta_fast",
                        "beta_slow",
                        "mscale",
                        "mscale_all_dim",
                    ]
                    if key in self.config.rope_scaling
                }
                self.rotary_emb = DeepseekV3YarnRotaryEmbedding(
                    self.qk_rope_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                    **kwargs,
                )
            elif rope_type == "llama3":
                print("already initialized otherwhere")
            else:
                raise ValueError(f"Unknown RoPE scaling type {rope_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.v_head_dim)
            .transpose(1, 2)
            .contiguous()
        )
    
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        dtype = self.kv_a_proj_with_mqa.weight.dtype if dtype is None else dtype
        device = self.kv_a_proj_with_mqa.weight.device
        conv_state = None
        kv_cache = torch.randn(
            batch_size, max_seqlen, self.kv_lora_rank+self.qk_rope_head_dim, dtype=dtype, device=device,
        )
        return kv_cache, conv_state

    def _update_kv_cache(self, kv, inference_params):
        """kv: (batch_size, seqlen, 2, nheads, head_dim) or (batch_size, 1, 2, nheads, head_dim)"""
        assert self.layer_idx is not None, "Generation requires layer_idx in the constructor"
        return _update_kv_cache(kv, inference_params, self.layer_idx)


    # Need further testing
    def _update_kvcache_attention(self, q_nope, q_pe, compressed_kv, inference_params):
        """Write kv to inference_params, then do attention"""
        
        # Prefiling
        if inference_params.seqlen_offset == 0 or not self.absorb:
            
            bsz = q_nope.shape[0]
            q_len = q_nope.shape[2]

            # Step #1: update the kv-cache
            compressed_kv = self._update_kv_cache(compressed_kv, inference_params)
            kv_cache_len = compressed_kv.shape[1]

            # Step #2: Separate normed compressed kv and kpe
            compressed_kv, k_pe = torch.split(
                compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
            )
            
            kv = (
                self.kv_b_proj(compressed_kv)
                .view(bsz, kv_cache_len, self.num_kv_heads, self.qk_nope_head_dim + self.v_head_dim)
                .transpose(1, 2)
            )
            
            kv = repeat_kv(kv, self.num_heads//self.num_kv_heads)

            k_nope, value_states = torch.split(
                kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
            )
            
            k_pe = k_pe.view(bsz, kv_cache_len, 1, self.qk_rope_head_dim).transpose(1, 2)

            query_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
            query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
            query_states[:, :, :, self.qk_nope_head_dim :] = q_pe

            key_states = k_pe.new_empty(bsz, self.num_heads, kv_cache_len, self.q_head_dim)
            key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
            key_states[:, :, :, self.qk_nope_head_dim :] = k_pe
            
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)
            value_states = value_states.transpose(1, 2)

            attn_output = flash_attn_func(
                query_states,
                key_states,
                value_states,
                0.0,
                softmax_scale=self.softmax_scale,
                causal=self.is_causal,
            )
            return attn_output
        else:
            # MLA inference using Matrix Absorb
            # Step #1: update the kv-cache
            compressed_kv = self._update_kv_cache(compressed_kv, inference_params)
            kv_cache_len = compressed_kv.shape[1]
            
            # Step #2: Separate normed compressed kv and kpe
            compressed_kv, k_pe = torch.split(
                compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
            )
            
            # Separete the matrix kv_b_proj for matrix absorbing
            wkv_b = self.kv_b_proj.weight.view(self.num_kv_heads, self.qk_nope_head_dim+self.v_head_dim, -1)
            wkv_b = torch.repeat_interleave(wkv_b, repeats=self.num_heads//self.num_kv_heads, dim=0)
            wk_b = wkv_b[:, :self.qk_nope_head_dim, :]
            wv_b = wkv_b[:, self.qk_nope_head_dim:, :]
            
            q_nope = torch.einsum("bshd,hdc->bshc", q_nope.transpose(1, 2), wk_b)
            scores = (torch.einsum("bshc,btc->bsht", q_nope, compressed_kv) +
                          torch.einsum("bshr,btr->bsht", q_pe.transpose(1, 2), k_pe)) * self.softmax_scale
            scores = scores.softmax(dim=-1).type_as(q_nope)
            x = torch.einsum("bsht,btc->bshc", scores, compressed_kv)
            x = torch.einsum("bshc,hdc->bshd", x, wv_b)
            return x


    def forward_static_1( 
            self, 
            hidden_states: torch.Tensor,
            inference_params=None,
            **kwargs,):

        x = hidden_states
        bsz, _, _ = x.size()
        
        # Get queries
        if self.q_lora_rank is None:
            q = self.q_proj(x) 
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))

        q = q.view(bsz, 1, self.num_heads, self.q_head_dim)

        # Divide it into q_nope and q_pe
        q_nope, q_pe = torch.split(
                q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
            )
        
        # Get the compressed kv and kp
        compressed_kv = self.kv_a_proj_with_mqa(x)
        compressed_kv, k_pe = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        compressed_kv = self.kv_a_layernorm(compressed_kv)
        k_pe = k_pe.view(bsz, 1, 1, self.qk_rope_head_dim)

        # Apply rope to k_pe and p_pe
        cos, sin = self.rotary_emb(k_pe, seq_len=inference_params.max_seqlen)
        
        return q_nope, q_pe, k_pe, cos, sin, compressed_kv


    def forward_static_2(self, x):        
        x = torch.einsum("bshc,hdc->bshd", x, self.wv_b)
        context = rearrange(x, "... h d -> ... (h d)")
        out = self.out_proj(context)
        return out


    def forward_dynamic(self, q_nope, q_pe, k_pe, cos, sin, compressed_kv, inference_params, position_ids=None):
        # Apply positional embedding
        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, inference_params.seqlen_offset)
        k_pe = k_pe.view(q_nope.shape[0], 1, self.qk_rope_head_dim)
        compressed_kv_comb = torch.cat([compressed_kv, k_pe], dim=-1)

        # Update kv cache
        compressed_kv = self._update_kv_cache(compressed_kv_comb, inference_params)
        
        # MLA inference implementation
        compressed_kv, k_pe = compressed_kv[..., :self.kv_lora_rank], compressed_kv[..., -self.qk_rope_head_dim:]

        q_nope = torch.einsum("bshd,hdc->bshc", q_nope, self.wk_b)
        scores = (torch.einsum("bshc,btc->bsht", q_nope, compressed_kv) +
                          torch.einsum("bshr,btr->bsht", q_pe, k_pe)) * self.softmax_scale
        scores = scores.softmax(dim=-1).type_as(q_nope)
        x = torch.einsum("bsht,btc->bshc", scores, compressed_kv)
        return x

    def forward(
            self, 
            hidden_states: torch.Tensor,
            inference_params=None,
            position_ids=None,
            **kwargs,
            ):
        """
        Arguments:
            x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim) if
                cu_seqlens is None and max_seqlen is None, else (total, hidden_dim) where total
                is the is the sum of the sequence lengths in the batch.
            inference_params: for generation. Adapted from Megatron-LM (and Apex)
            https://github.com/NVIDIA/apex/blob/3ff1a10f72ec07067c4e44759442329804ac5162/apex/transformer/testing/standalone_transformer_lm.py#L470
        """
        
        if inference_params is not None and self.layer_idx not in inference_params.key_value_memory_dict:
            inference_params.key_value_memory_dict[self.layer_idx] = self.allocate_inference_cache(
                hidden_states.shape[0], inference_params.max_seqlen, dtype=hidden_states.dtype
            )
        seqlen_offset = (
            0
            if inference_params is None
            else (
                inference_params.lengths_per_sample
                if inference_params.lengths_per_sample is not None
                else inference_params.seqlen_offset
            )
        )
        
        x = hidden_states
        bsz, q_len, _ = x.size()
        
        # Get queries
        if self.q_lora_rank is None:
            q = self.q_proj(x) 
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))

        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)

        # Divide it into q_nope and q_pe
        q_nope, q_pe = torch.split(
                q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
            )
        
        # Get the compressed kv and kp
        compressed_kv = self.kv_a_proj_with_mqa(x)
        compressed_kv, k_pe = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        compressed_kv = self.kv_a_layernorm(compressed_kv)
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)

        # Apply rope to k_pe and p_pe
        cos, sin = self.rotary_emb(k_pe, seq_len=inference_params.max_seqlen)

        positions = inference_params.seqlen_offset if position_ids is not None else None
        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, positions)

        # Prepare the contents that need to be stored in the kv cache
        k_pe = k_pe.view(bsz, q_len, self.qk_rope_head_dim)
        compressed_kv_comb = torch.cat([compressed_kv, k_pe], dim=-1)

        context = self._update_kvcache_attention(q_nope, q_pe, compressed_kv_comb, inference_params)
        context = rearrange(context, "... h d -> ... (h d)")
        out = self.out_proj(context)

        return out