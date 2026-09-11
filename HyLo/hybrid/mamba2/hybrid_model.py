# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

from mamba_ssm.ops.triton.layer_norm import RMSNorm
from torch import Tensor
from transformers.activations import ACT2FN

from hybrid.mamba2.hybrid_mamba2_layer import Mamba2
from hybrid.hybrid_modeling import HybridDynamicCache

import torch
import torch.nn as nn

try:
    from torch.distributed.tensor import DTensor
    _HAS_DTENSOR = True
except ImportError:
    _HAS_DTENSOR = False


def _maybe_all_gather_dtensor(t: Tensor):
    """If *t* is a DTensor (e.g. sharded by CP along the seq dim), materialise
    the full tensor on every rank.  Returns (full_tensor, placements, mesh) so
    the caller can re-shard later.  If *t* is a regular tensor, returns it
    unchanged with None metadata."""
    if _HAS_DTENSOR and isinstance(t, DTensor):
        placements = t.placements
        mesh = t.device_mesh
        return t.full_tensor(), placements, mesh
    return t, None, None


def _maybe_redistribute_as_dtensor(t: Tensor, placements, mesh):
    """Convert *t* back to a DTensor with the given placements/mesh.  No-op
    when placements is None (i.e. CP is not active)."""
    if placements is None:
        return t
    from torch.distributed.tensor import Shard, distribute_tensor
    return distribute_tensor(t, mesh, placements)

class MLP(nn.Module):
    def __init__(self, config):
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


class Mamba2DecoderLayer(nn.Module):
    def __init__(
        self,
        config,
        layer_idx: int,
        device=None,
        dtype=None,
        residual_in_fp32=True,
    ):
        super(Mamba2DecoderLayer, self).__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.layer_idx = layer_idx
        self.config = config
        self.mamba = Mamba2(d_model=config.d_model, d_xb=config.d_xb, d_inner=config.d_inner, layer_idx=layer_idx, **config.ssm_cfg, **factory_kwargs)
        self.mlp = MLP(config=config)
        self.input_layernorm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.residual_in_fp32 = True
        
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mamba.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

    def forward(self, hidden_states: Tensor, past_key_value=None, *args, **kwargs):
        # If CP is active, hidden_states is a DTensor sharded along seq dim.
        # Mamba2 needs the full sequence (SSM is sequential), so all-gather
        # before the scan and redistribute after.
        hidden_states, cp_placements, cp_mesh = _maybe_all_gather_dtensor(hidden_states)

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.mamba(hidden_states, past_key_value=past_key_value)
        attn_output = hidden_states.clone()
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        # Re-shard back to DTensor so downstream layers see the CP layout
        hidden_states = _maybe_redistribute_as_dtensor(hidden_states, cp_placements, cp_mesh)
        
        # so here is just to be compatible with Transformer
        if kwargs is None:
            return (hidden_states)
        else:
            past_key_value = kwargs.get("past_key_value", None)
            output_attentions = kwargs.get("output_attentions", False)
            use_cache = kwargs.get("use_cache", False)
            
            outputs = (hidden_states,)

            if past_key_value is not None:
                dummy_keys = torch.ones(
                    1, 1, hidden_states.size(1), 1, device=hidden_states.device, dtype=hidden_states.dtype
                )
                dummy_values = torch.ones(
                    1, 1, hidden_states.size(1), 1, device=hidden_states.device, dtype=hidden_states.dtype
                )
                # Update kv cache with dummy values
                past_key_value.update(dummy_keys, dummy_values, self.layer_idx)

            if output_attentions:
                outputs += (attn_output,)

            if use_cache:
                outputs += (past_key_value,)
                
            return outputs