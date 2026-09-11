# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
# GDN (Gated Delta Network) decoder layer wrapping fla.layers.GatedDeltaNet.

from __future__ import annotations

import os
import sys

# Allow loading fla from repo submodule when not installed via pip
_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_fla_path = os.path.join(_repo_root, "3rdparty", "flash-linear-attention")
if os.path.isdir(_fla_path) and _fla_path not in sys.path:
    sys.path.insert(0, _fla_path)

from torch import Tensor
import torch
import torch.nn as nn
from transformers.activations import ACT2FN

from mamba_ssm.ops.triton.layer_norm import RMSNorm
from fla.layers import GatedDeltaNet


class _GDNCacheAdapter:
    """Thin cache adapter that provides the interface fla's GatedDeltaNet expects.

    Attaches itself to the parent HybridDynamicCache so it shares the same
    lifecycle (created per-generation, passed through each forward step).
    """

    def __init__(self):
        self._states: dict[int, dict] = {}

    def __len__(self):
        if not self._states:
            return 0
        return max(self._states.keys()) + 1

    def __getitem__(self, layer_idx: int):
        return self._states.get(layer_idx)

    def update(self, *, recurrent_state=None, conv_state=None,
               layer_idx=0, offset=1, **kwargs):
        if layer_idx not in self._states:
            self._states[layer_idx] = {
                'recurrent_state': None,
                'conv_state': None,
            }
        state = self._states[layer_idx]
        if recurrent_state is not None:
            state['recurrent_state'] = recurrent_state
        if conv_state is not None:
            state['conv_state'] = conv_state
        return state


class MLP(nn.Module):
    """Same MLP as in Mamba2DecoderLayer for consistency."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = getattr(config, "d_model", config.hidden_size)
        self.intermediate_size = getattr(config, "intermediate_size", None) or (4 * self.hidden_size)
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[getattr(config, "hidden_act", "silu")]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class GDNDecoderLayer(nn.Module):
    """
    Decoder layer that uses Gated Delta Network (GDN) from fla as the mixer.
    API matches Mamba2DecoderLayer so it can be used interchangeably in the hybrid stack.
    """

    def __init__(
        self,
        config,
        layer_idx: int,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config
        hidden_size = getattr(config, "d_model", config.hidden_size)
        rms_norm_eps = getattr(config, "rms_norm_eps", 1e-5)
        # GDN: num_heads * head_dim = 0.75 * hidden_size when use_gate=True
        gdn_num_heads = getattr(config, "gdn_num_heads", None)
        gdn_head_dim = getattr(config, "gdn_head_dim", None)
        if gdn_num_heads is None or gdn_head_dim is None:
            key_dim = int(0.75 * hidden_size)
            # default: 6 heads, head_dim 256 for hidden_size 2048
            gdn_num_heads = 6
            gdn_head_dim = key_dim // gdn_num_heads
        # Persist the resolved values back onto the shared config so that
        # `HybridModelWrapper.save_config` writes the actual dims to disk
        # (instead of leaving them as `null`, which has caused load-time
        # shape mismatches on every published GDN ILD checkpoint).
        try:
            if getattr(config, "gdn_num_heads", None) is None:
                config.gdn_num_heads = gdn_num_heads
            if getattr(config, "gdn_head_dim", None) is None:
                config.gdn_head_dim = gdn_head_dim
        except Exception:
            # Some config classes may be frozen / immutable; harmless to skip.
            pass
        self.gdn = GatedDeltaNet(
            hidden_size=hidden_size,
            expand_v=2,
            head_dim=gdn_head_dim,
            num_heads=gdn_num_heads,
            num_v_heads=gdn_num_heads,
            mode="chunk",
            use_gate=True,
            use_short_conv=True,
            conv_size=4,
            conv_bias=False,
            layer_idx=layer_idx,
            norm_eps=rms_norm_eps,
        )
        self.mlp = MLP(config=config)
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        if device is not None:
            self.to(device)
        if dtype is not None:
            self.to(dtype)

    def forward(self, hidden_states: Tensor, past_key_value=None, *args, **kwargs):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        use_cache = kwargs.get("use_cache", False)

        # Obtain (or create) a fla-compatible cache adapter stored on the parent
        # HybridDynamicCache so it persists across generation steps.
        gdn_cache = None
        if use_cache and past_key_value is not None:
            if not hasattr(past_key_value, '_gdn_cache'):
                past_key_value._gdn_cache = _GDNCacheAdapter()
            gdn_cache = past_key_value._gdn_cache

        o, _, gdn_cache = self.gdn(
            hidden_states,
            attention_mask=None,
            past_key_values=gdn_cache,
            use_cache=use_cache,
            output_attentions=False,
        )

        if use_cache and past_key_value is not None:
            past_key_value._gdn_cache = gdn_cache

        attn_output = o.clone()
        hidden_states = residual + o

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        output_attentions = kwargs.get("output_attentions", False)
        if output_attentions:
            outputs += (attn_output,)
        if use_cache and past_key_value is not None:
            outputs += (past_key_value,)
        return outputs
