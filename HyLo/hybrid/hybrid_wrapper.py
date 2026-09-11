# Copyright (c) 2023, Albert Gu, Tri Dao.
# Modifications Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
import os
import json

# ---------------------------------------------------------------------------
# AMD ROCm (MI300x/MI325x) compatibility shim for mamba_ssm >= 2.3.1.
#
# Upstream `mamba_ssm/__init__.py` unconditionally does
#     from mamba_ssm.modules.mamba3 import Mamba3
# and `Mamba3` transitively imports NVIDIA `cutlass`
# (mamba_ssm/ops/cute/mamba3/mamba3_step_fn.py: `import cutlass`).
# `cutlass` is NVIDIA-only, so on AMD ROCm ANY `from mamba_ssm....` import
# (e.g. the `RMSNorm` pulled in by hybrid/mamba2/hybrid_model.py and
# hybrid/gdn/hybrid_model.py) crashes with `ModuleNotFoundError: No module
# named 'cutlass'`, even for pure-GDN training runs that never touch Mamba2
# or Mamba3.
#
# We pre-register a stub for `mamba_ssm.modules.mamba3` in `sys.modules`
# BEFORE `mamba_ssm` loads, so the `from mamba_ssm.modules.mamba3 import
# Mamba3` line in mamba_ssm's package init resolves to our placeholder
# (`Mamba3 = None`) instead of executing the real module's cutlass import.
# Nothing in this repo references `mamba_ssm.Mamba3`, so the None
# placeholder is harmless. Must run before the `from hybrid.mamba2...` /
# `from hybrid.gdn...` imports below.
# ---------------------------------------------------------------------------
import sys as _sys
import types as _types
if "mamba_ssm.modules.mamba3" not in _sys.modules:
    _mamba3_stub = _types.ModuleType("mamba_ssm.modules.mamba3")
    _mamba3_stub.Mamba3 = None
    _sys.modules["mamba_ssm.modules.mamba3"] = _mamba3_stub

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from dataclasses import dataclass, field, asdict

from transformers import AutoModelForCausalLM, LlamaConfig
from transformers import LlamaModel, LlamaForCausalLM, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast

from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
from transformers.utils.hub import cached_file
from transformers.cache_utils import Cache, DynamicCache

from hybrid.mla.hybrid_model import MLADecoderLayer
from hybrid.mamba2.hybrid_model import Mamba2DecoderLayer
from hybrid.hybrid_config import HybridConfig
from hybrid.gdn import GDNDecoderLayer
from hybrid.hybrid_modeling import CustomLlamaForCausalLM
from typing import List, Optional, Tuple, Union

from util import load_safetensors_to_dict
from safetensors import safe_open
import logging

logger = logging.getLogger(__name__)


def _peek_gdn_dims_from_safetensors_dir(directory: str, hybrid_config) -> Tuple[Optional[int], Optional[int]]:
    """Inspect safetensors files in `directory` to infer the GDN num_heads/head_dim
    actually used to produce the checkpoint, by reading just the tensor shape
    metadata (no payload materialization).

    Returns (num_v_heads, head_dim) where:
      - num_v_heads = first found `model.layers.{i}.gdn.a_proj.weight`.shape[0]
      - head_dim    = first found `model.layers.{i}.gdn.o_norm.weight`.shape[0] // expand_v
        (GDNDecoderLayer hardcodes expand_v=2, so head_dim = head_v_dim // 2)

    Returns (None, None) if no GDN tensors are found (e.g. MLA-only checkpoint
    or non-safetensors checkpoint).
    """
    if not os.path.isdir(directory):
        return None, None
    a_proj_shape: Optional[List[int]] = None
    o_norm_shape: Optional[List[int]] = None
    safetensor_files = sorted(f for f in os.listdir(directory) if f.endswith('.safetensors'))
    n_layer = getattr(hybrid_config, "n_layer", 0)
    mla_layers = set(getattr(hybrid_config, "mla_layers", []) or [])
    for filename in safetensor_files:
        file_path = os.path.join(directory, filename)
        try:
            with safe_open(file_path, framework="pt") as f:
                keys = set(f.keys())
                for layer_idx in range(n_layer):
                    if layer_idx in mla_layers:
                        continue
                    a_key = f"model.layers.{layer_idx}.gdn.a_proj.weight"
                    o_key = f"model.layers.{layer_idx}.gdn.o_norm.weight"
                    if a_proj_shape is None and a_key in keys:
                        a_proj_shape = list(f.get_slice(a_key).get_shape())
                    if o_norm_shape is None and o_key in keys:
                        o_norm_shape = list(f.get_slice(o_key).get_shape())
                    if a_proj_shape is not None and o_norm_shape is not None:
                        break
        except Exception as e:
            logger.warning(f"Could not inspect {file_path} for GDN shapes: {e}")
            continue
        if a_proj_shape is not None and o_norm_shape is not None:
            break

    if a_proj_shape is None or o_norm_shape is None:
        return None, None
    num_v_heads = int(a_proj_shape[0])
    head_v_dim = int(o_norm_shape[0])
    head_dim = head_v_dim // 2
    return num_v_heads, head_dim


def _resolve_gdn_dims_from_checkpoint(directory: str, hybrid_config) -> None:
    """If `hybrid_config.linear_type == 'gdn'`, ensure `gdn_num_heads`/`gdn_head_dim`
    in `hybrid_config` reflect the actual checkpoint tensor shapes.

    Background: prior to this fix, `HybridModelWrapper.save_config` persisted
    the raw `HybridConfig` fields, which were typically `None` because the
    `GDNDecoderLayer` resolved its dimensions locally (via the
    `(6, key_dim//6)` fallback) without writing the values back. As a result,
    the published GDN ILD checkpoints all have `gdn_num_heads: null,
    gdn_head_dim: null` in their `hybrid_config.json`, even though the
    tensors inside them were saved at very different shapes — e.g.
    `Zebra-Qwen3-1.7B-ILD-GDN` => `(6, 256)`,
    `Zebra-Qwen3-4B-ILD-GDN`   => `(8, 240)`,
    `Zebra-3B-ILD-GDN`         => `(9, 256)`.
    Loading them with the code default rebuilds the GDN modules at the wrong
    shape and crashes `load_state_dict` with mismatch errors on
    `a_proj.weight`, `o_norm.weight`, `A_log`, `dt_bias`, `b_proj.weight`.
    The Qwen3-1.7B path historically worked only by arithmetic coincidence
    (`1536/6 == 256`), which is also where the in-code default came from.

    This helper peeks the safetensors header (no payload reads) and patches the
    config in place so the GDN modules are built at the right shape before
    `load_state_dict` is called. If the config already has explicit values that
    disagree with the tensors, we trust the tensors and warn loudly.
    """
    linear_type = (getattr(hybrid_config, "linear_type", None) or "").lower()
    if linear_type != "gdn":
        return
    inferred_num_heads, inferred_head_dim = _peek_gdn_dims_from_safetensors_dir(directory, hybrid_config)
    if inferred_num_heads is None or inferred_head_dim is None:
        return
    cfg_num_heads = getattr(hybrid_config, "gdn_num_heads", None)
    cfg_head_dim = getattr(hybrid_config, "gdn_head_dim", None)
    if cfg_num_heads is None or cfg_head_dim is None:
        logger.info(
            f"Inferred GDN dims from checkpoint safetensors: "
            f"gdn_num_heads={inferred_num_heads}, gdn_head_dim={inferred_head_dim} "
            f"(saved hybrid_config had nulls)."
        )
        hybrid_config.gdn_num_heads = inferred_num_heads
        hybrid_config.gdn_head_dim = inferred_head_dim
    elif cfg_num_heads != inferred_num_heads or cfg_head_dim != inferred_head_dim:
        logger.warning(
            f"GDN dims in hybrid_config ({cfg_num_heads}, {cfg_head_dim}) "
            f"disagree with checkpoint safetensors ({inferred_num_heads}, "
            f"{inferred_head_dim}). Using checkpoint values."
        )
        hybrid_config.gdn_num_heads = inferred_num_heads
        hybrid_config.gdn_head_dim = inferred_head_dim


def _copy_llama_attn_to_gdn(llama_attn, gdn_module, dtype, num_attention_heads=None, num_key_value_heads=None):
    """
    Copy Llama attention Q/K/V/O projections into GDN mixer for Stage 1 ILD init.
    Aligned with fla-org/hybrid-distillation: when teacher uses GQA (num_kv_heads < num_heads),
    K and V are repeated to full num_heads before copy (see student_layers.py GatedDeltaNetStudentV1–V4).
    When GDN key_dim/value_dim differ from teacher (e.g. 0.75*hidden_size, expand_v=2), we truncate
    or copy overlapping regions.
    """
    from einops import repeat
    tq = llama_attn.q_proj.weight.data.to(dtype)
    tk = llama_attn.k_proj.weight.data.to(dtype)
    tv = llama_attn.v_proj.weight.data.to(dtype)
    to = llama_attn.o_proj.weight.data.to(dtype)
    # GQA: repeat K/V to num_heads like hybrid-distillation (student_layers.py GatedDeltaNetStudentV1)
    if num_attention_heads is not None and num_key_value_heads is not None and num_key_value_heads < num_attention_heads:
        g = num_attention_heads // num_key_value_heads
        head_dim = tk.shape[0] // num_key_value_heads
        # (num_kv_heads * head_dim, in) -> repeat each kv head g times -> (num_heads * head_dim, in)
        tk = repeat(tk, "(kv h) d -> (kv g h) d", kv=num_key_value_heads, h=head_dim, g=g)
        tv = repeat(tv, "(kv h) d -> (kv g h) d", kv=num_key_value_heads, h=head_dim, g=g)
    # PyTorch Linear: weight (out_features, in_features)
    key_dim = gdn_module.q_proj.weight.shape[0]
    value_dim = gdn_module.v_proj.weight.shape[0]
    # Q
    copy_q = min(tq.shape[0], key_dim)
    gdn_module.q_proj.weight.data[:copy_q, :].copy_(tq[:copy_q, :])
    # K
    copy_k = min(tk.shape[0], key_dim)
    gdn_module.k_proj.weight.data[:copy_k, :].copy_(tk[:copy_k, :])
    # V
    copy_v = min(tv.shape[0], value_dim)
    gdn_module.v_proj.weight.data[:copy_v, :].copy_(tv[:copy_v, :])
    # O: teacher (hidden_size, hidden_size), GDN (hidden_size, value_dim)
    copy_o_cols = min(to.shape[1], value_dim)
    gdn_module.o_proj.weight.data[:, :copy_o_cols].copy_(to[:, :copy_o_cols])
    # g_proj / conv / A_log / dt_bias / a_proj / b_proj stay at default init

HYBRID_CONFIG_NAME = "hybrid_config.json"


def _is_local_path(model_name: str) -> bool:
    """Return True if model_name is an existing local directory (not a Hub repo id)."""
    if not model_name or not isinstance(model_name, str):
        return False
    # Absolute path or path containing sep => treat as local if it exists
    if os.path.isdir(model_name):
        return True
    return False


def load_config_hf(model_name):
    if _is_local_path(model_name):
        config_path = os.path.join(model_name, CONFIG_NAME)
        if os.path.isfile(config_path):
            with open(config_path) as f:
                return json.load(f)
        raise FileNotFoundError(f"Config file {CONFIG_NAME} not found in {model_name}")
    resolved_archive_file = cached_file(model_name, CONFIG_NAME, _raise_exceptions_for_missing_entries=False)
    return json.load(open(resolved_archive_file))

def load_state_dict_hf(model_name: str, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> dict:
    """Loads a state dict from a local path or the Hugging Face Hub, supporting safetensors."""
    if _is_local_path(model_name):
        bin_path = os.path.join(model_name, WEIGHTS_NAME)
        safetensors_path = os.path.join(model_name, "model.safetensors")
        if os.path.isfile(bin_path):
            return torch.load(bin_path, map_location="cpu", weights_only=False)
        if os.path.isfile(safetensors_path):
            return load_safetensors_to_dict(model_name)
        raise FileNotFoundError(f"Neither {WEIGHTS_NAME} nor model.safetensors found in {model_name}")
    resolved_archive_file = cached_file(model_name, WEIGHTS_NAME, _raise_exceptions_for_missing_entries=False)
    # Try to load from safetensors if .bin file is not found
    if not resolved_archive_file:
        resolved_archive_file = cached_file(model_name, 'model.safetensors', _raise_exceptions_for_missing_entries=False)
        if resolved_archive_file:
            state_dict = load_safetensors_to_dict(os.path.dirname(resolved_archive_file))
        else:
            raise FileNotFoundError(f"Neither {WEIGHTS_NAME} nor 'model.safetensors' found in {model_name}")
    else:
        state_dict = torch.load(resolved_archive_file, map_location="cpu", weights_only=False)
    return state_dict


class HybridModelWrapper(nn.Module):

    def __init__(self, checkpoint_path, transformer_model, hybrid_config, mla_layers, dtype, init_with_svd, init_with_kqvo, mamba_model=None, mla_model=None, linear_model=None, linear_type="mamba", load_from_hub=False, attn_implementation="flash_attention_2", **kwargs):
        super(HybridModelWrapper, self).__init__()
                
        self.hybrid_config = hybrid_config
        self.mla_layers = mla_layers
        self.model = transformer_model
        self.config = self.model.config
        self.dtype = dtype
        self.linear_type = (getattr(hybrid_config, "linear_type", None) or linear_type).lower()
        
        if not hasattr(self.config, 'head_dim'):
            self.config.head_dim = self.config.hidden_size // self.config.num_attention_heads

        # ILD checkpoints: linear_model (Mamba or GDN) and mla_model
        has_linear_ild = linear_model is not None and mla_model is not None
        self.has_linear_ild = has_linear_ild
        self.has_mla_ild = mla_model is not None

        for layer_idx in range(hybrid_config.n_layer):
            if layer_idx in mla_layers:
                
                # Initialize the MLA layer
                MLA_encoder = MLADecoderLayer(
                    hybrid_config,
                    layer_idx,
                    device="cuda",
                    dtype=dtype,
                    attn_implementation=attn_implementation,
                )
                
                # Initialize from ILD
                if has_linear_ild:
                    MLA_encoder.load_state_dict(mla_model.model.model.layers._modules[f'{layer_idx}'].state_dict()) 
                    MLA_encoder = MLA_encoder.to(dtype)
                    print(f"layerid: {layer_idx}, MLA initialization from MHA/GQA")
                else:  
                    # Initialize from SVD
                    if init_with_svd:

                        MLA_encoder.mlp.load_state_dict(transformer_model.model.layers._modules[f'{layer_idx}'].mlp.state_dict())
                        MLA_encoder.input_layernorm.load_state_dict(transformer_model.model.layers._modules[f'{layer_idx}'].input_layernorm.state_dict())
                        MLA_encoder.post_attention_layernorm.load_state_dict(transformer_model.model.layers._modules[f'{layer_idx}'].post_attention_layernorm.state_dict())
                        orig_o = transformer_model.model.layers._modules[f'{layer_idx}'].self_attn.o_proj.weight.data
                        mla_o_in = MLA_encoder.mla.out_proj.weight.shape[1]
                        MLA_encoder.mla.out_proj.weight.data.copy_(orig_o[:, :mla_o_in])
                        if MLA_encoder.mla.out_proj.bias is not None and transformer_model.model.layers._modules[f'{layer_idx}'].self_attn.o_proj.bias is not None:
                            MLA_encoder.mla.out_proj.bias.data.copy_(transformer_model.model.layers._modules[f'{layer_idx}'].self_attn.o_proj.bias.data)

                        q_matrix = transformer_model.model.layers._modules[f'{layer_idx}'].self_attn.q_proj.weight.data                   
                        q_rank = MLA_encoder.mla.re_init_q(q_matrix, self.config.head_dim, dtype)

                        k_matrix = transformer_model.model.layers._modules[f'{layer_idx}'].self_attn.k_proj.weight.data
                        v_matrix = transformer_model.model.layers._modules[f'{layer_idx}'].self_attn.v_proj.weight.data
                        kv_rank = MLA_encoder.mla.re_init_kv(k_matrix, v_matrix, self.config.num_key_value_heads, self.config.head_dim, dtype)
                    
                        print(f"Layernorm: {hybrid_config.use_lora_layer_norm}; layerid: {layer_idx}; q_rank: {q_rank}; kv_rank: {kv_rank}")

                        # keep dtype to be the same
                        MLA_encoder.mlp = MLA_encoder.mlp.to(dtype)
                        MLA_encoder.input_layernorm = MLA_encoder.input_layernorm.to(dtype)
                        MLA_encoder.post_attention_layernorm = MLA_encoder.post_attention_layernorm.to(dtype)

                self.model.model.layers[layer_idx] = MLA_encoder
            
            else:
                # Linear (non-MLA) layer: Mamba or GDN
                if self.linear_type == "gdn":
                    linear_encoder = GDNDecoderLayer(
                        hybrid_config,
                        layer_idx,
                        device="cuda",
                        dtype=dtype,
                    )
                    if has_linear_ild:
                        linear_encoder.load_state_dict(linear_model.model.model.layers._modules[f'{layer_idx}'].state_dict())
                        linear_encoder = linear_encoder.to(dtype)
                        print(f"layerid: {layer_idx}, GDN initialization from ILD checkpoint")
                    else:
                        if init_with_kqvo:
                            llayer = transformer_model.model.layers._modules[f'{layer_idx}']
                            linear_encoder.mlp.load_state_dict(llayer.mlp.state_dict())
                            linear_encoder.input_layernorm.load_state_dict(llayer.input_layernorm.state_dict())
                            linear_encoder.post_attention_layernorm.load_state_dict(llayer.post_attention_layernorm.state_dict())
                            _copy_llama_attn_to_gdn(
                                llayer.self_attn,
                                linear_encoder.gdn,
                                dtype,
                                num_attention_heads=getattr(self.config, "num_attention_heads", None),
                                num_key_value_heads=getattr(self.config, "num_key_value_heads", None),
                            )
                            linear_encoder.mlp = linear_encoder.mlp.to(dtype)
                            linear_encoder.input_layernorm = linear_encoder.input_layernorm.to(dtype)
                            linear_encoder.post_attention_layernorm = linear_encoder.post_attention_layernorm.to(dtype)
                            print(f"layerid: {layer_idx}, GDN initialization from teacher (norms+MLP+Q/K/V/O copy); Stage 1 ILD ready.")
                else:
                    # Mamba2
                    linear_encoder = Mamba2DecoderLayer(
                        hybrid_config,
                        layer_idx,
                        device="cuda",
                        dtype=dtype,
                    )
                    if has_linear_ild:
                        linear_encoder.load_state_dict(linear_model.model.model.layers._modules[f'{layer_idx}'].state_dict()) 
                        linear_encoder = linear_encoder.to(dtype)
                        print(f"layerid: {layer_idx}, SSM initialization from MHA/GQA")
                    else:
                        if init_with_kqvo:
                            linear_encoder.mlp.load_state_dict(transformer_model.model.layers._modules[f'{layer_idx}'].mlp.state_dict())
                            linear_encoder.input_layernorm.load_state_dict(transformer_model.model.layers._modules[f'{layer_idx}'].input_layernorm.state_dict())
                            linear_encoder.post_attention_layernorm.load_state_dict(transformer_model.model.layers._modules[f'{layer_idx}'].post_attention_layernorm.state_dict())
                            linear_encoder.mamba.out_proj.load_state_dict(transformer_model.model.layers._modules[f'{layer_idx}'].self_attn.o_proj.state_dict())

                            linear_encoder.mamba.in_proj.weight.data[hybrid_config.d_inner:hybrid_config.d_inner+hybrid_config.d_xb, :].copy_(transformer_model.model.layers._modules[f'{layer_idx}'].self_attn.v_proj.weight.data)
                            linear_encoder.mamba.in_proj.weight.data[hybrid_config.d_inner+hybrid_config.d_xb:hybrid_config.d_inner+2*hybrid_config.d_xb, :].copy_(transformer_model.model.layers._modules[f'{layer_idx}'].self_attn.k_proj.weight.data)
                            linear_encoder.mamba.in_proj.weight.data[hybrid_config.d_inner+2*hybrid_config.d_xb:2*hybrid_config.d_inner+2*hybrid_config.d_xb, :].copy_(transformer_model.model.layers._modules[f'{layer_idx}'].self_attn.q_proj.weight.data)
                            print(f"layerid: {layer_idx}, SSM initialization from MHA/GQA")
                        
                            linear_encoder.mlp = linear_encoder.mlp.to(dtype)
                            linear_encoder.input_layernorm = linear_encoder.input_layernorm.to(dtype)
                            linear_encoder.post_attention_layernorm = linear_encoder.post_attention_layernorm.to(dtype)

                self.model.model.layers[layer_idx] = linear_encoder
        
        if checkpoint_path is not None:
            if load_from_hub:
                # load from a huggingface hub
                self.model.load_state_dict(load_state_dict_hf(checkpoint_path, device=torch.device("cpu"), dtype=dtype))                
            else:
                # load from a local directory
                if os.path.exists(f"{checkpoint_path}/pytorch_model.bin"):
                    # support save from bin file
                    self.model.load_state_dict(torch.load(f"{checkpoint_path}/pytorch_model.bin", map_location=torch.device("cpu")))
                else:
                    # support save from safetensors                    
                    self.model.load_state_dict(load_safetensors_to_dict(checkpoint_path), strict=False)
        self.model = self.model.to(dtype).cuda()


    def forward(
        self,
        input_ids,
        **kwargs,
    ):          
        return self.model(input_ids, **kwargs)

    def generate(
        self,
        input_ids,
        **kwargs,
    ):
        output = self.model.generate(
            input_ids,
            use_cache=True,
            **kwargs,
        )
        return output
    
    @staticmethod
    def init_distillation(
        checkpoint_path,
        tranformer_name,
        hybrid_config,
        mla_layers,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        init_with_svd=True,
        init_with_kqvo=True,
        mamba_model_path=None,
        mla_model_path=None,
        linear_ILD_path=None,
        linear_type="mamba",
        **kwargs,
    ):  
        # linear_ILD_path overrides mamba_model_path when set (generic linear ILD checkpoint)
        linear_model_path = linear_ILD_path if linear_ILD_path is not None else mamba_model_path
        if linear_model_path is not None:
            linear_model = HybridModelWrapper.from_pretrained(linear_model_path, torch_dtype=dtype, attn_implementation=attn_implementation)
        else:
            linear_model = None
        if mla_model_path is not None:
            mla_model = HybridModelWrapper.from_pretrained(mla_model_path, torch_dtype=dtype, attn_implementation=attn_implementation)
        else:
            mla_model = None
        linear_type_resolved = (getattr(hybrid_config, "linear_type", None) or linear_type or "mamba").lower()

        # Propagate GDN dims from the ILD checkpoint to the outer student
        # config. The ILD wrapper's `hybrid_config` is now authoritative for
        # GDN topology because `from_pretrained_*` infers the actual values
        # from the saved tensor shapes when the on-disk config has nulls
        # (see `_resolve_gdn_dims_from_checkpoint`). The student's outer
        # config typically inherits its GDN dims from the YAML, which in
        # most cases is left null, so without this propagation the student
        # GDN modules are built at the wrong shape and `load_state_dict` of
        # the ILD layer state into the student crashes with a
        # `size mismatch for gdn.A_log/a_proj/b_proj/o_norm/dt_bias` error.
        if (linear_type_resolved == "gdn" and linear_model is not None
                and getattr(linear_model, "hybrid_config", None) is not None):
            ild_cfg = linear_model.hybrid_config
            ild_num_heads = getattr(ild_cfg, "gdn_num_heads", None)
            ild_head_dim = getattr(ild_cfg, "gdn_head_dim", None)
            outer_num_heads = getattr(hybrid_config, "gdn_num_heads", None)
            outer_head_dim = getattr(hybrid_config, "gdn_head_dim", None)
            if ild_num_heads is not None and ild_head_dim is not None:
                if outer_num_heads is None or outer_head_dim is None:
                    logger.info(
                        f"Adopting GDN dims from ILD checkpoint into student config: "
                        f"gdn_num_heads={ild_num_heads}, gdn_head_dim={ild_head_dim} "
                        f"(was: {outer_num_heads}, {outer_head_dim})."
                    )
                    hybrid_config.gdn_num_heads = ild_num_heads
                    hybrid_config.gdn_head_dim = ild_head_dim
                elif (outer_num_heads != ild_num_heads
                      or outer_head_dim != ild_head_dim):
                    logger.warning(
                        f"GDN dims mismatch between student YAML "
                        f"({outer_num_heads}, {outer_head_dim}) and ILD checkpoint "
                        f"({ild_num_heads}, {ild_head_dim}). Adopting the ILD values "
                        f"because Stage-2 ILD weight transfer requires identical "
                        f"GDN topology."
                    )
                    hybrid_config.gdn_num_heads = ild_num_heads
                    hybrid_config.gdn_head_dim = ild_head_dim

        transformer_model = CustomLlamaForCausalLM.from_pretrained(tranformer_name, hybrid_config, torch_dtype=dtype)
        return HybridModelWrapper(
            checkpoint_path, transformer_model, hybrid_config, mla_layers, dtype,
            init_with_svd, init_with_kqvo,
            mamba_model=None, mla_model=mla_model, linear_model=linear_model, linear_type=linear_type_resolved,
            attn_implementation=attn_implementation,
        )

    @staticmethod
    def _apply_yarn_factor(hybrid_config, yarn_factor):
        """Override rope_scaling factor in hybrid_config if yarn_factor is provided."""
        if yarn_factor is None:
            return
        yarn_factor = float(yarn_factor)
        if hybrid_config.rope_scaling is None:
            hybrid_config.rope_scaling = {
                "rope_type": "yarn",
                "factor": yarn_factor,
                "original_max_position_embeddings": getattr(hybrid_config, "max_position_embeddings", 2048),
            }
        else:
            hybrid_config.rope_scaling["factor"] = yarn_factor
        logger.info(f"Overriding rope_scaling factor to {yarn_factor}: {hybrid_config.rope_scaling}")

    @staticmethod
    def from_pretrained_local(pretrained_model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", yarn_factor=None):
        config_data = load_config_hf(pretrained_model_name)
        with open(f'{pretrained_model_name}/{HYBRID_CONFIG_NAME}', 'r') as json_file:
            config_dict = json.load(json_file)
        hybrid_config = HybridConfig(**config_dict)
        HybridModelWrapper._apply_yarn_factor(hybrid_config, yarn_factor)
        # Patch null GDN dims from the actual checkpoint tensor shapes (see
        # _resolve_gdn_dims_from_checkpoint docstring for why this is needed).
        _resolve_gdn_dims_from_checkpoint(pretrained_model_name, hybrid_config)
        transformer_model = CustomLlamaForCausalLM(LlamaConfig(**config_data), hybrid_config, data_dtype=torch_dtype)
        linear_type = getattr(hybrid_config, "linear_type", "mamba")
        return HybridModelWrapper(pretrained_model_name, transformer_model, hybrid_config, hybrid_config.mla_layers, torch_dtype, init_with_svd=False, init_with_kqvo=False, linear_model=None, linear_type=linear_type, attn_implementation=attn_implementation) 

    @staticmethod
    def from_pretrained_hub(pretrained_model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", yarn_factor=None):
        config_data = load_config_hf(pretrained_model_name)
        resolved_archive_file = cached_file(pretrained_model_name, HYBRID_CONFIG_NAME, _raise_exceptions_for_missing_entries=False)
        config_dict = json.load(open(resolved_archive_file))
        hybrid_config = HybridConfig(**config_dict)
        HybridModelWrapper._apply_yarn_factor(hybrid_config, yarn_factor)
        # Patch null GDN dims from the actual checkpoint tensor shapes. The Hub
        # snapshot directory is the parent of the resolved hybrid_config.json
        # symlink and contains the same safetensors symlinks.
        snapshot_dir = os.path.dirname(resolved_archive_file) if resolved_archive_file else None
        if snapshot_dir is not None:
            _resolve_gdn_dims_from_checkpoint(snapshot_dir, hybrid_config)
        transformer_model = CustomLlamaForCausalLM(LlamaConfig(**config_data), hybrid_config, data_dtype=torch_dtype)
        linear_type = getattr(hybrid_config, "linear_type", "mamba")
        return HybridModelWrapper(pretrained_model_name, transformer_model, hybrid_config, hybrid_config.mla_layers, torch_dtype, init_with_svd=False, init_with_kqvo=False, linear_model=None, linear_type=linear_type, load_from_hub=True, attn_implementation=attn_implementation) 

    @staticmethod
    def from_pretrained(pretrained_model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", yarn_factor=None):
        if os.path.exists(pretrained_model_name):
            return HybridModelWrapper.from_pretrained_local(pretrained_model_name, torch_dtype, attn_implementation, yarn_factor=yarn_factor)
        else:
            return HybridModelWrapper.from_pretrained_hub(pretrained_model_name, torch_dtype, attn_implementation, yarn_factor=yarn_factor)

    def save_config(self, save_directory, config_file_path):
        os.makedirs(save_directory, exist_ok=True)
        config_path = os.path.join(save_directory, 'hybrid_config.json')
        with open(config_path, 'w') as f:
            config_dict = asdict(self.hybrid_config)
            json.dump(config_dict, f, indent=4)
        os.system("cp {} {}/".format(config_file_path, save_directory))