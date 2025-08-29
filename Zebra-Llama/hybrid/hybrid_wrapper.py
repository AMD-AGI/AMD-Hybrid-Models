# Copyright (c) 2023, Albert Gu, Tri Dao.
import os
import json

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
from hybrid.hybrid_modeling import CustomLlamaForCausalLM
from typing import List, Optional, Tuple, Union

from util import load_safetensors_to_dict
import logging

logger = logging.getLogger(__name__)

HYBRID_CONFIG_NAME = "hybrid_config.json"

def load_config_hf(model_name):
    resolved_archive_file = cached_file(model_name, CONFIG_NAME, _raise_exceptions_for_missing_entries=False)
    return json.load(open(resolved_archive_file))

def load_state_dict_hf(model_name: str, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> dict:
    """Loads a state dict from a local path or the Hugging Face Hub, supporting safetensors."""
    resolved_archive_file = cached_file(model_name, WEIGHTS_NAME, _raise_exceptions_for_missing_entries=False)
    
    # Try to load from safetensors if .bin file is not found
    if not resolved_archive_file:
        resolved_archive_file = cached_file(model_name, 'model.safetensors', _raise_exceptions_for_missing_entries=False)
        if resolved_archive_file:
            # Assumes load_safetensors_to_dict can handle both a file path and a directory
            state_dict = load_safetensors_to_dict(os.path.dirname(resolved_archive_file))
        else:
            raise FileNotFoundError(f"Neither {WEIGHTS_NAME} nor 'model.safetensors' found in {model_name}")
    else:
        state_dict = torch.load(resolved_archive_file, map_location="cpu", weights_only=False)
        
    return state_dict


class HybridModelWrapper(nn.Module):

    def __init__(self, checkpoint_path, transformer_model, hybrid_config, mla_layers, dtype, init_with_svd, init_with_kqvo, mamba_model=None, mla_model=None, load_from_hub=False, **kwargs):
        super(HybridModelWrapper, self).__init__()
                
        self.hybrid_config = hybrid_config
        self.mla_layers = mla_layers
        self.model = transformer_model
        self.config = self.model.config
        self.dtype = dtype
        self.layer_rank_list = self.hybrid_config.layer_rank_list
        
        if not hasattr(self.config, 'head_dim'):
            self.config.head_dim = self.config.hidden_size // self.config.num_attention_heads

        for layer_idx in range(hybrid_config.n_layer):
            if layer_idx in mla_layers:
                
                # Initialize the MLA layer
                MLA_encoder = MLADecoderLayer(
                    hybrid_config,
                    layer_idx,
                    device="cuda",
                    dtype=dtype,
                )
                
                # Initialize from ILD
                if mamba_model is not None and mla_model is not None:
                    MLA_encoder.load_state_dict(mla_model.model.model.layers._modules[f'{layer_idx}'].state_dict()) 
                    MLA_encoder = MLA_encoder.to(dtype)
                    print(f"layerid: {layer_idx}, MLA initialization from MHA/GQA")
                else:  
                    # Initialize from SVD
                    if init_with_svd:

                        MLA_encoder.mlp.load_state_dict(transformer_model.model.layers._modules[f'{layer_idx}'].mlp.state_dict())
                        MLA_encoder.input_layernorm.load_state_dict(transformer_model.model.layers._modules[f'{layer_idx}'].input_layernorm.state_dict())
                        MLA_encoder.post_attention_layernorm.load_state_dict(transformer_model.model.layers._modules[f'{layer_idx}'].post_attention_layernorm.state_dict())
                        MLA_encoder.mla.out_proj.load_state_dict(transformer_model.model.layers._modules[f'{layer_idx}'].self_attn.o_proj.state_dict())

                        use_dynamic_rank = not layer_idx in [0, hybrid_config.n_layer-1] or not(hybrid_config.use_fixed_rank_for_first_and_last_block)
                    
                        q_matrix = transformer_model.model.layers._modules[f'{layer_idx}'].self_attn.q_proj.weight.data                   
                        q_rank = MLA_encoder.mla.re_init_q(q_matrix, self.config.head_dim, dtype, use_dynamic_rank)

                        k_matrix = transformer_model.model.layers._modules[f'{layer_idx}'].self_attn.k_proj.weight.data
                        v_matrix = transformer_model.model.layers._modules[f'{layer_idx}'].self_attn.v_proj.weight.data
                        kv_rank = MLA_encoder.mla.re_init_kv(k_matrix, v_matrix, self.config.num_key_value_heads, self.config.head_dim, dtype, use_dynamic_rank)
                    
                        print(f"dynamic_rank: {use_dynamic_rank & (hybrid_config.q_energy_ratio is not None)}; Layernorm: {hybrid_config.use_lora_layer_norm}; layerid: {layer_idx}; q_rank: {q_rank}; kv_rank: {kv_rank}; total_rank: {q_rank+kv_rank}")
                        self.layer_rank_list[str(layer_idx)] = {"q_rank": q_rank, "kv_rank": kv_rank}

                        # keep dtype to be the same
                        MLA_encoder.mlp = MLA_encoder.mlp.to(dtype)
                        MLA_encoder.input_layernorm = MLA_encoder.input_layernorm.to(dtype)
                        MLA_encoder.post_attention_layernorm = MLA_encoder.post_attention_layernorm.to(dtype)

                self.model.model.layers[layer_idx] = MLA_encoder
            
            else:

                # Initialize the Mamba2 layer
                mamba_encoder = Mamba2DecoderLayer(
                    hybrid_config,
                    layer_idx,
                    device="cuda",
                    dtype=dtype,
                )
                if mamba_model is not None and mla_model is not None:
                    mamba_encoder.load_state_dict(mamba_model.model.model.layers._modules[f'{layer_idx}'].state_dict()) 
                    mamba_encoder = mamba_encoder.to(dtype)
                    print(f"layerid: {layer_idx}, SSM initialization from MHA/GQA")
                else:
                    if init_with_kqvo:
                        mamba_encoder.mlp.load_state_dict(transformer_model.model.layers._modules[f'{layer_idx}'].mlp.state_dict())
                        mamba_encoder.input_layernorm.load_state_dict(transformer_model.model.layers._modules[f'{layer_idx}'].input_layernorm.state_dict())
                        mamba_encoder.post_attention_layernorm.load_state_dict(transformer_model.model.layers._modules[f'{layer_idx}'].post_attention_layernorm.state_dict())
                        mamba_encoder.mamba.out_proj.load_state_dict(transformer_model.model.layers._modules[f'{layer_idx}'].self_attn.o_proj.state_dict())

                        # [z, x, B, C, dt]
                        mamba_encoder.mamba.in_proj.weight.data[hybrid_config.d_inner:hybrid_config.d_inner+hybrid_config.d_xb, :].copy_(transformer_model.model.layers._modules[f'{layer_idx}'].self_attn.v_proj.weight.data)
                        mamba_encoder.mamba.in_proj.weight.data[hybrid_config.d_inner+hybrid_config.d_xb:hybrid_config.d_inner+2*hybrid_config.d_xb, :].copy_(transformer_model.model.layers._modules[f'{layer_idx}'].self_attn.k_proj.weight.data)
                        mamba_encoder.mamba.in_proj.weight.data[hybrid_config.d_inner+2*hybrid_config.d_xb:2*hybrid_config.d_inner+2*hybrid_config.d_xb, :].copy_(transformer_model.model.layers._modules[f'{layer_idx}'].self_attn.q_proj.weight.data)
                        print(f"layerid: {layer_idx}, SSM initialization from MHA/GQA")
                    
                        # keep dtype to be the same
                        mamba_encoder.mlp = mamba_encoder.mlp.to(dtype)
                        mamba_encoder.input_layernorm = mamba_encoder.input_layernorm.to(dtype)
                        mamba_encoder.post_attention_layernorm = mamba_encoder.post_attention_layernorm.to(dtype)

                self.model.model.layers[layer_idx] = mamba_encoder

        self.hybrid_config.layer_rank_list = self.layer_rank_list
        
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
        **kwargs,
    ):  
        if mamba_model_path is not None: 
            mamba_model = HybridModelWrapper.from_pretrained(mamba_model_path, torch_dtype=dtype, attn_implementation=attn_implementation)
        else: 
            mamba_model = None

        if mla_model_path is not None: 
            mla_model = HybridModelWrapper.from_pretrained(mla_model_path, torch_dtype=dtype, attn_implementation=attn_implementation)
        else:
            mla_model=None
        transformer_model = CustomLlamaForCausalLM.from_pretrained(tranformer_name, hybrid_config, torch_dtype=dtype)
        return HybridModelWrapper(checkpoint_path, transformer_model, hybrid_config, mla_layers, dtype, init_with_svd, init_with_kqvo, mamba_model, mla_model)

    @staticmethod
    def from_pretrained_local(pretrained_model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"):
        config_data = load_config_hf(pretrained_model_name)
        with open(f'{pretrained_model_name}/{HYBRID_CONFIG_NAME}', 'r') as json_file:
            config_dict = json.load(json_file)
        hybrid_config = HybridConfig(**config_dict)
        transformer_model = CustomLlamaForCausalLM(LlamaConfig(**config_data), hybrid_config, data_dtype=torch_dtype)
        return HybridModelWrapper(pretrained_model_name, transformer_model, hybrid_config, hybrid_config.mla_layers, torch_dtype, init_with_svd=False, init_with_kqvo=False) 

    @staticmethod
    def from_pretrained_hub(pretrained_model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"):
        config_data = load_config_hf(pretrained_model_name)
        resolved_archive_file = cached_file(pretrained_model_name, HYBRID_CONFIG_NAME, _raise_exceptions_for_missing_entries=False)
        config_dict = json.load(open(resolved_archive_file))
        hybrid_config = HybridConfig(**config_dict)
        transformer_model = CustomLlamaForCausalLM(LlamaConfig(**config_data), hybrid_config, data_dtype=torch_dtype)
        return HybridModelWrapper(pretrained_model_name, transformer_model, hybrid_config, hybrid_config.mla_layers, torch_dtype, init_with_svd=False, init_with_kqvo=False, load_from_hub=True) 

    @staticmethod
    def from_pretrained(pretrained_model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"):
        if os.path.exists(pretrained_model_name):
            return HybridModelWrapper.from_pretrained_local(pretrained_model_name, torch_dtype, attn_implementation)
        else:
            return HybridModelWrapper.from_pretrained_hub(pretrained_model_name, torch_dtype, attn_implementation)

    def save_config(self, save_directory, config_file_path):
        os.makedirs(save_directory, exist_ok=True)
        config_path = os.path.join(save_directory, 'hybrid_config.json')
        with open(config_path, 'w') as f:
            config_dict = asdict(self.hybrid_config)
            json.dump(config_dict, f, indent=4)
        os.system("cp {} {}/".format(config_file_path, save_directory))