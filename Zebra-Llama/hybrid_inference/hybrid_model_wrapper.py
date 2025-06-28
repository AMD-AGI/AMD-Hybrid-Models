# Copyright (c) 2023, Albert Gu, Tri Dao.
import os
import json

import torch
import torch.nn as nn

from dataclasses import dataclass, field, asdict

from transformers import AutoModelForCausalLM

from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
from transformers.utils.hub import cached_file

from hybrid_inference.mla.hybrid_model import MLADecoderLayer
from hybrid.mamba2.hybrid_model import Mamba2DecoderLayer
from mamba_ssm.utils.generation import GenerationMixin
from hybrid.hybrid_config import HybridConfig
from collections import namedtuple

from typing import List, Optional, Tuple, Union
from util import load_safetensors_to_dict

HYBRID_CONFIG_NAME = "hybrid_config.json"

def load_config_hf(model_name):
    resolved_archive_file = cached_file(model_name, CONFIG_NAME, _raise_exceptions_for_missing_entries=False)
    return json.load(open(resolved_archive_file))


def load_state_dict_hf(model_name, device=None, dtype=None):
    # If not fp32, then we don't want to load directly to the GPU
    mapped_device = "cpu" if dtype not in [torch.float32, None] else device
    resolved_archive_file = cached_file(model_name, WEIGHTS_NAME, _raise_exceptions_for_missing_entries=False)
    return torch.load(resolved_archive_file, map_location=mapped_device)
    # Convert dtype before moving to GPU to save memory
    if dtype is not None:
        state_dict = {k: v.to(dtype=dtype) for k, v in state_dict.items()}
    state_dict = {k: v.to(device=device) for k, v in state_dict.items()}
    return state_dict


class HybridModelWrapper(nn.Module, GenerationMixin):

    def __init__(self, checkpoint_path, transformer_model, hybrid_config, mla_layers, dtype, absorb=False, load_from_hub=False, **kwargs):
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
                    absorb=absorb,
                    device="cuda",
                    dtype=dtype,
                )
                self.model.model.layers[layer_idx] = MLA_encoder
            
            else:

                # Initialize the Mamba2 layer
                mamba_encoder = Mamba2DecoderLayer(
                    hybrid_config,
                    layer_idx,
                    device="cuda",
                    dtype=dtype,
                )
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


    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.model.model.layers)
        }

    def forward(self, input_ids, position_ids=None, inference_params=None, num_last_tokens=0, **mixer_kwargs):
        """
        "position_ids" is just to be compatible with Transformer generation. We don't use it.
        num_last_tokens: if > 0, only return the logits for the last n tokens
        """
        
        # print(input_ids.shape)
        # print(f"offset: {inference_params.seqlen_offset}")
        hidden_states = self.model.model.embed_tokens(input_ids, **mixer_kwargs)
        for decoder_layer in self.model.model.layers:
            hidden_states = decoder_layer(hidden_states, inference_params=inference_params, **mixer_kwargs)
            hidden_states = hidden_states[0]
        hidden_states = self.model.model.norm(hidden_states)
        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]
        lm_logits = self.model.lm_head(hidden_states)
        CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])
        return CausalLMOutput(logits=lm_logits)

    @staticmethod
    def from_pretrained_local(pretrained_model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", absorb=False):
        config_data = load_config_hf(pretrained_model_name)
        transformer_model = AutoModelForCausalLM.from_pretrained(config_data["_name_or_path"], torch_dtype=torch_dtype, attn_implementation=attn_implementation)
        with open(f'{pretrained_model_name}/{HYBRID_CONFIG_NAME}', 'r') as json_file:
            config_dict = json.load(json_file)
        hybrid_config = HybridConfig(**config_dict)
        return HybridModelWrapper(pretrained_model_name, transformer_model, hybrid_config, hybrid_config.mla_layers, torch_dtype, absorb=absorb) 

    @staticmethod
    def from_pretrained_hub(pretrained_model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", absorb=False):
        config_data = load_config_hf(pretrained_model_name)
        transformer_model = AutoModelForCausalLM.from_pretrained(config_data["_name_or_path"], torch_dtype=torch_dtype, attn_implementation=attn_implementation)
        resolved_archive_file = cached_file(pretrained_model_name, HYBRID_CONFIG_NAME, _raise_exceptions_for_missing_entries=False)
        config_dict = json.load(open(resolved_archive_file))
        hybrid_config = HybridConfig(**config_dict)
        return HybridModelWrapper(pretrained_model_name, transformer_model, hybrid_config, hybrid_config.mla_layers, torch_dtype, absorb=absorb, load_from_hub=True) 

    @staticmethod
    def from_pretrained(pretrained_model_name, torch_dtype=torch.bfloat16, absorb=False, attn_implementation="flash_attention_2"):
        if os.path.exists(pretrained_model_name):
            return HybridModelWrapper.from_pretrained_local(pretrained_model_name, torch_dtype, attn_implementation, absorb)
        else:
            return HybridModelWrapper.from_pretrained_hub(pretrained_model_name, torch_dtype, attn_implementation, absorb)

    def save_config(self, save_directory, config_file_path):
        os.makedirs(save_directory, exist_ok=True)
        config_path = os.path.join(save_directory, 'hybrid_config.json')
        with open(config_path, 'w') as f:
            config_dict = asdict(self.hybrid_config)
            json.dump(config_dict, f, indent=4)
        os.system("cp {} {}/".format(config_file_path, save_directory))

    def generate(
        self,
        input_ids,
        max_length=1024,
        top_k=1,
        top_p=0.0,
        min_p=0.0,
        temperature=1.0,
        return_dict_in_generate=False,
        output_scores=False,
        **kwargs,
    ):

        if kwargs is not None:
            max_new_tokens = kwargs.pop('max_new_tokens', None)
            if max_new_tokens is not None:
                max_length = max_new_tokens + input_ids.shape[1]
            do_sample = kwargs.pop('do_sample', True)
            if not do_sample:
                top_k, top_p, min_p = 1, 0.0, 0.0
            cg = kwargs.pop('cg', False)

            eos_token_id = kwargs.pop('eos_token_id', None)
            if eos_token_id is None:
                eos_token_id = self.config.eos_token_id

            attention_mask = kwargs.pop('attention_mask', None)
            pad_token_id = kwargs.pop('pad_token_id', None)
            no_repeat_ngram_size = kwargs.pop('no_repeat_ngram_size', None)
            length_penalty = kwargs.pop('length_penalty', None)
            num_return_sequences = kwargs.pop('num_return_sequences', None)
            num_beams = kwargs.pop('num_beams', None)
            low_memory = kwargs.pop('low_memory', None)
            stopping_criteria = kwargs.pop('stopping_criteria', None)
        
        return super().generate(
            input_ids=input_ids,
            max_length=max_length,
            cg=cg,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            temperature=temperature,
            return_dict_in_generate=return_dict_in_generate,
            output_scores=output_scores,
            eos_token_id=eos_token_id,
            **kwargs,
        )