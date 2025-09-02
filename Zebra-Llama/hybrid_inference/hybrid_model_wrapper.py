# Copyright (c) 2023, Albert Gu, Tri Dao.
import os
import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple, Union
from collections import namedtuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, LlamaConfig
from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
from transformers.utils.hub import cached_file

from hybrid_inference.mla.hybrid_model import MLADecoderLayer
from hybrid_inference.mamba2.hybrid_model import Mamba2DecoderLayer
from hybrid_inference.generation import GenerationMixin
from hybrid.hybrid_config import HybridConfig
from hybrid.hybrid_modeling import CustomLlamaForCausalLM
from util import load_safetensors_to_dict

HYBRID_CONFIG_NAME = "hybrid_config.json"
CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])

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

class HybridModelWrapper(nn.Module, GenerationMixin):

    def __init__(self, checkpoint_path, transformer_model, hybrid_config, mla_layers, dtype, absorb=True, load_from_hub=False, **kwargs):
        super(HybridModelWrapper, self).__init__()
                
        self.hybrid_config = hybrid_config
        self.mla_layers = mla_layers
        self.model = transformer_model
        self.config = self.model.config
        self.dtype = dtype
        
        if not hasattr(self.config, 'head_dim'):
            self.config.head_dim = self.config.hidden_size // self.config.num_attention_heads
        
        self._initialize_layers()
        self._load_checkpoint(checkpoint_path, load_from_hub, dtype)
        
        for layer_idx in range(hybrid_config.n_layer):
            if layer_idx in mla_layers:
                self.model.model.layers[layer_idx]._init_merge()
        self.model = self.model.to(dtype).cuda()
    
    def _load_checkpoint(self, checkpoint_path, load_from_hub, dtype):
        """Loads model weights from a local or hub checkpoint."""
        if checkpoint_path is None:
            return

        if load_from_hub:
            state_dict = load_state_dict_hf(checkpoint_path, dtype=dtype)
            self.model.load_state_dict(state_dict)
        else:
            if os.path.exists(os.path.join(checkpoint_path, "pytorch_model.bin")):
                state_dict = torch.load(os.path.join(checkpoint_path, "pytorch_model.bin"), map_location="cpu")
                self.model.load_state_dict(state_dict)
            elif os.path.exists(os.path.join(checkpoint_path, "model.safetensors")):
                state_dict = load_safetensors_to_dict(checkpoint_path)
                self.model.load_state_dict(state_dict, strict=False)
            else:
                raise FileNotFoundError(f"No valid checkpoint found at {checkpoint_path}")

    def _initialize_layers(self):
        """Initializes MLA and Mamba2 layers in the model."""
        for layer_idx in range(self.hybrid_config.n_layer):
            is_mla_layer = layer_idx in self.mla_layers
            if is_mla_layer:
                layer = MLADecoderLayer(self.hybrid_config, layer_idx, absorb=True, device="cuda", dtype=self.dtype)
            else:
                layer = Mamba2DecoderLayer(self.hybrid_config, layer_idx, device="cuda", dtype=self.dtype)
            self.model.model.layers[layer_idx] = layer


    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        """Allocates inference cache for each layer."""
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.model.model.layers)
        }
    
    def capture_graph(self, inference_params, **kwargs):
        """
        Captures CUDA graphs for model segments.
        This function has been refactored to be more concise and readable.
        """
        self.graphed_segments = []
        layer_splits = [-1] + self.mla_layers + [self.hybrid_config.n_layer]

        for idx in range(len(layer_splits) - 1):
            start_layer = layer_splits[idx]
            end_layer = layer_splits[idx+1]
            is_first_segment = (idx == 0)
            is_last_segment = (idx == len(layer_splits) - 2)

            # Define inputs for the segment
            if is_first_segment:
                static_inputs = {'input_ids': torch.randint(0, 100, (inference_params.max_batch_size, 1), device="cuda", dtype=torch.long)}
            else:
                static_inputs = {
                    'x': torch.randn((inference_params.max_batch_size, 1, self.hybrid_config.num_attention_heads, self.hybrid_config.kv_lora_rank), device="cuda", dtype=self.dtype),
                    'residual': torch.randn((inference_params.max_batch_size, 1, self.hybrid_config.hidden_size), device="cuda", dtype=self.dtype),
                }
            
            static_outputs = {}

            # Define the forward pass for this segment
            def segment_forward_pass_graphed():
                if is_first_segment:
                    hidden_states = self.model.model.embed_tokens(static_inputs['input_ids'])
                    current_layers = range(start_layer+1, end_layer)
                else:
                    hidden_states = self.model.model.layers[start_layer].forward_static_2(static_inputs['x'], static_inputs['residual'])
                    current_layers = range(start_layer+1, end_layer)

                for i in current_layers:
                    hidden_states = self.model.model.layers[i](hidden_states, inference_params=inference_params)

                if is_last_segment:
                    hidden_states = self.model.model.norm(hidden_states)
                    lm_logits = self.model.lm_head(hidden_states)
                    static_outputs['outputs_logits'] = CausalLMOutput(logits=lm_logits)
                else:
                    (
                        static_outputs['q_nope'], 
                        static_outputs['q_pe'], 
                        static_outputs['k_pe'], 
                        static_outputs['cos'], 
                        static_outputs['sin'], 
                        static_outputs['compressed_kv_comb'], 
                        static_outputs['residual']
                    ) = self.model.model.layers[end_layer].forward_static_1(hidden_states, inference_params=inference_params)
            

            # Capture the graph
            s = torch.cuda.Stream()    
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(2):
                    segment_forward_pass_graphed()
            s.synchronize()

            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            
            torch.cuda.current_stream().wait_stream(s)

            segment_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(segment_graph, pool=torch.cuda.graphs.graph_pool_handle()):
                segment_forward_pass_graphed()
            
            self.graphed_segments.append({
                "graph": segment_graph,
                "static_inputs": static_inputs,
                "static_outputs": static_outputs,
            })
            
    def forward_graph(self, input_ids, position_ids=None, inference_params=None, num_last_tokens=0, **mixer_kwargs): 
        """
        Executes the model using the captured CUDA graphs.
        This function has been streamlined.
        """
        for idx, segment in enumerate(self.graphed_segments):
            if idx == 0: # This attribute needs to be set in capture_graph
                segment['static_inputs']['input_ids'].copy_(input_ids)
            else:
                segment['static_inputs']['x'].copy_(x)
                segment['static_inputs']['residual'].copy_(residual)
            
            # Replay the graph
            segment["graph"].replay()
            
            # Process outputs and prepare for next segment
            # Note: This logic assumes the graph's forward pass returns a tuple
            if idx == len(self.graphed_segments)-1:
                return segment['static_outputs']['outputs_logits'] # This attribute needs to be populated

            outputs = segment['static_outputs']
            residual = outputs['residual'].clone()
            
            x = self.model.model.layers[self.mla_layers[idx]].forward_dynamic(
                outputs['q_nope'], outputs['q_pe'], outputs['k_pe'], outputs['cos'],
                outputs['sin'], outputs['compressed_kv_comb'], inference_params=inference_params
            )

    def forward(self, input_ids, position_ids=None, inference_params=None, num_last_tokens=0, **mixer_kwargs):
        """
        "position_ids" is just to be compatible with Transformer generation. We don't use it.
        num_last_tokens: if > 0, only return the logits for the last n tokens
        """
        # print(input_ids.shape)
        # print(f"offset: {inference_params.seqlen_offset}")
        hidden_states = self.model.model.embed_tokens(input_ids)
        for decoder_layer in self.model.model.layers:
            hidden_states = decoder_layer(hidden_states, inference_params=inference_params, position_ids=position_ids,)
            # hidden_states = hidden_states[0]
        hidden_states = self.model.model.norm(hidden_states)
        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]
        lm_logits = self.model.lm_head(hidden_states)
        CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])
        return CausalLMOutput(logits=lm_logits)
        
    @staticmethod
    def from_pretrained_local(pretrained_model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", absorb=False):
        config_data = load_config_hf(pretrained_model_name)
        with open(f'{pretrained_model_name}/{HYBRID_CONFIG_NAME}', 'r') as json_file:
            config_dict = json.load(json_file)
        hybrid_config = HybridConfig(**config_dict)
        transformer_model = CustomLlamaForCausalLM(LlamaConfig(**config_data), hybrid_config, data_dtype=torch_dtype)
        return HybridModelWrapper(pretrained_model_name, transformer_model, hybrid_config, hybrid_config.mla_layers, torch_dtype, absorb=absorb) 

    @staticmethod
    def from_pretrained_hub(pretrained_model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", absorb=False):
        config_data = load_config_hf(pretrained_model_name)
        resolved_archive_file = cached_file(pretrained_model_name, HYBRID_CONFIG_NAME, _raise_exceptions_for_missing_entries=False)
        config_dict = json.load(open(resolved_archive_file))
        hybrid_config = HybridConfig(**config_dict)
        transformer_model = CustomLlamaForCausalLM(LlamaConfig(**config_data), hybrid_config, data_dtype=torch_dtype)
        return HybridModelWrapper(pretrained_model_name, transformer_model, hybrid_config, hybrid_config.mla_layers, torch_dtype, absorb=absorb, load_from_hub=True)

    @staticmethod
    def from_pretrained(pretrained_model_name, torch_dtype=torch.bfloat16, absorb=False, attn_implementation="flash_attention_2"):
        if os.path.exists(pretrained_model_name):
            return HybridModelWrapper.from_pretrained_local(pretrained_model_name, torch_dtype, attn_implementation, absorb)
        else:
            return HybridModelWrapper.from_pretrained_hub(pretrained_model_name, torch_dtype, attn_implementation, absorb)

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
            cg = kwargs.pop('cg', True)
            random_context = kwargs.pop('random_context', False)
            eos_token_id = kwargs.pop('eos_token_id', None)
            if eos_token_id is None:
                eos_token_id = self.config.eos_token_id
        
        return super().generate(
            input_ids=input_ids,
            max_length=max_length,
            cg=cg,
            random_context=random_context,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            temperature=temperature,
            return_dict_in_generate=return_dict_in_generate,
            output_scores=output_scores,
            eos_token_id=eos_token_id,
            **kwargs,
        )