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
from hybrid_inference.mamba2.hybrid_model import Mamba2DecoderLayer
from hybrid_inference.generation import GenerationMixin
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

    def __init__(self, checkpoint_path, transformer_model, hybrid_config, mla_layers, dtype, absorb=True, load_from_hub=False, **kwargs):
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
                    absorb=True,
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
        
        for layer_idx in range(hybrid_config.n_layer):
            if layer_idx in mla_layers:
                self.model.model.layers[layer_idx]._init_merge()
        self.model = self.model.to(dtype).cuda()


    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.model.model.layers)
        }
    
    def graph_block(self, inference_params):
        
        functions = []
        for idx, l in enumerate(self.mla_layers):
            if idx == 0:
                setattr(self, f"input_{idx}", torch.randint(0, 100, (inference_params.max_batch_size, 1), device="cuda", dtype=torch.long))
                def block(input_ids, inference_params):                    
                    hidden_states = self.model.model.embed_tokens(input_ids)
                    for i in range(self.mla_layers[0]):
                        hidden_states = self.model.model.layers[i](hidden_states, inference_params=inference_params)
                    return self.model.model.layers[self.mla_layers[0]].forward_static_1(hidden_states, inference_params=inference_params)
                functions.append(block)
            else:
                setattr(self, f"input_{idx}", torch.randn((inference_params.max_batch_size, self.hybrid_config.num_attention_heads, 1, self.hybrid_config.kv_lora_rank), device="cuda", dtype=self.dtype))
                setattr(self, f"residual_{idx}", torch.randn((inference_params.max_batch_size, 1, self.hybrid_config.hidden_size), device="cuda", dtype=self.dtype))
                def block(hidden_states, residual, idx, inference_params):
                    hidden_states = self.model.model.layers[self.mla_layers[idx-1]].forward_static_2(hidden_states, residual)
                    for i in range(self.mla_layers[idx-1]+1, self.mla_layers[idx]):
                        hidden_states = self.model.model.layers[i](hidden_states, inference_params=inference_params)
                    return self.model.model.layers[self.mla_layers[idx]].forward_static_1(hidden_states, inference_params=inference_params)
                functions.append(block)
        
        setattr(self, f"input_{len(self.mla_layers)}", torch.randn((inference_params.max_batch_size, self.hybrid_config.num_attention_heads, 1, self.hybrid_config.kv_lora_rank), device="cuda", dtype=self.dtype))
        setattr(self, f"residual_{len(self.mla_layers)}", torch.randn((inference_params.max_batch_size, 1, self.hybrid_config.hidden_size), device="cuda", dtype=self.dtype))

        def block(hidden_states, residual, inference_params):
            hidden_states = self.model.model.layers[self.mla_layers[-1]].forward_static_2(hidden_states, residual)
            for i in range(self.mla_layers[-1]+1, self.hybrid_config.n_layer):
                hidden_states = self.model.model.layers[i](hidden_states, inference_params=inference_params)
            hidden_states = self.model.model.norm(hidden_states)
            lm_logits = self.model.lm_head(hidden_states)
            CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])
            return CausalLMOutput(logits=lm_logits)
        functions.append(block)
 
        return functions


    def capture_graph(self, inference_params, **kwargs):
        # for i, layer in enumerate(self.model.model.layers):
        #     layer.capture_graph(inference_params)
        # self.hybrid_config.n_layer
        # count = 0
        # for idx, l in enumerate(self.mla_layers):
        #     if idx == 0:
        #         setattr(self, f"input_{idx}", torch.randn(inference_params.max_batch_size, 1))
        self.static_functions = self.graph_block(inference_params)

        self.graphed_segments = []
        for idx, l in enumerate(self.mla_layers):
            if idx == 0:
                static_inputs = {
                    'input_ids': torch.randint(0, 100, (inference_params.max_batch_size, 1), device="cuda", dtype=torch.long)
                }
                # Output placeholder (tuple to match the return of DummyLayer)
                static_outputs = {
                    # 'q_nope': torch.randn((inference_params.max_batch_size, 1, self.hybrid_config.num_attention_heads, self.hybrid_config.qk_nope_head_dim), device="cuda", dtype=self.dtype),
                    # 'q_pe': torch.randn((inference_params.max_batch_size, 1, self.hybrid_config.num_attention_heads, self.hybrid_config.qk_rope_head_dim), device="cuda", dtype=self.dtype),
                    # 'k_pe': torch.randn((inference_params.max_batch_size, 1, self.hybrid_config.num_attention_heads, self.hybrid_config.qk_rope_head_dim), device="cuda", dtype=self.dtype),
                    # 'cos': torch.randn((inference_params.max_batch_size, 1, self.hybrid_config.num_attention_heads, self.hybrid_config.qk_rope_head_dim), device="cuda", dtype=self.dtype),
                    # 'sin': torch.randn((inference_params.max_batch_size, 1, self.hybrid_config.num_attention_heads, self.hybrid_config.qk_rope_head_dim), device="cuda", dtype=self.dtype),
                    # 'compressed_kv_comb': torch.randn((inference_params.max_batch_size, 1, self.hybrid_config.kv_lora_rank+self.hybrid_config.qk_rope_head_dim), device="cuda", dtype=self.dtype),
                    # 'residual': torch.randn((inference_params.max_batch_size, 1, self.hybrid_config.hidden_size), device="cuda", dtype=self.dtype),
                }
                
                # q_nope, q_pe, k_pe, cos, sin, compressed_kv

                def _segment_forward_pass_graphed():
                    hidden_states = self.model.model.embed_tokens(static_inputs['input_ids'])
                    for i in range(self.mla_layers[0]):
                        hidden_states = self.model.model.layers[i](hidden_states, inference_params=inference_params)
                    (
                        static_outputs['q_nope'], 
                        static_outputs['q_pe'], 
                        static_outputs['k_pe'], 
                        static_outputs['cos'], 
                        static_outputs['sin'], 
                        static_outputs['compressed_kv_comb'], 
                        static_outputs['residual']
                    ) = self.model.model.layers[self.mla_layers[0]].forward_static_1(hidden_states, inference_params=inference_params)
                    # static_outputs['q_nope'].copy_(q_nope)
                    # static_outputs['q_pe'].copy_(q_pe)
                    # static_outputs['compressed_kv_comb'].copy_(compressed_kv_comb)
                    # static_outputs['residual'].copy_(residual)
                
                s = torch.cuda.Stream()    
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(2):
                        _segment_forward_pass_graphed()
                s.synchronize()

                if torch.distributed.is_initialized():
                    torch.distributed.barrier()
                
                torch.cuda.current_stream().wait_stream(s)

                segment_graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(segment_graph, pool=torch.cuda.graphs.graph_pool_handle()):
                    _segment_forward_pass_graphed()
                
                self.graphed_segments.append({
                    "graph": segment_graph,
                    "static_inputs": static_inputs,
                    "static_outputs": static_outputs,
                })
            else:
                static_inputs = {
                    'x': torch.randn((inference_params.max_batch_size, 1, self.hybrid_config.num_attention_heads, self.hybrid_config.kv_lora_rank), device="cuda", dtype=self.dtype),
                    'residual': torch.randn((inference_params.max_batch_size, 1, self.hybrid_config.hidden_size), device="cuda", dtype=self.dtype),
                }
                # Output placeholder (tuple to match the return of DummyLayer)
                static_outputs = {
                    # 'q_nope': torch.randn((inference_params.max_batch_size, self.hybrid_config.num_attention_heads, 1, self.hybrid_config.qk_nope_head_dim), device="cuda", dtype=self.dtype),
                    # 'q_pe': torch.randn((inference_params.max_batch_size, self.hybrid_config.num_attention_heads, 1, self.hybrid_config.qk_rope_head_dim), device="cuda", dtype=self.dtype),
                    # 'compressed_kv_comb': torch.randn((inference_params.max_batch_size, 1, self.hybrid_config.kv_lora_rank+self.hybrid_config.qk_rope_head_dim), device="cuda", dtype=self.dtype),
                    # 'residual': torch.randn((inference_params.max_batch_size, 1, self.hybrid_config.hidden_size), device="cuda", dtype=self.dtype),
                }
                def _segment_forward_pass_graphed():
                    x = static_inputs['x']
                    hidden_states = self.model.model.layers[self.mla_layers[idx-1]].forward_static_2(static_inputs['x'], static_inputs['residual'])

                    for i in range(self.mla_layers[idx-1]+1, self.mla_layers[idx]):
                        hidden_states = self.model.model.layers[i](hidden_states, inference_params=inference_params)
                    (
                        static_outputs['q_nope'], 
                        static_outputs['q_pe'], 
                        static_outputs['k_pe'], 
                        static_outputs['cos'], 
                        static_outputs['sin'], 
                        static_outputs['compressed_kv_comb'], 
                        static_outputs['residual']
                    ) = self.model.model.layers[self.mla_layers[idx]].forward_static_1(hidden_states, inference_params=inference_params)
                    # static_outputs['q_nope'].copy_(q_nope)
                    # static_outputs['q_pe'].copy_(q_pe)
                    # static_outputs['compressed_kv_comb'].copy_(compressed_kv_comb)
                    # static_outputs['residual'].copy_(residual)
                
                s = torch.cuda.Stream()    
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(2):
                        _segment_forward_pass_graphed()
                s.synchronize()

                if torch.distributed.is_initialized():
                    torch.distributed.barrier()
                
                torch.cuda.current_stream().wait_stream(s)

                segment_graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(segment_graph, pool=torch.cuda.graphs.graph_pool_handle()):
                    _segment_forward_pass_graphed()
                
                self.graphed_segments.append({
                    "graph": segment_graph,
                    "static_inputs": static_inputs,
                    "static_outputs": static_outputs,
                })
        
        static_inputs = {
            'x': torch.randn((inference_params.max_batch_size, 1, self.hybrid_config.num_attention_heads, self.hybrid_config.kv_lora_rank), device="cuda", dtype=self.dtype),
            'residual': torch.randn((inference_params.max_batch_size, 1, self.hybrid_config.hidden_size), device="cuda", dtype=self.dtype),
        }
        # Output placeholder (tuple to match the return of DummyLayer)
        static_outputs = {
        }
        def _segment_forward_pass_graphed():
            hidden_states = self.model.model.layers[self.mla_layers[-1]].forward_static_2(static_inputs['x'], static_inputs['residual'])
            for i in range(self.mla_layers[-1]+1, self.hybrid_config.n_layer):
                hidden_states = self.model.model.layers[i](hidden_states, inference_params=inference_params)
            hidden_states = self.model.model.norm(hidden_states)
            lm_logits = self.model.lm_head(hidden_states)
            CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])
            static_outputs['outputs_logits'] = CausalLMOutput(logits=lm_logits)
        
        s = torch.cuda.Stream()    
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                _segment_forward_pass_graphed()
        s.synchronize()

        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        
        torch.cuda.current_stream().wait_stream(s)

        segment_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(segment_graph, pool=torch.cuda.graphs.graph_pool_handle()):
            _segment_forward_pass_graphed()
        
        self.graphed_segments.append({
            "graph": segment_graph,
            "static_inputs": static_inputs,
            "static_outputs": static_outputs,
        })



        # self.input_0 = torch.randint(0, 100, (inference_params.max_batch_size, 1), device="cuda", dtype=torch.long)


        # s = torch.cuda.Stream()
        # s.wait_stream(torch.cuda.current_stream())
        # with torch.cuda.stream(s):
        #     for _ in range(2):
        #         self.q_nope, self.q_pe, self.compressed_kv_comb, self.residual_1 = self.static_functions[0](self.input_0, inference_params)
        # s.synchronize()

        # if torch.distributed.is_initialized():
        #     torch.distributed.barrier()
        
        # torch.cuda.current_stream().wait_stream(s)
        # graph = torch.cuda.CUDAGraph()
        # with torch.cuda.graph(graph, pool=torch.cuda.graphs.graph_pool_handle()):
        #     self.q_nope, self.q_pe, self.compressed_kv_comb, self.residual_1 = self.static_functions[0](self.input_0, inference_params)
        # self.graph = graph

        # for idx, func in enumerate(self.static_functions):
        #     if idx >= 2:
        #         break
        #     # Captures the graph
        #     # To allow capture, automatically sets a side stream as the current stream in the context
        #     if idx == len(self.static_functions)-1:
        #         s = torch.cuda.Stream()
        #         s.wait_stream(torch.cuda.current_stream())
        #         with torch.cuda.stream(s):
        #             for _ in range(2):
        #                 self.outputs_logits = func(self.input_1, self.residual_1, inference_params)
        #         s.synchronize()
        #         if torch.distributed.is_initialized():
        #             torch.distributed.barrier()
        #         torch.cuda.current_stream().wait_stream(s)
        #         graph = torch.cuda.CUDAGraph()
        #         with torch.cuda.graph(graph, pool=torch.cuda.graphs.graph_pool_handle()):
        #             self.outputs_logits = func(self.input_1, self.residual_1, inference_params)
        #         self.graph.append(graph) 
        #     elif idx == 0: 
        #         s = torch.cuda.Stream()
        #         s.wait_stream(torch.cuda.current_stream())
        #         with torch.cuda.stream(s): 
        #             for _ in range(2):
        #                 self.q_nope, self.q_pe, self.compressed_kv_comb, self.residual_1 = func(self.input_0, inference_params)
        #         s.synchronize()
        #         if torch.distributed.is_initialized():
        #             torch.distributed.barrier() 
        #         torch.cuda.current_stream().wait_stream(s)             
        #         graph0 = torch.cuda.CUDAGraph()
        #         with torch.cuda.graph(graph0, pool=torch.cuda.graphs.graph_pool_handle()):
        #             self.q_nope, self.q_pe, self.compressed_kv_comb, self.residual_1 = func(self.input_0, inference_params)
        #         self.graph.append(graph0)
        #     else:
        #         s = torch.cuda.Stream()
        #         s.wait_stream(torch.cuda.current_stream())
        #         for _ in range(2):
        #             self.q_nope1, self.q_pe1, self.compressed_kv_comb1, self.residual_2 = func(self.input_1, self.residual_1, self.idx, inference_params)
        #         s.synchronize()
        #         if torch.distributed.is_initialized():
        #             torch.distributed.barrier() 
        #         torch.cuda.current_stream().wait_stream(s)
        #         graph1 = torch.cuda.CUDAGraph()
        #         with torch.cuda.graph(graph1, pool=torch.cuda.graphs.graph_pool_handle()):
        #             self.q_nope1, self.q_pe1, self.compressed_kv_comb1, self.residual_2 = func(self.input_1, self.residual_1, self.idx, inference_params)
        #         self.graph1 = graph1
        #         self.graph.append(graph1)


    
    def forward_graph(self, input_ids, position_ids=None, inference_params=None, num_last_tokens=0, **mixer_kwargs): 
                
        # hidden_states = self.model.model.embed_tokens(input_ids)
        # for decoder_layer in self.model.model.layers:
        #     hidden_states = decoder_layer.forward_graph(hidden_states, inference_params=inference_params, position_ids=position_ids)
        #     # hidden_states = hidden_states[0]
        # hidden_states = self.model.model.norm(hidden_states)
        # if num_last_tokens > 0:
        #     hidden_states = hidden_states[:, -num_last_tokens:]
        # lm_logits = self.model.lm_head(hidden_states)
        # CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])
        # return CausalLMOutput(logits=lm_logits)
        

        for idx, segment in enumerate(self.graphed_segments):
            if idx == 0:
                segment["static_inputs"]['input_ids'].copy_(input_ids)
                segment["graph"].replay()
                residual = segment['static_outputs']['residual'].clone()  
                x = self.model.model.layers[self.mla_layers[idx]].mla.forward_dynamic(
                    segment['static_outputs']['q_nope'], 
                    segment['static_outputs']['q_pe'], 
                    segment['static_outputs']['k_pe'], 
                    segment['static_outputs']['cos'], 
                    segment['static_outputs']['sin'],
                    segment['static_outputs']['compressed_kv_comb'], inference_params=inference_params)
                # q_nope, q_pe, compressed_kv_comb, residual = self.static_functions[idx](input_ids, inference_params=inference_params)         
                # x = self.model.model.layers[self.mla_layers[idx]].mla.forward_dynamic(q_nope, q_pe, compressed_kv_comb, inference_params=inference_params)
            elif idx == len(self.graphed_segments)-1:
                segment["static_inputs"]['x'].copy_(x)
                segment["static_inputs"]['residual'].copy_(residual)
                segment["graph"].replay()
                return segment["static_outputs"]['outputs_logits']
                # outputs_logits = self.static_functions[idx](x, residual, inference_params=inference_params)
                # return outputs_logits
            else:
                segment["static_inputs"]['x'].copy_(x)
                segment["static_inputs"]['residual'].copy_(residual)
                segment["graph"].replay()
                residual = segment['static_outputs']['residual'].clone() 
                x = self.model.model.layers[self.mla_layers[idx]].mla.forward_dynamic(
                    segment['static_outputs']['q_nope'], 
                    segment['static_outputs']['q_pe'], 
                    segment['static_outputs']['k_pe'], 
                    segment['static_outputs']['cos'], 
                    segment['static_outputs']['sin'],
                    segment['static_outputs']['compressed_kv_comb'], inference_params=inference_params)
                # q_nope, q_pe, k_pe, cos, sin, compressed_kv_comb, residual = self.static_functions[idx](x, residual, idx, inference_params=inference_params)
                # x = self.model.model.layers[self.mla_layers[idx]].mla.forward_dynamic(q_nope, q_pe, k_pe, cos, sin, compressed_kv_comb, inference_params=inference_params)


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
            cg = kwargs.pop('cg', True)
            cg_piecewise = kwargs.pop('cg_piecewise', True)
            profile = kwargs.pop('profile', False)
            random_context = kwargs.pop('random_context', False)
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
            cg_piecewise=cg_piecewise,
            profile=profile,
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