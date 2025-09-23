# Copyright (c) 2023, Albert Gu, Tri Dao.
import os
import json

import torch
import torch.nn as nn

from dataclasses import dataclass, field, asdict

from transformers import AutoModelForCausalLM, LlamaConfig

from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
from transformers.utils.hub import cached_file

from mla_inference.hybrid_model import MLADecoderLayer
from mla_inference.generation import GenerationMixin
from mla.hybrid_mla_config import MLAConfig, DeepseekV3Config
from typing import List, Optional, Tuple, Union
from collections import namedtuple
from util import load_safetensors_to_dict

MLA_CONFIG_NAME = "MLA_config.json"
MLA_LAYER_CONFIG_NAME = "mla_layer_config.json"
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


class MLATransformerHybridModelWrapper(nn.Module, GenerationMixin):

    def __init__(self, checkpoint_path, transformer_model, MLA_config, attn_layers, dtype, init_with_svd, load_from_hub=False, **kwargs):
        super(MLATransformerHybridModelWrapper, self).__init__()
        self.MLA_config = MLA_config
        self.attn_layers = attn_layers 
        self.model = transformer_model
        self.config = self.model.config
        self.layer_rank_list = self.MLA_config.deepseek_cfg.layer_rank_list
        self.dtype=dtype
        self.mla_layers = [i for i in range(MLA_config.n_layer) if i not in attn_layers]

        for layer_idx in range(MLA_config.n_layer):
            if layer_idx not in attn_layers:
                MLA_encoder = MLADecoderLayer(
                    MLA_config,
                    layer_idx,
                    device="cuda",
                    dtype=dtype,
                )
                
                MLA_encoder.mlp.load_state_dict(transformer_model.model.layers._modules[f'{layer_idx}'].mlp.state_dict())
                MLA_encoder.input_layernorm.load_state_dict(transformer_model.model.layers._modules[f'{layer_idx}'].input_layernorm.state_dict())
                MLA_encoder.post_attention_layernorm.load_state_dict(transformer_model.model.layers._modules[f'{layer_idx}'].post_attention_layernorm.state_dict())
                MLA_encoder.mla.out_proj.load_state_dict(transformer_model.model.layers._modules[f'{layer_idx}'].self_attn.o_proj.state_dict())
                
                if not hasattr(self.config, 'head_dim'):
                    self.config.head_dim = self.config.hidden_size // self.config.num_attention_heads
                    
                if init_with_svd:
                    use_dynamic_rank = not layer_idx in [0, MLA_config.n_layer-1] or not(MLA_config.deepseek_cfg.use_fixed_rank_for_first_and_last_block)
                    
                    q_matrix = transformer_model.model.layers._modules[f'{layer_idx}'].self_attn.q_proj.weight.data                   
                    q_rank = MLA_encoder.mla.re_init_q(q_matrix, self.config.head_dim, dtype, use_dynamic_rank)

                    k_matrix = transformer_model.model.layers._modules[f'{layer_idx}'].self_attn.k_proj.weight.data
                    v_matrix = transformer_model.model.layers._modules[f'{layer_idx}'].self_attn.v_proj.weight.data
                    kv_rank = MLA_encoder.mla.re_init_kv(k_matrix, v_matrix, self.config.head_dim, dtype, use_dynamic_rank)
                    
                    print(f"dynamic_rank: {use_dynamic_rank & (MLA_config.deepseek_cfg.q_energy_ratio is not None)}; Layernorm: {MLA_config.deepseek_cfg.use_lora_layer_norm}; layerid: {layer_idx}; q_rank: {q_rank}; kv_rank: {kv_rank}; total_rank: {q_rank+kv_rank}")
                    self.layer_rank_list[str(layer_idx)] = {"q_rank": q_rank, "kv_rank": kv_rank}

                # keep dtype to be the same
                MLA_encoder.mlp = MLA_encoder.mlp.to(dtype)
                MLA_encoder.input_layernorm = MLA_encoder.input_layernorm.to(dtype)
                MLA_encoder.post_attention_layernorm = MLA_encoder.post_attention_layernorm.to(dtype)

                self.model.model.layers[layer_idx] = MLA_encoder
                
        self.MLA_config.deepseek_cfg.layer_rank_list = self.layer_rank_list
        
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

        for layer_idx in range(MLA_config.n_layer):
            if layer_idx not in attn_layers:
                self.model.model.layers[layer_idx]._init_merge()

        self.model = self.model.to(dtype).cuda()
    
    @staticmethod
    def init_distillation(
        checkpoint_path,
        tranformer_name,
        MLA_config,
        attn_layers,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        init_with_svd=True,
        **kwargs,
    ):
        transformer_model = AutoModelForCausalLM.from_pretrained(tranformer_name, torch_dtype=dtype, attn_implementation=attn_implementation)
        return MLATransformerHybridModelWrapper(checkpoint_path, transformer_model, MLA_config, attn_layers, dtype, init_with_svd)

    @staticmethod
    def from_pretrained_local(pretrained_model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"):
        config_data = load_config_hf(pretrained_model_name)
        transformer_model = AutoModelForCausalLM.from_config(LlamaConfig(**config_data), torch_dtype=torch_dtype, attn_implementation=attn_implementation)
        with open(f'{pretrained_model_name}/{MLA_LAYER_CONFIG_NAME}', 'r') as json_file:
            config_dict = json.load(json_file)
        layer_config = DeepseekV3Config(**config_dict)
        with open(f'{pretrained_model_name}/{MLA_CONFIG_NAME}', 'r') as json_file:
            config_dict = json.load(json_file)
        MLA_config = MLAConfig(deepseek_cfg=layer_config, **config_dict)
        return MLATransformerHybridModelWrapper(pretrained_model_name, transformer_model, MLA_config, MLA_config.attn_layers, torch_dtype, init_with_svd=False) 

    @staticmethod
    def from_pretrained_hub(pretrained_model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"):
        config_data = load_config_hf(pretrained_model_name)
        transformer_model = AutoModelForCausalLM.from_config(LlamaConfig(**config_data), torch_dtype=torch_dtype, attn_implementation=attn_implementation)
        resolved_archive_file = cached_file(pretrained_model_name, MLA_LAYER_CONFIG_NAME, _raise_exceptions_for_missing_entries=False)
        config_dict = json.load(open(resolved_archive_file))
        layer_config = DeepseekV3Config(**config_dict)
        resolved_archive_file = cached_file(pretrained_model_name, MLA_CONFIG_NAME, _raise_exceptions_for_missing_entries=False)
        config_dict = json.load(open(resolved_archive_file))
        MLA_config = MLAConfig(deepseek_cfg=layer_config, **config_dict)
        return MLATransformerHybridModelWrapper(pretrained_model_name, transformer_model, MLA_config, MLA_config.attn_layers, torch_dtype, init_with_svd=False, load_from_hub=True) 

    @staticmethod
    def from_pretrained(pretrained_model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"):
        if os.path.exists(pretrained_model_name):
            return MLATransformerHybridModelWrapper.from_pretrained_local(pretrained_model_name, torch_dtype, attn_implementation)
        else:
            return MLATransformerHybridModelWrapper.from_pretrained_hub(pretrained_model_name, torch_dtype, attn_implementation)

    def save_config(self, save_directory, config_file_path):
        os.makedirs(save_directory, exist_ok=True)
        config_path = os.path.join(save_directory, 'MLA_config.json')
        with open(config_path, 'w') as f:
            config_dict = asdict(self.MLA_config)
            if "deepseek_cfg" in config_dict:
                del config_dict["deepseek_cfg"]
            json.dump(config_dict, f, indent=4)

        config_path = os.path.join(save_directory, 'mla_layer_config.json')
        self.MLA_config.deepseek_cfg.to_json_file(config_path, use_diff=True)
        os.system("cp {} {}/".format(config_file_path, save_directory))

    def freeze_non_mla_params(self, ):
        for name, param in self.model.named_parameters():
            # if 'mla' in name and 'out_proj' not in name:
            if 'mla' in name:
                print(f"Unfreeze {name}")
                param.requires_grad = True
            else:
                param.requires_grad = False
                print(f"Freeze {name}")

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
        layer_splits = [-1] + self.mla_layers + [self.MLA_config.n_layer]
        
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
                    'x': torch.randn((inference_params.max_batch_size, 1, self.config.num_attention_heads, self.MLA_config.deepseek_cfg.kv_lora_rank), device="cuda", dtype=self.dtype),
                    'residual': torch.randn((inference_params.max_batch_size, 1, self.config.hidden_size), device="cuda", dtype=self.dtype),
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



