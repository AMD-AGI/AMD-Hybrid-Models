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

from typing import List, Optional, Tuple, Union

from util import load_safetensors_to_dict
import logging

logger = logging.getLogger(__name__)

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


class CustomLlamaModel(LlamaModel):
    def __init__(self, config):
        super().__init__(config)
        # Add your custom layers or modules here if needed
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        layer_input: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):  # kept for BC (non `Cache` `past_key_values` inputs)
            return_legacy_cache = True
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            logger.warning_once(
                "We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. "
                "Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)"
            )

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        
        if layer_input is not None:
            output_hidden_states = True

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for idx, decoder_layer in enumerate(self.layers):
            
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            # If performing intermediate layer distillation, replace the layer input
            if layer_input is not None:
                hidden_states = layer_input[idx]

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class CustomLlamaForCausalLM(LlamaForCausalLM):

    def __init__(self, config):
        super().__init__(config)
        self.model = CustomLlamaModel(config)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        layer_input: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            layer_input=layer_input,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


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
            use_cache=False,
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
        transformer_model = CustomLlamaForCausalLM.from_pretrained(tranformer_name, torch_dtype=dtype, attn_implementation=attn_implementation)
        return HybridModelWrapper(checkpoint_path, transformer_model, hybrid_config, mla_layers, dtype, init_with_svd, init_with_kqvo, mamba_model, mla_model)

    @staticmethod
    def from_pretrained_local(pretrained_model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"):
        config_data = load_config_hf(pretrained_model_name)
        transformer_model = AutoModelForCausalLM.from_config(LlamaConfig(**config_data), torch_dtype=torch_dtype, attn_implementation=attn_implementation)
        with open(f'{pretrained_model_name}/{HYBRID_CONFIG_NAME}', 'r') as json_file:
            config_dict = json.load(json_file)
        hybrid_config = HybridConfig(**config_dict)
        return HybridModelWrapper(pretrained_model_name, transformer_model, hybrid_config, hybrid_config.mla_layers, torch_dtype, init_with_svd=False, init_with_kqvo=False) 

    @staticmethod
    def from_pretrained_hub(pretrained_model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"):
        config_data = load_config_hf(pretrained_model_name)
        transformer_model = AutoModelForCausalLM.from_config(LlamaConfig(**config_data), torch_dtype=torch_dtype, attn_implementation=attn_implementation)
        resolved_archive_file = cached_file(pretrained_model_name, HYBRID_CONFIG_NAME, _raise_exceptions_for_missing_entries=False)
        config_dict = json.load(open(resolved_archive_file))
        hybrid_config = HybridConfig(**config_dict)
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