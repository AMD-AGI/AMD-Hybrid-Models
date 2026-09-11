# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

import os
import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
    LlamaModel,
)

from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaMLP,
    LlamaRMSNorm,
    LlamaAttention,
)

from transformers.utils import (
    WEIGHTS_NAME,
    CONFIG_NAME,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.generation import GenerationMixin
from transformers.utils.hub import cached_file

from transformers.cache_utils import (
    Cache,
    DynamicCache,
)

from util import load_safetensors_to_dict

# from fla.modules import FusedLinearCrossEntropyLoss
from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

try:
    from fla.modules import FusedKLDivLoss
    _FLA_FUSED_KL_AVAILABLE = True
except ImportError:
    _FLA_FUSED_KL_AVAILABLE = False

# Chunked lm_head + cross_entropy: processes logits in small chunks so the full
# (seq_len, vocab_size) tensor is never materialised.  At chunk_size=4096 the
# peak logits allocation is only ~1 GB regardless of total sequence length.
_CE_IGNORE_INDEX = -100

def _chunked_lm_head_ce(weight, bias, hidden_states_flat, shift_labels, chunk_size=4096):
    total_loss = torch.zeros(1, device=hidden_states_flat.device, dtype=torch.float32)
    total_tokens = 0
    for i in range(0, hidden_states_flat.shape[0], chunk_size):
        h_chunk = hidden_states_flat[i:i + chunk_size]
        l_chunk = shift_labels[i:i + chunk_size]
        logits_chunk = F.linear(h_chunk, weight, bias).float()
        n = (l_chunk != _CE_IGNORE_INDEX).sum()
        if n > 0:
            total_loss += F.cross_entropy(
                logits_chunk, l_chunk, ignore_index=_CE_IGNORE_INDEX, reduction="sum"
            )
            total_tokens += n
        del logits_chunk
    return total_loss / total_tokens


# Copied from https://github.com/huggingface/transformers/blob/main/src/transformers/models/jamba/modeling_jamba.py
class HybridDynamicCache(DynamicCache):
    """
    A dynamic cache that can handle both the attention cache (which has a seq_len dimension) and the mamba cache
    (which has a constant shape regardless of seq_len).
    This cache has two sets of lists of tensors: `key_cache` and `value_cache` for attention cache and `conv_states`
    and `ssm_states` for mamba cache. Each of these lists has `num_layers` tensors. The expected shape for each tensor
    For attention layers, `key_cache` and `value_cache` have a shape of `(batch_size, num_heads, seq_len, head_dim)`,
    while `conv_states` and `ssm_states` have a shape of `(batch_size, 0)` (empty tensors).
    For mamba layers, `key_cache` and `value_cache` have a shape of `(batch_size, 0)` (empty tensors),
    while `conv_states` represents the convolution state and has a shape of `(batch_size, d_inner, d_conv)`,
    and `ssm_states` represents the ssm state and has a shape of `(batch_size, d_inner, d_state)`.
    """

    def __init__(self, config, batch_size, dtype=torch.float16, device=None):
        super().__init__()
        self.dtype = dtype
        self.mla_layers = config.mla_layers
        linear_type = getattr(config, 'linear_type', 'mamba')
        has_mamba = linear_type != 'gdn' and config.ssm_cfg

        if has_mamba:
            d_conv = 4
            if config.ssm_cfg["repeat_kv_before_conv"]:
                conv_dim = config.d_inner + config.d_inner + config.d_inner
            else:
                conv_dim = config.d_inner + config.d_xb + config.d_xb
            nheads = config.ssm_cfg["ngroups"]
            headdim = config.ssm_cfg["d_state"]
            d_state = config.ssm_cfg["d_state"]

        self.conv_states = []
        self.ssm_states = []
        self.transformer_layers = []
        for i in range(config.n_layer):
            if i not in self.mla_layers:
                if has_mamba:
                    self.conv_states += [
                        torch.zeros(batch_size, d_conv, conv_dim, device=device, dtype=dtype).transpose(1, 2)
                    ]
                    self.ssm_states += [
                        torch.zeros(batch_size, nheads, headdim, d_state, device=device, dtype=dtype)
                    ]
                else:
                    self.conv_states += [torch.tensor([[]] * batch_size, device=device)]
                    self.ssm_states += [torch.tensor([[]] * batch_size, device=device)]
            else:
                # Attention or MLA layer
                self.conv_states += [torch.tensor([[]] * batch_size, device=device)]
                self.ssm_states += [torch.tensor([[]] * batch_size, device=device)]
                self.transformer_layers.append(i)

        self.key_cache = [torch.tensor([[]] * batch_size, device=device) for _ in range(config.n_layer)]
        self.value_cache = [torch.tensor([[]] * batch_size, device=device) for _ in range(config.n_layer)]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Update the cache
        if self.key_cache[layer_idx].shape[-1] == 0:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        for layer_idx in range(len(self.key_cache)):
            device = self.key_cache[layer_idx].device
            self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, beam_idx.to(device))
            device = self.value_cache[layer_idx].device
            self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, beam_idx.to(device))

            device = self.conv_states[layer_idx].device
            self.conv_states[layer_idx] = self.conv_states[layer_idx].index_select(0, beam_idx.to(device))
            device = self.ssm_states[layer_idx].device
            self.ssm_states[layer_idx] = self.ssm_states[layer_idx].index_select(0, beam_idx.to(device))

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # take any layer that contains cache and not empty tensor
        layer_idx = self.transformer_layers[0] if layer_idx not in self.transformer_layers else layer_idx
        if len(self.key_cache) <= layer_idx:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        raise NotImplementedError("HybridMambaAttentionDynamicCache does not have a legacy cache equivalent.")

    @classmethod
    def from_legacy_cache(cls, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None) -> "DynamicCache":
        raise NotImplementedError("HybridMambaAttentionDynamicCache does not have a legacy cache equivalent.")

    # Copied from modeling_mamba2.py
    def update_conv_state(
        self, layer_idx: int, new_conv_state: torch.Tensor, cache_init: bool = False
    ) -> torch.Tensor:
        self.conv_states[layer_idx] = new_conv_state.to(self.conv_states.device)
        return self.conv_states[layer_idx]

    def update_ssm_state(self, layer_idx: int, new_ssm_state: torch.Tensor):
        self.ssm_states[layer_idx] = new_ssm_state.to(self.ssm_states.device)
        return self.ssm_states[layer_idx]

    def reset(self):
        self.conv_states.zero_()
        self.ssm_states.zero_()


LLAMA_ATTENTION_CLASSES = {
    "eager": LlamaAttention,
    "flash_attention_2": LlamaAttention,
    "sdpa": LlamaAttention,
}


class LlamaAttentionWithQKNorm(LlamaAttention):
    """LlamaAttention + per-head QK-Norm (RMSNorm on Q/K after projection, before RoPE).
    Required by Qwen3; checkpoint weights ``self_attn.q_norm`` / ``self_attn.k_norm``
    are loaded automatically by ``from_pretrained``.  Llama models never carry these
    weights so this class is only instantiated for Qwen3."""

    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.q_norm = LlamaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = LlamaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(self, hidden_states, position_embeddings, attention_mask=None,
                past_key_value=None, cache_position=None, **kwargs):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


def _needs_qk_norm(config) -> bool:
    """Detect Qwen3 models loaded as LlamaConfig (they carry max_window_layers)."""
    return hasattr(config, "max_window_layers")


QKNORM_ATTENTION_CLASSES = {
    "eager": LlamaAttentionWithQKNorm,
    "flash_attention_2": LlamaAttentionWithQKNorm,
    "sdpa": LlamaAttentionWithQKNorm,
}


class CustomLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.hidden_size = config.hidden_size

        attn_classes = QKNORM_ATTENTION_CLASSES if _needs_qk_norm(config) else LLAMA_ATTENTION_CLASSES
        self.self_attn = attn_classes[config._attn_implementation](config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        attn_output = hidden_states.clone()
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_output,)

        return outputs


class CustomLlamaModel(LlamaModel):
    """
    Customized Llama to support HybridDynamicCache and Itermediate Layer Distillation
    """

    def __init__(self, config, hybrid_config, data_dtype):
        super().__init__(config)

        self.layers = nn.ModuleList(
            [CustomLlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        self.hybrid_config = hybrid_config
        self.data_dtype = data_dtype
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[HybridDynamicCache, List[torch.FloatTensor]]] = None,
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
        use_cache = use_cache if use_cache is not None else False
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
                
        if not isinstance(past_key_values, HybridDynamicCache): 
            if use_cache: 
                past_key_values = HybridDynamicCache(self.hybrid_config, input_ids.shape[0], self.data_dtype, device=inputs_embeds.device)
            else:
                past_key_values = None
        
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

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, past_key_values, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class CustomLlamaForCausalLM(LlamaForCausalLM, GenerationMixin):

    def __init__(self, config, hybrid_config, data_dtype=torch.bfloat16):
        super().__init__(config)
        self.model = CustomLlamaModel(config, hybrid_config, data_dtype).to(data_dtype)
        self.hybrid_config = hybrid_config
        self.data_dtype = data_dtype

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[HybridDynamicCache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        layer_input: Optional[torch.Tensor] = None,
        _labels_pre_shifted: bool = False,
        _skip_lm_head: bool = False,
        _teacher_for_kl: Optional[dict] = None,
        **kwargs,
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
        if _skip_lm_head or _teacher_for_kl is not None:
            output_hidden_states = True
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
        use_fused_linear_cross_entropy = self.hybrid_config.fused_linear_cross_entropy
        use_fused_linear_cross_entropy_compile = self.hybrid_config.fused_linear_cross_entropy_compile
        use_liger = use_fused_linear_cross_entropy and not use_fused_linear_cross_entropy_compile

        hidden_states = outputs[0]

        # --- Fused hidden-state KL: compute KL from hidden states via FLA ---
        if _teacher_for_kl is not None:
            if not _FLA_FUSED_KL_AVAILABLE:
                raise RuntimeError(
                    "fused_kl_hidden requires flash-linear-attention (FLA). "
                    "Install via: pip install -e 3rdparty/flash-linear-attention"
                )
            hidden_flat = hidden_states.view(-1, self.config.hidden_size)
            t_hidden = _teacher_for_kl['teacher_hidden'].view(
                -1, _teacher_for_kl['teacher_hidden'].shape[-1]
            )
            t_weight = _teacher_for_kl['teacher_lm_head_weight']
            loss = FusedKLDivLoss(reduction='batchmean')(
                hidden_flat, t_hidden, self.lm_head.weight, t_weight
            )
            if not return_dict:
                return (loss,) + (None,) + outputs[1:]
            return CausalLMOutputWithPast(
                loss=loss,
                logits=None,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

        # --- Skip lm_head: return hidden states only (no logits) ---
        if _skip_lm_head:
            if not return_dict:
                return (None,) + (None,) + outputs[1:]
            return CausalLMOutputWithPast(
                loss=None,
                logits=None,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        elif (use_liger or use_fused_linear_cross_entropy_compile) and labels is not None:
            # when using fused (Liger or compile) path with labels, we do not compute logits here to save memory
            logits = None
        else:
            logits = self.lm_head(hidden_states)
            logits = logits.float()

        loss = None
        if labels is not None:
            if use_fused_linear_cross_entropy_compile:
                hidden_states_flat = hidden_states.view(-1, self.config.hidden_size)
                if _labels_pre_shifted:
                    shift_labels = labels.view(-1).to(hidden_states.device)
                else:
                    shift_labels = torch.cat(
                        (labels[..., 1:], torch.full_like(labels[:, :1], _CE_IGNORE_INDEX)), dim=1
                    )
                    shift_labels = shift_labels.view(-1).to(hidden_states.device)
                loss = _chunked_lm_head_ce(
                    self.lm_head.weight, self.lm_head.bias, hidden_states_flat, shift_labels
                )
            elif use_liger:
                loss_fct = LigerFusedLinearCrossEntropyLoss()
                hidden_states_flat = hidden_states.view(-1, self.config.hidden_size)
                if _labels_pre_shifted:
                    shift_labels = labels.view(-1).to(hidden_states.device)
                else:
                    shift_labels = torch.cat((labels[..., 1:], torch.full_like(labels[:, :1], loss_fct.ignore_index)), 1)
                    shift_labels = shift_labels.view(-1)
                    shift_labels = shift_labels.to(hidden_states.device)
                loss = loss_fct(self.lm_head.weight, hidden_states_flat, shift_labels, self.lm_head.bias)
            else:
                if _labels_pre_shifted:
                    shift_logits = logits.contiguous()
                    shift_labels = labels.contiguous()
                else:
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                loss_fct = CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
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