# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

import os
import sys
import transformers
import dataclasses
from dataclasses import dataclass, field
from typing import Any, List, Dict, Tuple, Optional, NewType
from transformers import HfArgumentParser
from alignment import SFTConfig

DataClassType = NewType("DataClassType", Any)

@dataclass
class SFTDistillConfig(SFTConfig):
    """
    Arguments related to the distillation process.
    """
    with_distill: bool = field(
        default=True,
        metadata={"help": "Whether we have the first stage of distillation."},
    )
    ILD: bool = field(
        default=False,
        metadata={"help": "Whether we are performing intermediate layer distillation"},
    )
    linear_type: str = field(
        default="mamba",
        metadata={"help": "Linear layer type: 'mamba' or 'gdn'. Determines which linear (non-MLA) layers to use."},
    )
    linear_ILD_path: str = field(
        default=None,
        metadata={"help": "Path to the linear layers (Mamba or GDN) after ILD. Overrides mamba_ILD_path when set."},
    )
    mamba_ILD_path: str = field(
        default=None,
        metadata={"help": "Path to the Mamba layers after ILD (legacy). Ignored if linear_ILD_path is set."},
    )
    mla_ILD_path: str = field(
        default=None,
        metadata={"help": "Path to the MLA layers after ILD."},
    )
    mla_layers: List[int] = field(
        default=None,
        metadata={"help": "List of MLA layers."},
    )
    q_lora_rank: int = field(
        default=128,
        metadata={"help": "Rank of Q."},
    )
    qk_rope_head_dim: int = field(
        default=64,
        metadata={"help": "Dimension of qk rope."},
    )
    kv_lora_rank: int = field(
        default=128,
        metadata={"help": "Rank of LV."},
    )
    use_lora_layer_norm: bool = field(
        default=True,
        metadata={"help": "use_lora_layer_norm"},
    )
    v_head_dim: int = field(
        default=128,
        metadata={"help": "Dimension of v head."},
    )
    qk_nope_head_dim: int = field(
        default=64,
        metadata={"help": "Dimension of qk nope."},
    )
    use_full_kv_head: bool = field(
        default=False,
        metadata={"help": "whether to use the maximum kv head for GQA/MQA"},
    )
    use_rope: bool = field(
        default=True,
        metadata={"help": "Whether to use RoPE in MLA layers. Set to false for NoPE variant."},
    )
    disable_qk_norm: bool = field(
        default=False,
        metadata={"help": "Force-disable QK-Norm in MLA layers even for models that use it (e.g. Qwen3)."},
    )
    layer_rank_list: dict = field(
        default=None,
        metadata={"help": "layer_rank_list."},
    )
    repeat_kv_before_conv: bool = field(
        default=False,
        metadata={"help": "Whether we are performing intermediate layer distillation."},
    )
    gdn_num_heads: int = field(
        default=None,
        metadata={"help": "GDN num_heads when linear_type=gdn. Default derived from 0.75*hidden_size."},
    )
    gdn_head_dim: int = field(
        default=None,
        metadata={"help": "GDN head_dim when linear_type=gdn. Default derived from key_dim // num_heads."},
    )
    init_with_svd: bool = field(
        default=True, 
        metadata={"help": "Whether to init MLA weights."},
    )
    init_with_kqvo: bool = field(
        default=True, 
        metadata={"help": "Whether to init Mamba weights."},
    )
    decontaminate: bool = field(
        default=False, 
        metadata={"help": "Whether to apply the decontaminate steps."},
    )
    teacher_model_name_or_path: Optional[str] = field(
        default=None, 
        metadata={"help": "Path to the Teacher Model."},
    )
    kl_weight: float = field(
        default=0.1,
        metadata={"help": "Ratio of KL loss."},
    )
    ce_weight: float = field(
        default=1,
        metadata={"help": "Ratio of CE loss."},
    )
    data_ratio: float = field(
        default=1,
        metadata={"help": "Indicates the portion of training data."},
    )
    factor: float = field(
        default=1.0,
        metadata={"help": "Define RoPE scaling factor."},
    )
    original_max_position_embeddings: int = field(
        default=2048,
        metadata={"help": "Define the original max positional embeddings"},
    )
    fused_linear_cross_entropy: bool = field(
        default=False,
        metadata={"help": "Whether to use fused linear lm_head + cross entropy (Liger kernel)."},
    )
    fused_linear_cross_entropy_compile: bool = field(
        default=False,
        metadata={"help": "Use torch.compile for lm_head + cross_entropy instead of Liger (avoids Triton/liger multinode mem faults)."},
    )
    chunked_kl: bool = field(
        default=False,
        metadata={"help": "Compute KL divergence in chunks along the sequence dim to avoid OOM on long contexts with a teacher."},
    )
    fused_kl: bool = field(
        default=False,
        metadata={"help": "Use FLA's Triton-fused KL divergence (fused lm_head + KL). Much faster than chunked_kl; never materializes full logits."},
    )
    fused_kl_hidden: bool = field(
        default=False,
        metadata={"help": "Compute KL from hidden states via FLA FusedKLDivLoss (never materializes full logits). For long-context KD."},
    )
    packing: bool = field(
        default=True,
        metadata={"help": "Whether to pack the data."},
    )

# Copy from HuggingFace H4ArgumentParser
class DistillArgumentParser(HfArgumentParser):
    def parse_yaml_and_args(self, yaml_arg: str, other_args: Optional[List[str]] = None) -> List[dataclass]:
        """
        Parse a YAML file and overwrite the default/loaded values with the values provided to the command line.

        Args:
            yaml_arg (`str`):
                The path to the config file used
            other_args (`List[str]`, *optional`):
                A list of strings to parse as command line arguments, e.g. ['--arg=val', '--arg2=val2'].

        Returns:
            [`List[dataclass]`]: a list of dataclasses with the values from the YAML file and the command line
        """
        arg_list = self.parse_yaml_file(os.path.abspath(yaml_arg))

        outputs = []
        # strip other args list into dict of key-value pairs
        other_args = {arg.split("=")[0].strip("-"): arg.split("=")[1] for arg in other_args}
        used_args = {}

        # overwrite the default/loaded value with the value provided to the command line
        # adapted from https://github.com/huggingface/transformers/blob/d0b5002378daabf62769159add3e7d66d3f83c3b/src/transformers/hf_argparser.py#L327
        for data_yaml, data_class in zip(arg_list, self.dataclass_types):
            keys = {f.name for f in dataclasses.fields(data_yaml) if f.init}
            inputs = {k: v for k, v in vars(data_yaml).items() if k in keys}
            for arg, val in other_args.items():
                # add only if in keys

                if arg in keys:
                    base_type = data_yaml.__dataclass_fields__[arg].type
                    inputs[arg] = val

                    # cast type for ints, floats (default to strings)
                    if base_type in [int, float]:
                        inputs[arg] = base_type(val)

                    if base_type == List[str]:
                        inputs[arg] = [str(v) for v in val.split(",")]
                    
                    # if base_type == List[int]:
                    #     inputs[arg] = [int(v.strip()) for v in val.split(",")]

                    # bool of a non-empty string is True, so we manually check for bools
                    if base_type == bool:
                        if val in ["true", "True"]:
                            inputs[arg] = True
                        else:
                            inputs[arg] = False

                    # add to used-args so we can check if double add
                    if arg not in used_args:
                        used_args[arg] = val
                    else:
                        raise ValueError(f"Duplicate argument provided: {arg}, may cause unexpected behavior")

            obj = data_class(**inputs)
            outputs.append(obj)

        return outputs

    def parse(self) -> DataClassType | Tuple[DataClassType]:
        if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
            # If we pass only one argument to the script and it's the path to a YAML file,
            # let's parse it to get our arguments.
            output = self.parse_yaml_file(os.path.abspath(sys.argv[1]))
        # parse command line args and yaml file
        elif len(sys.argv) > 2 and sys.argv[1].endswith(".yaml"):
            output = self.parse_yaml_and_args(os.path.abspath(sys.argv[1]), sys.argv[2:])
        # parse command line args only
        else:
            output = self.parse_args_into_dataclasses()

        if len(output) == 1:
            output = output[0]
        return output

