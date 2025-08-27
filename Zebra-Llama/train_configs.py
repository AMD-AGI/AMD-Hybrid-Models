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
    mamba_ILD_path: str = field(
        default=None,
        metadata={"help": "Path to the Mamba layers after ILD."},
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
    use_fixed_rank_for_first_and_last_block: bool = field(
        default=True,
        metadata={"help": "use_fixed_rank_for_first_and_last_block"},
    )
    v_head_dim: int = field(
        default=128,
        metadata={"help": "Dimension of v head."},
    )
    qk_nope_head_dim: int = field(
        default=64,
        metadata={"help": "Dimension of qk nope."},
    )
    q_energy_ratio: float = field(
        default=None,
        metadata={"help": "q_energy_ratio."},
    )
    use_full_kv_head: bool = field(
        default=False,
        metadata={"help": "whether to use the maximum kv head for GQA/MQA"},
    )
    kv_energy_ratio: float = field(
        default=None,
        metadata={"help": "kv_energy_ratio."},
    )
    layer_rank_list: dict = field(
        default=None,
        metadata={"help": "layer_rank_list."},
    )
    repeat_kv_before_conv: bool = field(
        default=False,
        metadata={"help": "Whether we are performing intermediate layer distillation."},
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

