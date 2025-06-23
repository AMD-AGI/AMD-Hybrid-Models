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
class DistillConfig(transformers.TrainingArguments):
    """
    Arguments related to the distillation process.
    """

    model_name: str = field(
        default="HuggingFaceH4/zephyr-7b-beta",
        metadata={"help": "HuggingFace model to distill from."},
    )
    prev_checkpoint_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the previous distilled model in this progressive distillation."},
    )
    ssm_layers: List[int] = field(
        default=None,
        metadata={"help": "List of SSM layers."},
    )
    mla_layers: List[int] = field(
        default=None,
        metadata={"help": "List of MLA layers."},
    )
    q_lora_rank: int = field(
        default=None,
        metadata={"help": "Rank of Q."},
    )
    qk_rope_head_dim: int = field(
        default=None,
        metadata={"help": "Dimension of qk rope."},
    )
    kv_lora_rank: int = field(
        default=None,
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
        default=None,
        metadata={"help": "Dimension of v head."},
    )
    qk_nope_head_dim: int = field(
        default=None,
        metadata={"help": "Dimension of qk nope."},
    )
    q_energy_ratio: float = field(
        default=None,
        metadata={"help": "the ratio of kept svd energy for input_Q matrix."},
    )
    mla_num_key_value_heads: int = field(
        default=None,
        metadata={"help": "Number of key-value heads in MLA layers."},
    )
    kv_energy_ratio: float = field(
        default=None,
        metadata={"help": "kv_energy_ratio."},
    )
    layer_rank_list: dict = field(
        default=None,
        metadata={"help": "layer_rank_list."},
    )
    kl_weight: float = field(
        default=0.1,
        metadata={"help": "Ratio of KL loss."},
    )
    ce_weight: float = field(
        default=1,
        metadata={"help": "Ratio of CE loss."},
    )
    train_datasets_path: List[str] = field(
        default=None,
        metadata={"help": "Training datasets."},
    )
    test_datasets_path: List[str] = field(
        default=None,
        metadata={"help": "Test datasets."},
    )
    decontaminate: bool = field(default=False, metadata={"help": "Whether to apply the decontaminate steps."})
    max_seq_length: int = field(
        default=None,
        metadata={"help": "max_seq_length."},
    )
    use_flash_attention_2: bool = field(default=False, metadata={"help": "Whether to use the FLASH Attention."})
    with_distill: bool = field(
        default=True,
        metadata={"help": "Whether we have the first stage of distillation."},
    )
    torch_dtype: str = field(
        default=None,
        metadata={"help": "torch_dtype."},
    )
    init_with_svd: bool = field(default=True, metadata={"help": "Whether to init with transformer weights."})
    progressive_step: List[int] = field(
        default=None,
        metadata={"help": "The current progressive step."},
    )
    total_progressive_step: List[int] = field(
        default=None,
        metadata={"help": "The total progressive step."},
    )
    freeze_non_mla: bool = field(default=False, metadata={"help": "Whether to freeze non-mla parameters."})
    data_ratio: float = field(
        default=1,
        metadata={"help": "Indicates the portion of training data."},
    )

@dataclass
class SFTDistillConfig(SFTConfig):
    """
    Arguments related to the distillation process.
    """

    with_distill: bool = field(
        default=True,
        metadata={"help": "Whether we have the first stage of distillation."},
    )
    prev_checkpoint_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the previous distilled model in this progressive distillation."},
    )
    ssm_layers: List[int] = field(
        default=None,
        metadata={"help": "List of SSM layers."},
    )
    mla_layers: List[int] = field(
        default=None,
        metadata={"help": "List of MLA layers."},
    )
    q_lora_rank: int = field(
        default=None,
        metadata={"help": "Rank of Q."},
    )
    qk_rope_head_dim: int = field(
        default=None,
        metadata={"help": "Dimension of qk rope."},
    )
    kv_lora_rank: int = field(
        default=None,
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
        default=None,
        metadata={"help": "Dimension of v head."},
    )
    qk_nope_head_dim: int = field(
        default=None,
        metadata={"help": "Dimension of qk nope."},
    )
    q_energy_ratio: float = field(
        default=None,
        metadata={"help": "q_energy_ratio."},
    )
    mla_num_key_value_heads: int = field(
        default=None,
        metadata={"help": "Number of key-value heads in MLA layers."},
    )
    kv_energy_ratio: float = field(
        default=None,
        metadata={"help": "kv_energy_ratio."},
    )
    layer_rank_list: dict = field(
        default=None,
        metadata={"help": "layer_rank_list."},
    )
    init_with_svd: bool = field(default=True, metadata={"help": "Whether to init with transformer weights."})
    decontaminate: bool = field(default=False, metadata={"help": "Whether to apply the decontaminate steps."})

    teacher_model_name_or_path: Optional[str] = field(default=None, metadata={"help": "Teacher Model."})
    # teacher_model_init_kwargs: Optional[Dict[str, Any]] = field(default=None, metadata={"help": "Teacher Model Init Kwargs."})
    teacher_load_in_8bit: bool = field(default=False, metadata={"help": "Whether to load the Teacher Model in 8bit."})
    kl_weight: float = field(
        default=0.1,
        metadata={"help": "Ratio of KL loss."},
    )
    ce_weight: float = field(
        default=1,
        metadata={"help": "Ratio of CE loss."},
    )
    freeze_non_mla: bool = field(default=False, metadata={"help": "Whether to freeze non-mla parameters."})
    data_ratio: float = field(
        default=1,
        metadata={"help": "Indicates the portion of training data."},
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

