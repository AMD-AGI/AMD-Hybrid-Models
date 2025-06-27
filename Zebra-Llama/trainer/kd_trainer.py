import os
import warnings
from copy import deepcopy
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, PreTrainedModel, is_wandb_available
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from hybrid.mamba2.hybrid_model import Mamba2DecoderLayer
from trl.trainer.sft_trainer import SFTTrainer

from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from functools import partial

from train_configs import SFTDistillConfig

class KDTrainer(SFTTrainer):
    _tag_names = ["trl", "kd"]

    def __init__(
        self,
        teacher_model: Union[PreTrainedModel, nn.Module, str],
        args: Optional[SFTDistillConfig] = None,
        torch_dtype = torch.bfloat16,
        *sft_args,
        **kwargs,
    ):
        super().__init__(*sft_args, args=args, **kwargs)        
        if args.teacher_model_init_kwargs is None:
            teacher_model_init_kwargs = {}
        elif not isinstance(teacher_model, str):
            raise ValueError(
                "You passed teacher_model_init_kwargs to the KDConfig, but your teacher_model is already instantiated."
            )
        else:
            teacher_model_init_kwargs = args.teacher_model_init_kwargs

        if isinstance(teacher_model, str):
            warnings.warn(
                "You passed a teacher model_id to the KDTrainer. This will automatically create an "
                "`AutoModelForCausalLM`"
            )
            print("teacher_model_init_kwargs:", teacher_model_init_kwargs)
            teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model, **teacher_model_init_kwargs)
        
        if not self.args.ILD:
            self.teacher_model = self.prepare_fsdp(teacher_model, evaluation_mode=True)
        else:
            # self.teacher_model = self.accelerator.prepare_model(teacher_model, evaluation_mode=True)
            self.teacher_model = self.prepare_fsdp(teacher_model, evaluation_mode=True)

        self.kl_weight = args.kl_weight
        self.ce_weight = args.ce_weight
        self.torch_dtype = torch_dtype
        self.args = args

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # compute teacher output in eval mode
        self.teacher_model.eval()
        
        with torch.no_grad():
            outputs_teacher = self.teacher_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=inputs["labels"],
                output_hidden_states=self.args.ILD
            )
        
        if self.args.ILD:
            # Intermediate layer distillation
            teacher_all_states = outputs_teacher["hidden_states"]
            #model = model.module if hasattr(model, 'module') else model
            
            outputs_student = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=inputs["labels"],
                output_hidden_states=self.args.ILD,
                layer_input=teacher_all_states
                )
            
            student_all_states = outputs_student["hidden_states"]

            loss = 0
            for layer_idx in range(1, len(teacher_all_states)):
                loss_value = torch.norm(student_all_states[layer_idx] - teacher_all_states[layer_idx], p=2, dim=(-1,)).mean()
                loss += loss_value

            #     student_input = teacher_all_states[layer_idx].to(self.torch_dtype)
                
            #     outputs_student = student_layer(hidden_states=student_input)
            #     outputs_student = outputs_student[0]

            #     if layer_idx == len(model.model.layers)-1:
            #         outputs_student = model.model.norm(outputs_student)

            #     teacher_hstate = teacher_all_states[layer_idx + 1].to(self.torch_dtype)

            #     assert outputs_student.size() == teacher_hstate.size()
            #     loss_value = torch.norm(outputs_student - teacher_hstate, p=2, dim=(-1,)).mean()
                
            #     loss += loss_value

        else:
            # compute student output
            teacher_logits = outputs_teacher.logits.detach()
            if self.ce_weight == 0:
                outputs_student = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    output_hidden_states=False,
                    use_cache=False
                )
                cross_entropy_loss = 0.0
            else:            
                # compute student output
                outputs_student = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    labels=inputs["labels"],
                    output_hidden_states=False,
                    use_cache=False
                )
                cross_entropy_loss = outputs_student.loss
            # compute cross entropy loss
            student_logits = outputs_student.logits
            kl_loss = F.kl_div(F.log_softmax(student_logits, dim=-1), F.softmax(teacher_logits, dim=-1), reduction='batchmean')
            loss = self.kl_weight * kl_loss + self.ce_weight * cross_entropy_loss

        # Return loss
        return (loss, outputs_student) if return_outputs else loss

    def prepare_fsdp(self, model, evaluation_mode=False):
        # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1421
        # don't wrap it again
        if not isinstance(model, FSDP):
            # 
            fsdp_plugin = deepcopy(self.accelerator.state.fsdp_plugin)

            fsdp_wrap_policy = partial(
                fsdp_plugin.auto_wrap_policy, transformer_layer_cls=set([LlamaDecoderLayer])
            )
            # setattr(fsdp_plugin.mixed_precision_policy, '_module_classes_to_ignore', None)
            kwargs = {
                "sharding_strategy": fsdp_plugin.sharding_strategy or fsdp_plugin.reshard_after_forward,
                "cpu_offload": fsdp_plugin.cpu_offload,
                "auto_wrap_policy": fsdp_wrap_policy,
                "mixed_precision": fsdp_plugin.mixed_precision_policy,
                "sync_module_states": fsdp_plugin.sync_module_states,
                "backward_prefetch": fsdp_plugin.backward_prefetch,
                "forward_prefetch": fsdp_plugin.forward_prefetch,
                "use_orig_params": fsdp_plugin.use_orig_params,
                "param_init_fn": fsdp_plugin.param_init_fn,
                "ignored_modules": fsdp_plugin.ignored_modules,
                "limit_all_gathers": fsdp_plugin.limit_all_gathers,
                "device_id": self.accelerator.device,
            }
            model = FSDP(model, **kwargs)

        if evaluation_mode:
            model.eval()
        else:
            model.train()
        return model
