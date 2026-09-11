# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

import os
import warnings
from copy import deepcopy
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, PreTrainedModel, is_wandb_available
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from trl.trainer.sft_trainer import SFTTrainer

from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from functools import partial
from hybrid.hybrid_modeling import CustomLlamaForCausalLM
from train_configs import SFTDistillConfig

_KL_CHUNK = 4096

def _chunked_kl_div(student_logits, teacher_logits):
    """KL divergence computed in chunks along the seq dimension to avoid
    materialising the full (seq_len, vocab) softmax tensors at once."""
    flat_s = student_logits.reshape(-1, student_logits.shape[-1])
    flat_t = teacher_logits.reshape(-1, teacher_logits.shape[-1])
    total = torch.zeros(1, device=flat_s.device, dtype=torch.float32)
    n = flat_s.shape[0]
    for i in range(0, n, _KL_CHUNK):
        s = flat_s[i:i + _KL_CHUNK].float()
        t = flat_t[i:i + _KL_CHUNK].float()
        total += F.kl_div(
            F.log_softmax(s, dim=-1),
            F.softmax(t, dim=-1),
            reduction="sum",
        )
        del s, t
    return total / n


# ---------------------------------------------------------------------------
# Triton-fused KL divergence via FLA's kernel (no Python loop, no softmax
# materialization).  Falls back to chunked_kl if FLA is not installed.
# ---------------------------------------------------------------------------
try:
    import triton
    from fla.modules.fused_kl_div import kl_div_kernel
    from fla.utils import IS_AMD
    _KL_MAX_FUSED = 65536 // 2
    _KL_WARPS = 16 if IS_AMD else 32
    _FLA_KL_AVAILABLE = True
except ImportError:
    _FLA_KL_AVAILABLE = False


class _TritonKLDiv(torch.autograd.Function):
    """KL(teacher || student) computed entirely inside a single Triton kernel.

    The FLA kl_div_kernel tiles over the vocab dimension with online softmax,
    so it never materializes full (N, V) softmax/log-softmax tensors.
    Gradients w.r.t. student logits are written in-place during forward and
    returned in backward.
    """

    @staticmethod
    def forward(ctx, student_logits, teacher_logits):
        N, V = student_logits.shape
        BV = min(_KL_MAX_FUSED, triton.next_power_of_2(V))

        # Kernel overwrites the student tensor with dKL/d(logits) in-place,
        # so we operate on a clone to keep the original graph intact.
        grad_buf = student_logits.detach().contiguous().clone()
        tl_buf = teacher_logits.detach().contiguous()
        loss = torch.zeros(N, dtype=torch.float32, device=student_logits.device)

        kl_div_kernel[(N,)](
            logits=grad_buf,
            target_logits=tl_buf,
            loss=loss,
            s_logits=grad_buf.stride(0),
            s_loss=loss.stride(0),
            reduction='batchmean',
            N=N,
            V=V,
            BV=BV,
            num_warps=_KL_WARPS,
        )

        ctx.save_for_backward(grad_buf)
        return loss.sum()

    @staticmethod
    def backward(ctx, do):
        (grad_buf,) = ctx.saved_tensors
        return grad_buf * do, None


def _triton_kl_div(student_logits, teacher_logits):
    """Drop-in replacement for _chunked_kl_div backed by FLA's Triton kernel."""
    flat_s = student_logits.reshape(-1, student_logits.shape[-1])
    flat_t = teacher_logits.reshape(-1, teacher_logits.shape[-1])
    return _TritonKLDiv.apply(flat_s, flat_t)


class KDTrainer(SFTTrainer):
    _tag_names = ["trl", "kd"]

    def __init__(
        self,
        teacher_model: Union[PreTrainedModel, nn.Module, str],
        hybrid_config,
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
            teacher_model = CustomLlamaForCausalLM.from_pretrained(teacher_model, hybrid_config, **teacher_model_init_kwargs)
        
        self.teacher_model = self.prepare_fsdp(teacher_model, evaluation_mode=True)

        self.kl_weight = args.kl_weight
        self.ce_weight = args.ce_weight
        self.torch_dtype = torch_dtype
        self.args = args

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        self.teacher_model.eval()
        labels_pre_shifted = inputs.get("_labels_pre_shifted", False)
        use_fused_kl_hidden = getattr(self.args, "fused_kl_hidden", False)

        teacher_kwargs = dict(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_hidden_states=self.args.ILD or use_fused_kl_hidden,
            output_attentions=self.args.ILD,
        )
        if self.args.ILD:
            teacher_kwargs["labels"] = inputs["labels"]
            teacher_kwargs["_labels_pre_shifted"] = labels_pre_shifted
        if use_fused_kl_hidden:
            teacher_kwargs["_skip_lm_head"] = True

        with torch.no_grad():
            outputs_teacher = self.teacher_model(**teacher_kwargs)

        if use_fused_kl_hidden:
            # ---- Fused hidden-state KL: never materializes full logits ----
            teacher_hidden = outputs_teacher.hidden_states[-1].detach()
            del outputs_teacher

            with FSDP.summon_full_params(self.teacher_model, recurse=False, writeback=False):
                teacher_lm_head_weight = self.teacher_model.lm_head.weight.detach().clone()

            outputs_student = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                use_cache=False,
                _teacher_for_kl={
                    'teacher_hidden': teacher_hidden,
                    'teacher_lm_head_weight': teacher_lm_head_weight,
                },
            )
            kl_loss = outputs_student.loss
            del teacher_hidden, teacher_lm_head_weight
            loss = self.kl_weight * kl_loss

        elif self.args.ILD:
            # Intermediate layer distillation
            teacher_all_states = outputs_teacher["hidden_states"]
            teacher_all_attn_out = outputs_teacher["attentions"]
            
            outputs_student = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=inputs["labels"],
                output_hidden_states=self.args.ILD,
                output_attentions=self.args.ILD,
                layer_input=teacher_all_states,
                _labels_pre_shifted=labels_pre_shifted,
                )
            
            student_all_states = outputs_student["hidden_states"]
            student_all_attn_out = outputs_student["attentions"]

            loss = 0
            for layer_idx in range(1, len(teacher_all_states)):
                loss_value = torch.norm(student_all_states[layer_idx] - teacher_all_states[layer_idx], p=2, dim=(-1,)).mean() \
                         + 1.0 * torch.norm(teacher_all_attn_out[layer_idx-1] - student_all_attn_out[layer_idx-1], p=2, dim=(-1,)).mean()
                loss += loss_value

        else:
            teacher_logits = outputs_teacher.logits.detach()
            del outputs_teacher
            if self.ce_weight == 0:
                outputs_student = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    output_hidden_states=False,
                    use_cache=False,
                    _labels_pre_shifted=labels_pre_shifted,
                )
                cross_entropy_loss = 0.0
            else:
                outputs_student = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    labels=inputs["labels"],
                    output_hidden_states=False,
                    use_cache=False,
                    _labels_pre_shifted=labels_pre_shifted,
                )
                cross_entropy_loss = outputs_student.loss
            student_logits = outputs_student.logits
            if getattr(self.args, "fused_kl", False):
                if not _FLA_KL_AVAILABLE:
                    raise RuntimeError(
                        "fused_kl requires flash-linear-attention (FLA). "
                        "Install via: pip install -e 3rdparty/flash-linear-attention"
                    )
                kl_loss = _triton_kl_div(student_logits, teacher_logits)
            elif getattr(self.args, "chunked_kl", False):
                kl_loss = _chunked_kl_div(student_logits, teacher_logits)
            else:
                kl_loss = F.kl_div(
                    F.log_softmax(student_logits, dim=-1),
                    F.softmax(teacher_logits, dim=-1),
                    reduction="batchmean",
                )
            del teacher_logits, student_logits
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
                fsdp_plugin.auto_wrap_policy, transformer_layer_cls=set(type(layer) for layer in model.model.layers)
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
