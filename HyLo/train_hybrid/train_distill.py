#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
# Modifications Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Supervised fine-tuning script for decoder language models.
"""

import os as _os_for_triton

# ---------------------------------------------------------------------------
# Per-rank Triton cache directory.
#
# Triton's compiled-kernel disk cache (default: $HOME/.triton/cache, which
# resolves to /workspace/.triton/cache inside this container because
# start_docker.sh sets HOME=$WORKDIR) is shared across all training
# processes. With N ranks all autotuning the same kernel (e.g.
# `causal_conv1d_fwd_kernel` from FLA's GDN short-conv), the autotuner's
# `check_disk_cache` path races against in-progress writes from peer ranks
# and crashes with:
#     FileNotFoundError: [Errno 2] No such file or directory:
#     '/workspace/.triton/cache/<hash>/causal_conv1d_fwd_kernel.json'
# even though the kernel will eventually compile successfully on every rank.
#
# Triton's atomic-rename write protocol assumes a single writer per cache
# dir; the only robust fix is to give each rank its own cache directory.
# This MUST happen before any module that imports `triton` (transitively
# via `fla`, `mamba_ssm`, flash-attn, etc.) is imported, so it lives at
# the very top of the file.
#
# We key on global `RANK` (set by torchrun/accelerate for distributed
# launches), falling back to `LOCAL_RANK` for single-node launches and
# `0` for single-process invocations.
# ---------------------------------------------------------------------------
_triton_rank = _os_for_triton.environ.get("RANK") or _os_for_triton.environ.get("LOCAL_RANK") or "0"
# Also set TRITON_HOME for ROCm triton variants that key off it instead of
# TRITON_CACHE_DIR.
_triton_base = _os_for_triton.environ.get(
    "TRITON_HOME", _os_for_triton.path.expanduser("~/.triton")
)
_os_for_triton.environ["TRITON_HOME"] = _os_for_triton.path.join(
    _triton_base, f"rank_{_triton_rank}"
)
_os_for_triton.environ["TRITON_CACHE_DIR"] = _os_for_triton.path.join(
    _os_for_triton.environ["TRITON_HOME"], "cache"
)

import logging
import os
import random
import sys
import shutil

sys.path.append('./')

import datasets
import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, set_seed, BitsAndBytesConfig

from alignment import (
    DataArguments,
    H4ArgumentParser,
    ModelArguments,
    apply_chat_template,
    decontaminate_humaneval,
    get_checkpoint,
    get_datasets,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    get_tokenizer,
)
from trl import setup_chat_format
from trainer.kd_trainer import KDTrainer
from trl.trainer.sft_trainer import SFTTrainer

from hybrid.hybrid_wrapper import HybridModelWrapper
from hybrid.hybrid_config import HybridConfig

from train_configs import SFTDistillConfig
from util import construct_layer_dict

import torch.distributed as dist
from datetime import timedelta
from datasets import DatasetDict, load_dataset, concatenate_datasets, Dataset

# ---------------------------------------------------------------------------
# Monkey-patches for alignment.data:
#
# 1. load_dataset  – convert Alpaca-format datasets (instruction/input/output)
#    to chat-format ("messages" column), AND normalise struct field order in
#    the "messages" column to {role, content} so all datasets are compatible
#    for PyArrow concatenation.
#
# 2. concatenate_datasets – safety-net cast that compares actual PyArrow
#    schemas (not HF Features, whose __eq__ ignores struct field order).
# ---------------------------------------------------------------------------
import alignment.data as _alignment_data
import pyarrow as pa

def _messages_need_reorder(ds):
    """Return True if the 'messages' struct has fields in non-{role,content} order."""
    try:
        schema = ds._data.schema if hasattr(ds._data, "schema") else ds.data.schema
        msg_type = schema.field("messages").type
        while hasattr(msg_type, "value_type"):
            msg_type = msg_type.value_type
        return (isinstance(msg_type, pa.StructType)
                and msg_type.num_fields >= 2
                and msg_type.field(0).name != "role")
    except Exception:
        return False

def _reorder_messages(example):
    example["messages"] = [
        {"role": m["role"], "content": m["content"]} for m in example["messages"]
    ]
    return example

# --- patch 1: auto-convert Alpaca → messages on load, then normalise ------
_orig_load_dataset = _alignment_data.load_dataset

def _converting_load_dataset(*args, **kwargs):
    ds = _orig_load_dataset(*args, **kwargs)
    cols = ds.column_names if hasattr(ds, "column_names") else []

    if "messages" not in cols and "instruction" in cols:
        def _alpaca_to_messages(example):
            user_content = example["instruction"]
            inp = example.get("input", "") or ""
            if inp.strip():
                user_content += "\n" + inp
            return {
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": example.get("output", "")},
                ]
            }
        ds = ds.map(_alpaca_to_messages, remove_columns=[c for c in cols
                     if c not in ("messages",)])

    if "messages" in (ds.column_names if hasattr(ds, "column_names") else []):
        if _messages_need_reorder(ds):
            ds = ds.map(_reorder_messages)

    return ds

_alignment_data.load_dataset = _converting_load_dataset

# --- patch 2: safe concatenation (compare Arrow schemas, not HF Features) -
_orig_concatenate_datasets = _alignment_data.concatenate_datasets

def _safe_concatenate_datasets(dsets, **kwargs):
    if dsets and len(dsets) > 1:
        ref_schema = dsets[0]._data.schema
        for i in range(1, len(dsets)):
            if dsets[i]._data.schema != ref_schema:
                if set(dsets[i].column_names) == set(dsets[0].column_names):
                    dsets[i] = dsets[i].cast(dsets[0].features)
    return _orig_concatenate_datasets(dsets, **kwargs)

_alignment_data.concatenate_datasets = _safe_concatenate_datasets
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)


def fast_pack_dataset(dataset, tokenizer, max_seq_length, num_proc=None, dataset_text_field="text"):
    """
    Tokenize in parallel with multiprocessing, then concatenate all tokens and
    split into fixed-length chunks.  ~100x faster than TRL's single-threaded
    ConstantLengthDataset + Dataset.from_generator pipeline.
    """
    def tokenize_fn(examples):
        return tokenizer(
            examples[dataset_text_field],
            add_special_tokens=True,
            truncation=False,
            padding=False,
            return_attention_mask=False,
        )

    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        num_proc=num_proc,
        remove_columns=dataset.column_names,
        desc="Tokenizing for packing",
    )

    eos_id = tokenizer.eos_token_id

    all_ids = []
    for ids in tokenized["input_ids"]:
        all_ids.extend(ids)
        if not ids or ids[-1] != eos_id:
            all_ids.append(eos_id)

    total_tokens = len(all_ids)
    n_chunks = total_tokens // max_seq_length
    if n_chunks == 0:
        raise ValueError(
            f"Dataset too small to produce even one packed sequence "
            f"({total_tokens} tokens < {max_seq_length} max_seq_length)."
        )

    all_ids = all_ids[: n_chunks * max_seq_length]

    import numpy as np
    arr = np.array(all_ids, dtype=np.int64).reshape(n_chunks, max_seq_length)

    packed = Dataset.from_dict({
        "input_ids": arr.tolist(),
        "labels": arr.tolist(),
        "attention_mask": [[1] * max_seq_length] * n_chunks,
    })

    discarded = total_tokens - n_chunks * max_seq_length
    logger.info(
        f"Fast packing: {len(dataset)} examples -> {total_tokens:,} tokens "
        f"-> {n_chunks} packed sequences of length {max_seq_length} "
        f"({discarded} tokens discarded from tail)"
    )
    return packed


def main():
    
    dist.init_process_group(backend='nccl', timeout=timedelta(seconds=360000))

    parser = H4ArgumentParser((ModelArguments, DataArguments, SFTDistillConfig))
    config_file_path = sys.argv[1]

    model_args, data_args, training_args = parser.parse()
        
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # ----- Distributed connection check: global rank, world size, and env (to verify multi-node) -----
    world_size = dist.get_world_size()
    global_rank = dist.get_rank()
    main_addr = os.environ.get("MAIN_PROCESS_IP", os.environ.get("MASTER_ADDR", "unset"))
    main_port = os.environ.get("MAIN_PROCESS_PORT", os.environ.get("MASTER_PORT", "unset"))
    machine_rank = os.environ.get("MACHINE_RANK", "unset")
    num_machines = os.environ.get("NUM_MACHINES", os.environ.get("NNODES", "unset"))
    logger.warning(
        f"[Distributed] global_rank={global_rank} world_size={world_size} "
        f"local_rank={training_args.local_rank} | MAIN_PROCESS_IP={main_addr} MAIN_PROCESS_PORT={main_port} "
        f"MACHINE_RANK={machine_rank} NUM_MACHINES={num_machines}"
    )
    if global_rank == 0:
        logger.warning(
            f"[Distributed] Expected single process group: world_size should equal total GPUs across all nodes. "
            f"If you see duplicate local_rank 0-7 per node, check MASTER_ADDR/MACHINE_RANK are set correctly."
        )
    sys.stdout.flush()
    sys.stderr.flush()

    # Log on each process a small summary
    logger.warning(
        f"Process local_rank: {training_args.local_rank}, global_rank: {global_rank}, world_size: {world_size}, "
        f"device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f", 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    ################
    # Load tokenizer (before datasets so chat template can be applied)
    ################
    tokenizer = get_tokenizer(model_args, data_args)

    ###############
    # Load datasets — global rank 0 processes first so cache is ready for other ranks.
    #
    # IMPORTANT: pass `local=False` so the barrier is GLOBAL, not per-node.
    # The HF datasets cache is typically on a filesystem shared across nodes,
    # so the default `local=True` would let each
    # node's local rank-0 race against the others on the same Arrow files.
    # With `num_proc=preprocessing_num_workers` this means N_NODES * N_WORKERS
    # processes writing to the same cache shards, which corrupts them and
    # crashes processing (the symptom is the worker pool dying mid-`Filter:`
    # / `Map:` progress bar). With `local=False` only the global main process
    # builds the cache and every other rank on every node loads it.
    ###############
    with training_args.main_process_first(local=False, desc="Loading and processing datasets"):
        raw_datasets = get_datasets(
            data_args,
            splits=data_args.dataset_splits,
            configs=data_args.dataset_configs,
            columns_to_keep=["messages", "chosen", "rejected", "prompt", "completion", "label"],
        )

        logger.info(
            f"Training on the following datasets and their proportions: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
        )
        column_names = list(raw_datasets["train"].features)

        #####################
        # Apply chat template
        #####################
        raw_datasets = raw_datasets.map(
            apply_chat_template,
            fn_kwargs={
                "tokenizer": tokenizer,
                "task": "sft",
                "auto_insert_empty_system_msg": data_args.auto_insert_empty_system_msg,
            },
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            desc="Applying chat template",
        )

    #######################
    # Load pretrained model
    #######################
    logger.info("*** Load pretrained model ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        use_flash_attention_2=model_args.use_flash_attention_2,
        torch_dtype=torch_dtype,
        use_cache=False,
        # use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    model = model_args.model_name_or_path
    # For ChatML we need to add special tokens and resize the embedding layer
    if "<|im_start|>" in tokenizer.chat_template and "gemma-tokenizer-chatml" not in tokenizer.name_or_path:
        model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)
        model, tokenizer = setup_chat_format(model, tokenizer)
        model_kwargs = None

    ##########################
    # Decontaminate benchmarks
    ##########################
    # NOTE: shuffle/select/filter all write Arrow cache files. Same shared-FS
    # multi-node concern as the load/chat-template block above — use
    # `local=False` so only global rank 0 writes the cache.
    if training_args.decontaminate:
        with training_args.main_process_first(local=False, desc="Decontaminating dataset"):
            #this segment is added for quick verification of the SFT distillation training
            shuffled_dataset = raw_datasets["train"].shuffle(seed=42)
            if training_args.ILD:
                num_selected_train_samples = int(training_args.data_ratio*len(raw_datasets["train"])) 
                selected_raw_dataset = shuffled_dataset.select(list(range(1,num_selected_train_samples)))
            else:
                num_selected_train_samples = int((1-training_args.data_ratio)*len(raw_datasets["train"])) 
                selected_raw_dataset = shuffled_dataset.select(list(range(num_selected_train_samples, len(raw_datasets["train"]))))

            dd = {'train': selected_raw_dataset}
            if 'test' in raw_datasets:
                dd['test'] = raw_datasets['test']
            raw_datasets = DatasetDict(dd)
            num_raw_train_samples = len(raw_datasets["train"])
            raw_datasets = raw_datasets.filter(decontaminate_humaneval, batched=True, batch_size=10_000, num_proc=data_args.preprocessing_num_workers)
            num_filtered_train_samples = num_raw_train_samples - len(raw_datasets["train"])
            logger.info(
                f"Decontaminated {num_filtered_train_samples} ({num_filtered_train_samples/num_raw_train_samples * 100:.2f}%) samples from the training set."
            )

    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets.get("test", None)

    with training_args.main_process_first(local=False, desc="Log a few random samples from the processed training set"):
        for index in random.sample(range(len(raw_datasets["train"])), 3):
            logger.info(f"Sample {index} of the processed training set:\n\n{raw_datasets['train'][index]['text']}")

    use_fast_packing = False
    if use_fast_packing:
        logger.info("*** Fast packing: pre-tokenizing dataset in parallel ***")
        with training_args.main_process_first(local=False, desc="Fast parallel tokenization + packing"):
            train_dataset = fast_pack_dataset(
                train_dataset,
                tokenizer,
                max_seq_length=training_args.max_seq_length,
                num_proc=data_args.preprocessing_num_workers,
                dataset_text_field="text",
            )
            if eval_dataset is not None:
                eval_dataset = fast_pack_dataset(
                    eval_dataset,
                    tokenizer,
                    max_seq_length=training_args.max_seq_length,
                    num_proc=data_args.preprocessing_num_workers,
                    dataset_text_field="text",
                )

    attn_implementation="flash_attention_2"
    if not model_args.use_flash_attention_2:
        attn_implementation="sdpa"
        # Transformers disables the flash SDPA backend on ROCm multi-GPU setups.
        # Re-enable it so SDPA doesn't fall back to O(N^2) math mode.
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
    
    ##########################
    # Initialize the model
    ##########################
    if not training_args.with_distill:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, dtype=model_args.torch_dtype)
        mla_layers = training_args.mla_layers
        
        if not hasattr(config, 'head_dim'):
            d_xb = config.num_key_value_heads * \
                (config.hidden_size // config.num_attention_heads)
            d_inner = config.hidden_size
            d_state = config.hidden_size//config.num_attention_heads
        else:
            # to handle gemma2
            d_xb = config.num_key_value_heads * config.head_dim
            d_inner = config.num_attention_heads * config.head_dim
            d_state = config.head_dim
        
        # Set up positional embeddings
        rope_scaling = {
            "factor": training_args.factor,
            "original_max_position_embeddings": training_args.original_max_position_embeddings,
            "rope_type": "yarn",
        }
        
        linear_type = getattr(training_args, "linear_type", "mamba").lower()
        linear_ILD_path = getattr(training_args, "linear_ILD_path", None)
        if linear_ILD_path is None:
            linear_ILD_path = getattr(training_args, "mamba_ILD_path", None)
        hybrid_config = HybridConfig(
            # Common parameters
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            n_layer=config.num_hidden_layers,
            mla_layers=mla_layers,
            rms_norm_eps=config.rms_norm_eps,
            fused_linear_cross_entropy=training_args.fused_linear_cross_entropy,
            fused_linear_cross_entropy_compile=training_args.fused_linear_cross_entropy_compile,
            # MLA parameters
            attention_dropout=config.attention_dropout,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads if not training_args.use_full_kv_head else config.num_attention_heads,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            q_lora_rank=training_args.q_lora_rank,
            qk_rope_head_dim=training_args.qk_rope_head_dim,
            kv_lora_rank=training_args.kv_lora_rank,
            use_lora_layer_norm=training_args.use_lora_layer_norm,
            use_full_kv_head=training_args.use_full_kv_head,
            v_head_dim=training_args.v_head_dim,
            qk_nope_head_dim=training_args.qk_nope_head_dim,
            attention_bias=config.attention_bias,
            rope_scaling=rope_scaling,
            use_rope=training_args.use_rope,
            qk_norm=hasattr(config, "max_window_layers") and not training_args.disable_qk_norm,
            # Mamba Parameters
            d_model=config.hidden_size,
            ssm_cfg={"expand": 1, "ngroups":config.num_attention_heads, "d_state": d_state, "repeat_kv_before_conv": training_args.repeat_kv_before_conv},
            d_inner=d_inner,
            d_xb=d_xb,
            # Linear type (mamba | gdn) and optional GDN params
            linear_type=linear_type,
            gdn_num_heads=getattr(training_args, "gdn_num_heads", None),
            gdn_head_dim=getattr(training_args, "gdn_head_dim", None),
        )
        model = HybridModelWrapper.init_distillation(
            None, model_args.model_name_or_path, hybrid_config, mla_layers=mla_layers, init_with_svd=training_args.init_with_svd,
            init_with_kqvo=training_args.init_with_kqvo,
            mamba_model_path=training_args.mamba_ILD_path,
            mla_model_path=training_args.mla_ILD_path,
            linear_ILD_path=linear_ILD_path,
            linear_type=linear_type,
            attn_implementation=attn_implementation,
        )
    else:
        model = HybridModelWrapper.from_pretrained(model_args.model_name_or_path, attn_implementation=attn_implementation)
    
    print("#Params:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # ── Preflight ILD check ──────────────────────────────────────────────
    # Verify that ILD checkpoints were actually loaded before starting an
    # expensive training run.  Catches the case where the checkpoint dir
    # exists but has no model.safetensors / pytorch_model.bin.
    if not training_args.with_distill:
        _linear_path = linear_ILD_path
        _mla_path    = getattr(training_args, "mla_ILD_path", None)
        _expect_linear = _linear_path is not None and len(mla_layers) < hybrid_config.n_layer
        _expect_mla    = _mla_path   is not None and len(mla_layers) > 0

        if _expect_linear and not getattr(model, "has_linear_ild", False):
            raise RuntimeError(
                f"Preflight check FAILED: linear ILD path was set ({_linear_path}) "
                f"but linear ILD weights were NOT loaded. Ensure the checkpoint "
                f"contains model.safetensors or pytorch_model.bin "
                f"(run convert_to_safetensors.py if only FSDP shards exist)."
            )
        if _expect_mla and not getattr(model, "has_mla_ild", False):
            raise RuntimeError(
                f"Preflight check FAILED: MLA ILD path was set ({_mla_path}) "
                f"but MLA ILD weights were NOT loaded. Ensure the checkpoint "
                f"contains model.safetensors or pytorch_model.bin."
            )

        if _expect_linear or _expect_mla:
            # Quick sanity: check a few layer norms aren't all-ones (default init)
            _bad_layers = []
            for _li in range(hybrid_config.n_layer):
                _layer = model.model.model.layers[_li]
                _ln_w = _layer.input_layernorm.weight.data
                if _ln_w.float().std().item() < 1e-7:
                    _bad_layers.append(_li)
            if _bad_layers:
                logger.warning(
                    f"Preflight WARNING: layers {_bad_layers} have constant layernorm "
                    f"weights (std≈0). These may not have loaded from ILD checkpoints."
                )

            # Quick forward-pass smoke test on rank 0
            if global_rank == 0:
                try:
                    _dummy = torch.zeros(1, 4, dtype=torch.long, device="cuda")
                    _dummy[0] = torch.tensor([1, 2, 3, 4])
                    with torch.no_grad():
                        _out = model.model(_dummy)
                        _logits = _out.logits if hasattr(_out, "logits") else _out[0]
                        _loss_est = torch.nn.functional.cross_entropy(
                            _logits[0, :-1].float(), _dummy[0, 1:],
                        ).item()
                    logger.info(f"Preflight OK: smoke-test CE loss = {_loss_est:.2f} "
                                f"(expect < 12 for ILD-initialized model, "
                                f"> 12 suggests weights not loaded)")
                    if _loss_est > 15:
                        logger.warning(
                            f"Preflight WARNING: smoke-test loss {_loss_est:.2f} is very "
                            f"high — ILD weights may not have loaded correctly!"
                        )
                except Exception as e:
                    logger.warning(f"Preflight smoke test failed (non-fatal): {e}")

        logger.info(
            f"Preflight summary: linear_ILD={'LOADED' if getattr(model, 'has_linear_ild', False) else 'NOT loaded'}, "
            f"mla_ILD={'LOADED' if getattr(model, 'has_mla_ild', False) else 'NOT loaded'}, "
            f"linear_type={getattr(model, 'linear_type', 'unknown')}"
        )
    # ── End preflight ────────────────────────────────────────────────────

    model.save_config(training_args.output_dir, config_file_path)
    model = model.model
    
    ##############################
    # Set up teacher model configs
    ##############################
    teacher_model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    # Apply the SAME YaRN RoPE scaling to the teacher as the student. The teacher's
    # rotary embedding is built from its own HF config, so without this override it
    # would run at its native (e.g. 32k) window while the student is extended to
    # `factor * original_max_position_embeddings`. For long-context KD the teacher
    # must see the same positions, otherwise its hidden states and logits are
    # computed under a different positional regime than the student is distilled to.
    if getattr(training_args, "use_rope", True) and getattr(training_args, "factor", 1.0):
        teacher_model_kwargs["rope_scaling"] = {
            "rope_type": "yarn",
            "factor": training_args.factor,
            "original_max_position_embeddings": training_args.original_max_position_embeddings,
        }
        # Ensure the teacher's max_position_embeddings advertises the extended window
        # so length checks don't clamp positions below the student's.
        teacher_model_kwargs["max_position_embeddings"] = int(
            training_args.factor * training_args.original_max_position_embeddings
        )
    training_args.teacher_model_init_kwargs = teacher_model_kwargs

    ########################
    # Initialize the Trainer
    ########################
    trainer_packing = False if use_fast_packing else training_args.packing

    if training_args.teacher_model_name_or_path is not None:
        trainer = KDTrainer(
            model=model,
            teacher_model=training_args.teacher_model_name_or_path,
            hybrid_config=hybrid_config,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            dataset_text_field="text",
            max_seq_length=training_args.max_seq_length,
            tokenizer=tokenizer,
            packing=trainer_packing,
            peft_config=get_peft_config(model_args),
            dataset_kwargs=training_args.dataset_kwargs,
        )
    else:
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            dataset_text_field="text",
            max_seq_length=training_args.max_seq_length,
            tokenizer=tokenizer,
            packing=trainer_packing,
            peft_config=get_peft_config(model_args),
            dataset_kwargs=training_args.dataset_kwargs,
        )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
        
    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "dataset": list(data_args.dataset_mixer.keys()),
        "dataset_tags": list(data_args.dataset_mixer.keys()),
        "tags": ["alignment-handbook"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)
        trainer.tokenizer.save_pretrained(training_args.output_dir)
        
    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.push_to_hub is True:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)

    logger.info("*** Training complete ***")


if __name__ == "__main__":
    main()