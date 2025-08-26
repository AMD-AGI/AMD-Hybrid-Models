# Copyright (c) 2023, Tri Dao, Albert Gu.

import argparse
import time
import json
import torch
import torch.nn.functional as F
from einops import rearrange
from dataclasses import dataclass, field
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import numpy as np 

sys.path.append('/home/mingyyan@amd.com/AMD-Hybrid-Models/Zebra-Llama')
from hybrid.hybrid_config import HybridConfig
from hybrid_inference.hybrid_model_wrapper import HybridModelWrapper
# from hybrid.hybrid_wrapper import HybridModelWrapper


@torch.inference_mode()
def warmup(generate_fn):
    for _ in range(1):
        _ = generate_fn()
    torch.cuda.synchronize()

@torch.inference_mode()
def measure_inference_speed_and_memory(input_ids, generate_fn, repeat_time):

    total_time_list = []
    tokens_per_second_list = []
    memory_used_list = []
    
    for i in range(repeat_time):
        # Clear GPU memory cache
        torch.cuda.empty_cache()
        
        # Measure initial memory usage
        initial_memory = torch.cuda.memory_allocated()  # In bytes
        
        torch.cuda.synchronize()
        start_time = time.perf_counter()  # Start timer
        output = generate_fn()
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        # Measure peak memory usage
        peak_memory = torch.cuda.max_memory_allocated()  # In bytes

        # Calculate metrics
        total_tokens = (output.shape[1] - input_ids.shape[1]) * output.shape[0]  # Subtract input tokens
        total_time = end_time - start_time
        tokens_per_second = total_tokens / total_time

        # Convert memory usage to GB
        initial_memory_mb = initial_memory / (1024 ** 3)
        peak_memory_mb = peak_memory / (1024 ** 3)
        memory_used_mb = peak_memory_mb - initial_memory_mb
        
        # Append results to lists
        total_time_list.append(total_time)
        tokens_per_second_list.append(tokens_per_second)
        memory_used_list.append(memory_used_mb)


        print(f"Run {i + 1}:")
        print(f"Generation Length: {output.shape[1]} tokens")
        print(f"Total Tokens Generated: {total_tokens}")
        # print(f"Total Time: {total_time:.2f} seconds")
        # print(f"Tokens per Second: {tokens_per_second:.2f}")
        print(f"Initial Memory Usage: {initial_memory_mb:.4f} GB")
        print(f"Peak Memory Usage: {peak_memory_mb:.4f} GB")
        print(f"Memory Used During Inference: {memory_used_mb:.4f} GB")
        print("------")
           
    # Calculate averages
    avg_total_time = sum(total_time_list) / repeat_time
    std_total_time = np.std(total_time_list, ddof=1)
    avg_tokens_per_second = sum(tokens_per_second_list) / repeat_time
    std_tokens_per_second = np.std(tokens_per_second_list, ddof=1)
    avg_memory_used = sum(memory_used_list) / repeat_time
    std_memory_used = np.std(memory_used_list, ddof=1) 

    print(f"Average Results for {output.shape[1]} tokens:")
    # print(f"  Average ± SD Total Time: {avg_total_time:.2f} ± {std_total_time:.2f} seconds")
    # print(f"  Average ± SD Tokens per Second: {avg_tokens_per_second:.2f} ± {std_tokens_per_second:.2f}")
    print(f"  Average ± SD Memory Used During Inference: {avg_memory_used:.2f} ± {std_memory_used:.2f} MB")
    print("======")


def main(args):
    device = "cuda"
    dtype = torch.float16
    
    if args.model_size == '1B':
        pretrained_path = 'meta-llama/Llama-3.2-1B-Instruct'
        hidden_size=2048
        intermediate_size=8192
        n_layer=16   
        num_attention_heads=32
        num_key_value_heads=8
        q_lora_rank=1344
        qk_rope_head_dim=32
        kv_lora_rank=128
        v_head_dim=64
        qk_nope_head_dim=32
        use_full_kv_head=False
        mla_layers=[i for i in range(args.num_attn)]
    elif args.model_size == '3B':
        pretrained_path = 'meta-llama/Llama-3.2-3B-Instruct'
        hidden_size = 3072
        intermediate_size=8192
        num_attention_heads=24
        num_key_value_heads=8
        n_layer=28
        q_lora_rank=1536
        qk_rope_head_dim=64
        kv_lora_rank=128
        v_head_dim=128
        qk_nope_head_dim=64
        use_full_kv_head=False
        mla_layers=[i for i in range(args.num_attn)]
    elif args.model_size == '8B':
        pretrained_path = 'meta-llama/Llama-3.1-8B-Instruct'
        hidden_size = 4096
        intermediate_size=14336
        num_attention_heads=32
        num_key_value_heads=8
        n_layer=32
        q_lora_rank=2048
        qk_rope_head_dim=64
        kv_lora_rank=160 # if args.num_attn == 32 else 160
        v_head_dim=128
        qk_nope_head_dim=64
        use_full_kv_head=False
        mla_layers=[i for i in range(args.num_attn)]
        if args.num_attn == 8:
            mla_layers=[0,4,8,12,16,20,25,30]
        if args.num_attn == 16:
            mla_layers=[0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30]
    elif args.model_name == '70B':
        pretrained_path = 'meta-llama/Llama-3.1-70B-Instruct'
        hidden_size = 8192
        intermediate_size=28672
        num_attention_heads=64
        num_key_value_heads=8
        n_layer=80
        q_lora_rank=2048
        qk_rope_head_dim=64
        kv_lora_rank=128
        v_head_dim=128
        qk_nope_head_dim=64
        use_full_kv_head=False
        mla_layers=[i for i in range(args.num_attn)]

    hybrid_config = HybridConfig(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            n_layer=n_layer,
            mla_layers=mla_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            q_lora_rank=q_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            kv_lora_rank=kv_lora_rank,
            v_head_dim=v_head_dim,
            qk_nope_head_dim=qk_nope_head_dim,
            use_lora_layer_norm=False,
            use_full_kv_head=use_full_kv_head,
            layer_rank_list={},
            max_position_embeddings=131072,
            rope_theta=500000.0,
            rope_scaling={
                "factor": 1.0,
                "original_max_position_embeddings": 2048,
                "rope_type": "yarn"
            },
            rope_type="yarn",
            d_model=hidden_size,
            ssm_cfg={
                "expand":1,
                "ngroups":num_attention_heads,
                "d_state":hidden_size//num_attention_heads
            },
            d_inner=hidden_size,
            d_xb=hidden_size//num_attention_heads*num_key_value_heads, 
        )


    tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
    transformer_model = AutoModelForCausalLM.from_pretrained(pretrained_path, torch_dtype=dtype, attn_implementation="flash_attention_2")

    model = HybridModelWrapper(None, transformer_model, hybrid_config, mla_layers, dtype, absorb=True)

    # model = torch.compile(model, fullgraph=True)
    model.eval()
    model.to(device)
    print(f"#Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    torch.random.manual_seed(0)
    input_ids = torch.randint(1, 1000, (args.batch, args.promptlen), dtype=torch.long, device="cuda")

    # CUDA Graph Capture
    static_input_ids = input_ids.clone()
    max_length = input_ids.shape[1] + args.genlen

    def _gen_step():
        return model.generate(
            input_ids=static_input_ids,
            max_length=max_length,
            cg=True,
            cg_piecewise=True,
            profile=False,
            return_dict_in_generate=False,
            output_scores=False,
            random_context=True,
            enable_timing=True,
            eos_token_id=tokenizer.eos_token_id,
        )

    compiled_function  = torch.compile(_gen_step, mode="default") 
    warmup(compiled_function)
    measure_inference_speed_and_memory(static_input_ids, compiled_function, args.repeats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generation benchmarking")
    parser.add_argument("--model-size", type=str, default="8B")
    parser.add_argument("--promptlen", type=int, default=100)
    parser.add_argument("--genlen", type=int, default=100)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--num_attn", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)

    args = parser.parse_args()

    main(args)