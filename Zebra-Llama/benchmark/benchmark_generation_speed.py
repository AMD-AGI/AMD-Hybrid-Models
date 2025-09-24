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

from hybrid.hybrid_config import HybridConfig
from hybrid_inference.hybrid_model_wrapper import HybridModelWrapper


@torch.inference_mode()
def warmup(generate_fn):
    for _ in range(1):
        _ = generate_fn()
    torch.cuda.synchronize()

@torch.inference_mode()
def measure_inference_speed(generate_fn, repeat_time):
    for i in range(repeat_time):
        print(f"Run {i + 1}:")
        # Run the generation function
        output = generate_fn()
        
def main():
    """Main function to parse arguments, set up the model, and run the benchmark."""
    parser = argparse.ArgumentParser(description="Generation benchmarking for Hybrid-Llama models.")
    parser.add_argument("--model-path", type=str, default="amd/Zebra-Llama-3B-6MLA-22Mamba-SFT",
                        help="Path or name of the pretrained model.")
    parser.add_argument("--prompt-len", type=int, default=100,
                        help="The length of the input prompt.")
    parser.add_argument("--gen-len", type=int, default=100,
                        help="The number of tokens to generate.")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="The batch size for inference.")
    parser.add_argument("--repeats", type=int, default=3,
                        help="Number of times to repeat the benchmark measurement.")
    
    args = parser.parse_args()

    device = "cuda"
    dtype = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = HybridModelWrapper.from_pretrained(args.model_path, torch_dtype=dtype, absorb=True)

    model.eval()
    model.to(device)

    # Print model parameter count
    print(f"#Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Generate a dummy input tensor
    torch.manual_seed(0)
    input_ids = torch.randint(
        1, tokenizer.vocab_size, (args.batch_size, args.prompt_len),
        dtype=torch.long, device=device
    )

    max_length = args.prompt_len + args.gen_len

    def _gen_step():
        return model.generate(
            input_ids=input_ids,
            max_length=max_length,
            cg=True,
            return_dict_in_generate=False,
            output_scores=False,
            random_context=True,
            enable_timing=True,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    print("Warming up the model...")
    warmup(_gen_step)

    print(f"Starting {args.repeats} benchmark runs...")
    measure_inference_speed(_gen_step, args.repeats)


if __name__ == "__main__":
    main()