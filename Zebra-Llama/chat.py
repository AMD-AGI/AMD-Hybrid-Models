import torch
from transformers import AutoTokenizer
# from hybrid.hybrid_wrapper import HybridModelWrapper

from hybrid_inference.hybrid_model_wrapper import HybridModelWrapper
# checkpoint_path = "/home/mnt/mingyyan/checkpoints/hybrid_QWEN_7B_7B_mla_8_mamba20_Fix96_qr1536_qh64_stage2-dpo"
checkpoint_path = "/home/mingyyan@amd.com/checkpoints/Zebra-8B8B-8MLA24M2" 

model = HybridModelWrapper.from_pretrained(checkpoint_path, torch_dtype=torch.bfloat16).cuda()
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
model.eval()


prompt = [{"role": "user", "content": "Janet\u2019s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"}]
input_ids = tokenizer.apply_chat_template(
    prompt,
    add_generation_prompt=True,
    return_tensors='pt'
).cuda()

# compiled_model = torch.compile(model, mode="reduce-overhead", dynamic=True) 


prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            on_trace_ready=torch.profiler.tensorboard_trace_handler("runs/my_compressed_run")
) 

torch.manual_seed(42)  # For CPU operations
torch.cuda.manual_seed_all(42)  # For all GPUs
torch.use_deterministic_algorithms(True)

def _gen_function():
    return model.generate(
            input_ids, 
            max_new_tokens=512,
            temperature=0.7,
            top_k=1,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            cg=True,
            cg_piecewise=True,
            profile=False,
            random_context=False
            )

compiled_function  = torch.compile(_gen_function, mode="default")
output = compiled_function()


# prof.export_chrome_trace('torch_compile.json')

print(tokenizer.decode(output[0], skip_special_tokens=False))

