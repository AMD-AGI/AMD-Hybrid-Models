import torch
from transformers import AutoTokenizer
from hybrid.hybrid_wrapper import HybridModelWrapper

checkpoint_path = "/home/mnt/mingyyan/checkpoints/hybrid_QWEN_7B_7B_mla_8_mamba20_Fix96_qr1536_qh64_stage2-dpo"
# checkpoint_path = "amd/Zebra-Llama-8B-16MLA-16Mamba-DPO" 

model = HybridModelWrapper.from_pretrained(checkpoint_path, torch_dtype=torch.bfloat16).cuda()
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
model.eval()


prompt = [{"role": "user", "content": "What are the advantages of hybrid language models"}]
input_ids = tokenizer.apply_chat_template(
    prompt,
    add_generation_prompt=True,
    return_tensors='pt'
).cuda()

# Generate input prompts
with torch.no_grad():
    output = model.generate(
            input_ids, 
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id)
    
print(tokenizer.decode(output[0], skip_special_tokens=False))

