import torch
from transformers import AutoTokenizer
from mla_inference.hybrid_wrapper import MLATransformerHybridModelWrapper


checkpoint_path = "amd/X-EcoMLA-1B1B-fixed-kv512-DPO"

model = MLATransformerHybridModelWrapper.from_pretrained(checkpoint_path, torch_dtype=torch.bfloat16).cuda()
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
model.eval()


prompt = [{"role": "user", "content": "What are the advantages of hybrid language models?"}]
input_ids = tokenizer.apply_chat_template(
    prompt,
    add_generation_prompt=True,
    return_tensors='pt'
).cuda()

# Generate input prompts
with torch.no_grad():
    output = model.generate(
                input_ids,
                max_length=input_ids.shape[1] + 256,
                cg=True,
                return_dict_in_generate=False,
                output_scores=False,
                eos_token_id=tokenizer.eos_token_id,
                enable_timing=True
            )
    
print(tokenizer.decode(output[0], skip_special_tokens=False))
