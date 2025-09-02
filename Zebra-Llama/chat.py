import torch
from transformers import AutoTokenizer, GenerationConfig
from hybrid_inference.hybrid_model_wrapper import HybridModelWrapper

def main():
    """
    Main function to load a hybrid language model, generate a response, and print it.
    """
    # 1. Configuration 
    # Define model and tokenizer path, and generation parameters for easy modification.
    checkpoint_path = "amd/Zebra-Llama-1B-4MLA-12Mamba-SFT"
    max_new_tokens = 512
    temperature = 0.7
    
    try:
        # 2. Model and Tokenizer Loading 
        print("Loading model and tokenizer...")
        # Use a more robust loading method with .from_pretrained.
        model = HybridModelWrapper.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.bfloat16,
        ).cuda()
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        model.eval()

        # 3. Prepare Input
        prompt = [{"role": "user", "content": "What are the benefits of hybrid LLM models"}]
        input_ids = tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=True,
            return_tensors='pt'
        ).cuda()

        # 4. Model Generation 
        print("Generating response...")
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_length=input_ids.shape[1] + max_new_tokens,
                cg=True,
                return_dict_in_generate=False,
                output_scores=False,
                eos_token_id=tokenizer.eos_token_id,
                enable_timing=True
            )

        # 5. Decode and Print Output
        # Decode only the newly generated tokens to avoid repeating the prompt.
        decoded_output = tokenizer.decode(output[0, input_ids.shape[1]:], skip_special_tokens=True)
        print("\n--- Model Response ---")
        print(decoded_output)
        print("----------------------")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

