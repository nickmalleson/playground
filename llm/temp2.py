from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

# Function to generate a response
def generate_response(prompt, model, tokenizer, max_length=512, temperature=0.7):
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids

    # Generate response
    with torch.no_grad():
        outputs = model.generate(input_ids, max_length=max_length, temperature=temperature, pad_token_id=tokenizer.eos_token_id)

    # Decode and return the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Example usage
prompt = "What is the capital of France?"
response = generate_response(prompt, model, tokenizer)
print(f"Prompt: {prompt}\nResponse: {response}")