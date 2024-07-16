from transformers import pipeline

# Load the text generation pipeline with the specified model
text_generation_pipeline = pipeline("text-generation", model="meta-llama/Meta-Llama-3-8B-Instruct")

# Function to generate a response
def generate_response(prompt, pipeline, max_length=512, temperature=0.7):
    response = pipeline(prompt, max_length=max_length, temperature=temperature, pad_token_id=pipeline.tokenizer.eos_token_id)
    return response[0]['generated_text']

# Example usage
prompt = "What is the capital of France?"
response = generate_response(prompt, text_generation_pipeline)
print(f"Prompt: {prompt}\nResponse: {response}")