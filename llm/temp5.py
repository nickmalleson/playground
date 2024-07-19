# Use quantization to reduce the memory footprint of the model (only works on CPU)

from datetime import datetime
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import gc
import torch
torch.backends.quantized.engine = 'qnnpack'  # or 'fbgemm' for some setups (for quantisation on the cpu)


# Load the tokenizer and the model
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the model and apply dynamic quantization
print("Loading model...\n")
model = AutoModelForCausalLM.from_pretrained(model_name)
print("Quantizing model...\n")
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
# Free up memory
del model
gc.collect()

# Create the text generation pipeline with the quantized model
print("Creating text generation pipeline...\n")
text_generation_pipeline = pipeline("text-generation", model=quantized_model, tokenizer=tokenizer, device="cpu")

# Function to generate a response
def generate_response(prompt, pipeline, max_length=512, temperature=0.7):
    response = pipeline(prompt, max_length=max_length, temperature=temperature, pad_token_id=pipeline.tokenizer.eos_token_id)
    return response[0]['generated_text']

def create_prompt2(text):
    return f"""
Classify the following text as positive, neutral or negative sentiment, showing your working:
Text: "{text}"
Sentiment:
"""

# Example usage
texts = [
    "The movie was fantastic and I enjoyed every minute of it.",
    "What an atrocious display.",
    "Hollis' death scene will hurt me severely to watch on film wry is directors cut not out now?",
    "@smarrison i would've been the first, but i didn't have a gun. not really though, zac snyder's ju...",
    "about to file taxes ",
    "im sad now Miss.Lilly",
    "ooooh.... LOL that leslie.... and ok I won't do it again so leslie won't get mad again ",
    "Bed. Class 8-12. Work 12-3. Gym 3-5 or 6. Then class 6-10. Another day that's gonna fly by. I miss my old life",
    "Sad, sad, sad. I don't know why but I hate this feeling I wanna sleep and I still can't!",
    "great day!"
]

print("Beginning text generation...\n")
for i, text in enumerate(texts):
    start_time = datetime.now()
    prompt = create_prompt2(text)
    response = generate_response(prompt, text_generation_pipeline)
    end_time = datetime.now() - start_time
    print(f"{i} Time taken: {end_time}.\n\tText: {text}\n\tPrompt: {prompt}\n\tResponse: {response}")