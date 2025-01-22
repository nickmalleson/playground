import os
import torch
from datetime import datetime
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import gc

try:
    torch.backends.quantized.engine = 'qnnpack'  # or 'fbgemm' for some setups (for quantization on the cpu)
except Exception as e:  # Above doesn't work on my windows pc
    print(e, "using fbgemm instead")
    torch.backends.quantized.engine = 'fbgemm' # for some setups (for quantization on the cpu)

# Paths for saving and loading the quantized model
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
quantized_model_path = "quantized_model.pt"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Function to save the quantized model
def save_quantized_model(model, path):
    torch.save(model, path)

# Function to load the quantized model
def load_quantized_model(path):
    return torch.load(path)

# Check if quantized model exists
if os.path.exists(quantized_model_path):
    print("Loading quantized model from disk...\n")
    quantized_model = load_quantized_model(quantized_model_path)
else:
    # Load and quantize the model
    print("Loading model...\n")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    print("Quantizing model...\n")
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    # Save the quantized model to disk
    print("Saving quantized model...\n")
    save_quantized_model(quantized_model, quantized_model_path)
    # Free up memory
    del model
    gc.collect()

# Create the text generation pipeline with the quantized model
print("Creating text generation pipeline...\n")
text_generation_pipeline = pipeline("text-generation",
                                    model=quantized_model,
                                    tokenizer=tokenizer,
                                    device="cpu",
                                    max_new_tokens=10  # Cut off the output as we just want one word
                                    )

# Function to generate a response
def generate_response(prompt, pipeline, temperature=0.7):
    full_response = pipeline(prompt, temperature=temperature, pad_token_id=pipeline.tokenizer.eos_token_id)
    generated_text = full_response[0]['generated_text']
    # The generated text includes the prmopt. Extract the sentiment from the generated text by splitting on the prompt
    sentiment = generated_text.split("Sentiment (one word):")[-1].strip().split()[0].lower()
    return sentiment


def create_prompt2(text, show_working=False):
    return f"""
Classify the following text as positive, neutral or negative sentiment{'' if not show_working else ', showing your working'}:
Text: "{text}"
Sentiment (one word):
"""

def create_prompt3(text):
    return f"""
Classify the following text as "positive", "neutral", or "negative" sentiment:

Text: "I love sunny days!"
Sentiment: positive

Text: "It's an okay day."
Sentiment: neutral

Text: "I am not happy with the service."
Sentiment: negative

Text: "{text}"
Sentiment:"""

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