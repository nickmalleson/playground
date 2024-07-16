from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

#device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
device = torch.device("cpu")

# Load the model and tokenizer
model_name = "meta-llama/Meta-Llama-3-8B-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, token=True)
model = AutoModelForCausalLM.from_pretrained(model_name, token=True)
model.to(device)

# Define a function to create the prompt with examples
def create_prompt(text):
    """
    Create a prompt string with examples for text classification.

    Args:
        text (str): The text to classify.

    Returns:
        str: The complete prompt with the provided text.
    """
    return f"""
Classify the following text as positive or negative sentiment:

Text: "I had a wonderful experience with this product. Highly recommend it!"
Sentiment: Positive

Text: "The product broke the first time I used it. Very disappointing."
Sentiment: Negative

Text: "I absolutely love this! Will buy again."
Sentiment: Positive

Text: "I hate this product. It did not meet my expectations at all."
Sentiment: Negative

Text: "{text}"
Sentiment:
"""


def create_prompt2(text):
    return f"""
Classify the following text as positive or negative sentiment:
Text: "{text}"
Sentiment:
"""

# Define a function to classify text using the model
def classify_text(text, prompt_f):
    """
    Classify the sentiment of the provided text using the LLaMA model.

    Args:
        text (str): The text to classify.

    Returns:
        str: The classified sentiment (Positive or Negative).
    """
    # Create the prompt with the input text
    prompt = prompt_f(text)

    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt")

    # Move input tensors to the MPS device
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Generate a response from the model
    outputs = model.generate(inputs['input_ids'], max_length=inputs['input_ids'].shape[1] + 20, do_sample=False)

    # Decode the generated response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract the sentiment from the generated response
    sentiment = response.split("Sentiment:")[-1].strip().split("\n")[0]

    return sentiment


# Example usage
for f in [create_prompt, create_prompt2]:
    print(f"Using prompt function: {f.__name__}")

    text = "The movie was fantastic and I enjoyed every minute of it."
    sentiment = classify_text(text, prompt_f=f)
    print(f"Text: {text}\nSentiment: {sentiment}")

    text = "What an atrocious display."
    sentiment = classify_text(text, prompt_f=f)
    print(f"Text: {text}\nSentiment: {sentiment}")