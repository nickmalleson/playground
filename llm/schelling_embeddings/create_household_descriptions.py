# Generate household descriptions for the schelling model, using togeteher.ai to access an open-source LLM.

from datetime import datetime
from together import Together  # pip install together

# Number of descriptions to produce
N = 149

# Prepare the messages
prompt = \
    f"""Produce {N} one-sentence, anonymous, detailed descriptions of stereotypical UK households, describing 
    their household structure, income and political beliefs. 
    Output in CSV format with one line per household description and nothing else."""
messages = [ { "role": "system", "content": prompt } ]

# Get my API key
with open('together.ai_key.txt', 'r') as f:
    api_key = f.readline().strip()

# Create the client
client = Together(api_key=api_key)

# Call the API using parameters that ChatGPT recommends for this task
print("Calling together.ai...",)
response = client.chat.completions.create(
    # model="meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
    model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    messages=messages,
    #max_tokens=max_tokens,  # max length of output (in case I get the prompt wront and it talks for ages...)
    temperature=0.01,  # lower for more deterministic
    #top_p=0.9,  # ??
    #top_k=40,  # ??
    repetition_penalty=1,
    stop=["<|eot_id|>", "<|eom_id|>"],
    stream=False  # Set stream to False to get the full response
)
print("...received response.")

# Extract the assistant's reply and get the IDs and scores
assistant_reply = response.choices[0].message.content.strip()

# Useful to have a full log for debugging etc
print(f"{datetime.now().strftime('%Y-%m-%d-%H%M%S')} RESPONSE:\n" \
    f"**MESSAGE**\n{messages}\n" \
    f"**RESPONSE**\n{assistant_reply}\n\n")

with open(f"household_descriptions-{N}.csv", "w") as f:
    for line in assistant_reply.split('\n'):
        f.write(line + "\n")
    print(f"Saved to {f.name}")