# %% [markdown]
# # Hands-On Large Language Models: Chapter 2
# Experiments with the book code available on [github](https://github.com/HandsOnLLM/Hands-On-Large-Language-Models)

# %% 
# Imports
#
# !pip install transformers>=4.41.2 sentence-transformers>=3.0.1 gensim>=4.3.2 scikit-learn>=1.5.0 accelerate>=0.31.0
from transformers import AutoModel, AutoTokenizer

# %%
# # Tokens
# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")

# Load a LLM
model = AutoModel.from_pretrained("microsoft/deberta-base")

# Tokenize a sentence
tokens = tokenizer("Hello, world", return_tensors="pt")

# Process the tokens
output = model(**tokens)[0]

# View the tokens
for token in tokens['input_ids'][0]:
    print(tokenizer.decode(token))

# View raw output
print(output.shape)
