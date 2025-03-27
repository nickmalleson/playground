# %% [markdown]
# # Hands-On Large Language Models: Chapter 3
# Experiments with the book code available on [github](https://github.com/HandsOnLLM/Hands-On-Large-Language-Models)

# %% 
# Imports
#
# !pip install transformers>=4.41.2 sentence-transformers>=3.0.1 gensim>=4.3.2 scikit-learn>=1.5.0 accelerate>=0.31.0
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# %%
# # Prepare model
# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    device_map="cpu",
    torch_dtype="auto",
    trust_remote_code=True,
)

# Create a pipeline
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False,
    max_new_tokens=50,
    do_sample=False,
)

# %%
# CHUNK
print(generator("How to", max_length=50))

# %% 
# CHUNK 
pass

# %% 
# CHUNK 
pass

# %% 
# CHUNK 
pass

# %% 
# CHUNK 
pass

# %% 
# CHUNK 
pass

# %% 
# CHUNK 
pass

# %% 
# CHUNK 
pass

# %% 
# CHUNK 
pass

# %% 
# CHUNK 
pass
# %%
# CHUNK
pass

# %%
# CHUNK
pass

# %%
# CHUNK
pass

# %%
# CHUNK
pass

# %%
# CHUNK
pass
# %%
# CHUNK
pass

# %%
# CHUNK
pass

# %%
# CHUNK
pass

# %%
# CHUNK
pass

# %%
# CHUNK
pass
