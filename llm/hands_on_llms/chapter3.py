# %% [markdown]
# # Hands-On Large Language Models: Chapter 3
# Experiments with the book code available on [github](https://github.com/HandsOnLLM/Hands-On-Large-Language-Models)

# %% 
# Imports
#
import time

# !pip install transformers>=4.41.2 sentence-transformers>=3.0.1 gensim>=4.3.2 scikit-learn>=1.5.0 accelerate>=0.31.0
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Choose most suitable device
device = 'cpu'  # default
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
print(f"Using {device}")

# %%
# # Prepare model
model_name = "microsoft/Phi-3-mini-4k-instruct"
# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map=device,
    torch_dtype="auto",
    trust_remote_code=False,
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
# Test by generating some text
start_time = time.time()
for _ in range(5):
    print(generator("How to"))
print(f"Took {time.time() - start_time} seconds")

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
