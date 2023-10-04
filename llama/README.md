# llama

## Initialise / install

Create a conda environment (mine is called 'llama')

Install [llama.cpp](https://github.com/ggerganov/llama.cpp)

If need to convert models to gguf format: 

 - Create and execute script [convert-llama-ggml-to-gguf.py](https://github.com/ggerganov/llama.cpp/blob/master/convert-llama-ggml-to-gguf.py) (_may be available in more recent version of llama.cpp_)

## Install [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)

(I do  it with mac metal (GPU) support)

```
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir
```
