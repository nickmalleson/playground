from llama_cpp import Llama
llm = Llama(model_path="../../llama.cpp/models/13B/ggml-model-q4_0.gguf",
            n_gpu_layers=1)
for i in range(10):
    output = llm("Q: Name the planets in the solar system? A: ", max_tokens=32, stop=["Q:", "\n"], echo=True)

    print(f"\nOUTPUT{output['choices']}\n")
