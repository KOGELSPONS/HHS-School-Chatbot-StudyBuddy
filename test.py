# filename: local_llm_q6.py
from llama_cpp import Llama

MODEL_PATH = "./models/Llama-3.2-3B-Instruct-Q6_K_L.gguf"

# Initialize model
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=4096,   # max context tokens
    n_threads=8   # adjust to your CPU cores
)

# Simple prompt with streaming
print("Response:\n")
for chunk in llm(
    "Explain TF-IDF in 3 short bullet points.\n",
    max_tokens=150,
    temperature=0.6,
    stream=True
):
    token = chunk["choices"][0]["text"]
    print(token, end="", flush=True)  # show live
    output_text += token              # also collect final text

# Add a newline at the end for clean prompt
print("\n\n---\nFinal collected output:\n", output_text.strip())