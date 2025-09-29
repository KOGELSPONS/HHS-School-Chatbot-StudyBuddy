# LLaMA 3.2 3B Instruct (Q6_K_L) Model

This folder expects the file `Llama-3.2-3B-Instruct-Q6_K_L.gguf`.

## Manual Download

1. Download the model file from:
   `https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q6_K_L.gguf`
2. Place the downloaded `.gguf` file in this `models` directory.

If the file ever becomes gated, log in to Hugging Face and use a personal access token with read permissions; otherwise you can download anonymously.

## Automatic Download

Run the helper script from the project root while the virtual environment is active:

```bash
python models/download_model.py
```

Use `--token` or set `HUGGINGFACE_TOKEN` only if Hugging Face requires authentication for your account.
