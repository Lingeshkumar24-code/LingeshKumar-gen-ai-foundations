# Task 3 — Local Llama 3 8B Q4_K_M

PDF-aligned workflow: obtain a Llama-3-8B-Instruct GGUF Q4_K_M model, serve it locally with Ollama, and use the local model for generation. For analytical hidden-state/logit extraction, use the corresponding Transformers checkpoint when available; GGUF/Ollama serving and Transformers introspection are separate execution paths.

Example Ollama setup:

```bash
ollama pull llama3:8b
ollama run llama3:8b
```

For entropy of returned logits, use the Transformers path with `output_hidden_states=True` and `logits` from the model output. Do not claim hidden-state extraction from Ollama's text API alone.

The Q4_K_M requirement is a model-file/quantization requirement; verify the exact GGUF artifact before submission.
