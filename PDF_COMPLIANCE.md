# PDF Compliance Upgrade

This branch aligns the repository with the uploaded Generative AI Skill Development Tasks PDF.

## Tasks
1. Vectorized NumPy scaled dot-product attention with 4D multi-head causal masking.
2. Custom BPE vocabulary construction and causal language-model training with explicit masking.
3. Local Llama 3 8B-Instruct Q4_K_M GGUF workflow with Ollama/Transformers logit and entropy extraction.
4. Multi-layer HNSW graph with probabilistic level assignment and cosine search.
5. Manual self-attention backward pass with gradients for Q/K/V projection weights.
6. Async FastAPI RAG with SentenceTransformers bi-encoder, FAISS retrieval, cross-encoder reranking.
7. LoRA adapters with frozen base weights and dataset-ready training.
8. AutoGen-style specialist swarm with Redis blackboard, locking and PostgreSQL persistence.
9. Kafka/PySpark/SentenceTransformers/Qdrant streaming pipeline.
10. DDPM forward scheduler plus reverse denoising loop.
11. Semantic prompt-injection detection middleware with optional Transformers/NeMo integration.
12. CLIP teacher/student multimodal distillation with cosine + KL loss.
13. DPO policy/reference training with chosen/rejected preference pairs.
14. Actual PyTorch C++/CUDA SwiGLU extension and benchmark entry point.
15. FastAPI/Redis token-bucket gateway with provider failover and Prometheus metrics.

The original notebooks are retained as learning artifacts. The `implementation.py` files in each task folder are the PDF-aligned executable implementations/configuration entry points.