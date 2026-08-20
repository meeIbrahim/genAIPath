import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Settings:
    fetch_timeout_seconds: float = 10.0
    chunk_size_tokens: int = 400
    chunk_overlap_tokens: int = 75
    embedding_model: str = "qwen3-embedding:0.6b"
    ollama_base_url: str = "http://localhost:11434"
    qdrant_url: str | None = None
    qdrant_path: str = ".data/qdrant"
    qdrant_collection: str = "rag_chunks"
    vector_size: int = 1024
    retrieval_top_k: int = 20
    display_top_k: int = 8
    rrf_k: int = 60
    synthesis_context_budget: int = 6
    groq_model: str = "openai/gpt-oss-120b"
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_api_key: str = field(default_factory=lambda: os.environ.get("GROQ_API_KEY", ""))


settings = Settings()
