from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    fetch_timeout_seconds: float = 10.0
    user_agent: str = "Mozilla/5.0 (compatible; GenAI-RAG-Ingest/1.0)"
    max_pages: int = 20
    min_extract_length: int = 200
    chunk_size_tokens: int = 400
    chunk_overlap_tokens: int = 75
    embedding_model: str = "qwen3-embedding:0.6b"
    ollama_base_url: str = "http://localhost:11434"
    qdrant_url: str | None = None
    qdrant_path: str = ".data/qdrant"
    qdrant_collection: str = "rag_chunks"
    vector_size: int = 1024


settings = Settings()
