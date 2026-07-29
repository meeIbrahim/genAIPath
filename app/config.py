from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    fetch_timeout_seconds: float = 10.0
    user_agent: str = "Mozilla/5.0 (compatible; GenAI-RAG-Ingest/1.0)"
    max_pages: int = 20
    min_extract_length: int = 200


settings = Settings()
