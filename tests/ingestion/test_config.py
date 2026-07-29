from app.config import Settings, settings


def test_defaults():
    assert settings.fetch_timeout_seconds == 10.0
    assert settings.max_pages == 20
    assert settings.min_extract_length == 200
    assert "GenAI" in settings.user_agent


def test_settings_is_frozen():
    with __import__("pytest").raises(Exception):
        settings.max_pages = 5


def test_indexing_defaults():
    from app.config import settings
    assert settings.chunk_size_tokens == 400
    assert settings.chunk_overlap_tokens == 75
    assert settings.embedding_model == "qwen3-embedding:0.6b"
    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.qdrant_url is None
    assert settings.qdrant_collection == "rag_chunks"
    assert settings.vector_size == 1024
