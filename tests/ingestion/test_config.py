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


def test_retrieval_and_synthesis_defaults(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    import importlib
    import app.config as config_module
    importlib.reload(config_module)

    assert config_module.settings.retrieval_top_k == 20
    assert config_module.settings.display_top_k == 8
    assert config_module.settings.rrf_k == 60
    assert config_module.settings.synthesis_context_budget == 6
    assert config_module.settings.groq_model == "openai/gpt-oss-120b"
    assert config_module.settings.groq_api_key == ""
