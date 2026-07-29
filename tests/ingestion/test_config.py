from app.config import Settings, settings


def test_defaults():
    assert settings.fetch_timeout_seconds == 10.0
    assert settings.max_pages == 20
    assert settings.min_extract_length == 200
    assert "GenAI" in settings.user_agent


def test_settings_is_frozen():
    with __import__("pytest").raises(Exception):
        settings.max_pages = 5
