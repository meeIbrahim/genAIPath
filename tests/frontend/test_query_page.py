import uuid

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


def _isolated_settings(tmp_path) -> Settings:
    return Settings(
        qdrant_path=str(tmp_path / "qdrant"),
        qdrant_collection=f"test-{uuid.uuid4()}",
        vector_size=2,
    )


def test_query_page_served(tmp_path):
    app = create_app(_isolated_settings(tmp_path))
    with TestClient(app) as client:
        response = client.get("/query-ui")
    assert response.status_code == 200
    assert "Ask a question" in response.text
    assert '/static/js/query.js' in response.text
    assert 'id="preferences"' in response.text
    assert 'id="filtered-note"' in response.text
