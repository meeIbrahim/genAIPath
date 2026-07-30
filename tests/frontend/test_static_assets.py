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


def test_shared_css_is_served(tmp_path):
    app = create_app(_isolated_settings(tmp_path))
    with TestClient(app) as client:
        response = client.get("/static/css/app.css")
    assert response.status_code == 200
    assert "chunk-card" in response.text


def test_shared_api_js_is_served(tmp_path):
    app = create_app(_isolated_settings(tmp_path))
    with TestClient(app) as client:
        response = client.get("/static/js/api.js")
    assert response.status_code == 200
    assert "function postIngest" in response.text
    assert "function getIngestStatus" in response.text
    assert "function postQuery" in response.text
