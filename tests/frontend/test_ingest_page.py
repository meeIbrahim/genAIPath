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


def test_ingest_page_served_at_root(tmp_path):
    app = create_app(_isolated_settings(tmp_path))
    with TestClient(app) as client:
        response = client.get("/")
    assert response.status_code == 200
    assert "Ingest URLs" in response.text
    assert '/static/js/ingest.js' in response.text
