import uuid

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


def _isolated_settings(tmp_path) -> Settings:
    # create_app() builds a QdrantVectorIndex on whatever settings it's given.
    # QdrantClient's local (file-based) mode takes an exclusive lock on its
    # storage directory, so any test that calls create_app() against the
    # shared default path collides with the module's own `app = create_app()`
    # instance (created at import time) and with other tests in this file.
    # Isolate each call to its own tmp_path and collection name, matching the
    # convention already used throughout tests/indexing/ for QdrantVectorIndex.
    return Settings(
        qdrant_path=str(tmp_path / "qdrant"),
        qdrant_collection=f"test-{uuid.uuid4()}",
        vector_size=2,
    )


def test_ingest_routes_are_registered(tmp_path):
    fresh_app = create_app(_isolated_settings(tmp_path))
    paths = set(fresh_app.openapi()["paths"].keys())
    assert "/ingest" in paths
    assert "/ingest/{job_id}/status" in paths


def test_full_ingest_and_status_round_trip_with_fresh_app(tmp_path):
    fresh_app = create_app(_isolated_settings(tmp_path))
    with TestClient(fresh_app) as client:
        response = client.post("/ingest", json={"urls": ["https://example.com"]})
        job_id = response.json()["job_id"]
        status = client.get(f"/ingest/{job_id}/status")
    assert status.status_code == 200
    assert status.json()["job_id"] == job_id
