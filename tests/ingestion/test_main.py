from fastapi.testclient import TestClient

from app.main import create_app


def test_ingest_routes_are_registered():
    app = create_app()
    paths = set(app.openapi()["paths"].keys())
    assert "/ingest" in paths
    assert "/ingest/{job_id}/status" in paths


def test_full_ingest_and_status_round_trip_with_fresh_app():
    app = create_app()
    with TestClient(app) as client:
        response = client.post("/ingest", json={"urls": ["https://example.com"]})
        job_id = response.json()["job_id"]
        status = client.get(f"/ingest/{job_id}/status")
    assert status.status_code == 200
    assert status.json()["job_id"] == job_id
