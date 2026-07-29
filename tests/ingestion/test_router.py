# tests/ingestion/test_router.py
import asyncio
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.ingestion.job_store import JobStore
from app.ingestion.router import build_ingestion_router


def _app_with_store() -> tuple[FastAPI, JobStore]:
    store = JobStore()
    app = FastAPI()
    app.include_router(build_ingestion_router(store))
    return app, store


def test_post_ingest_returns_job_id_and_creates_queued_status():
    app, store = _app_with_store()
    with patch("app.ingestion.router.ingest_url", new=AsyncMock()):
        with TestClient(app) as client:
            response = client.post("/ingest", json={"urls": ["https://example.com"]})
    assert response.status_code == 200
    job_id = response.json()["job_id"]
    assert store.exists(job_id)


def test_post_ingest_returns_before_ingest_url_completes():
    app, store = _app_with_store()
    release = asyncio.Event()

    async def blocking_ingest_url(job_id, url, store, client):
        await release.wait()

    with patch("app.ingestion.router.ingest_url", new=blocking_ingest_url):
        with TestClient(app) as client:
            response = client.post("/ingest", json={"urls": ["https://example.com"]})
    # if we reach here, the response returned without waiting on `release`
    assert response.status_code == 200
    release.set()  # let the background task finish so it doesn't leak into other tests


def test_get_status_returns_404_for_unknown_job():
    app, _store = _app_with_store()
    with TestClient(app) as client:
        response = client.get("/ingest/does-not-exist/status")
    assert response.status_code == 404


def test_get_status_returns_contract_shape():
    app, store = _app_with_store()
    job_id = store.create_job(["https://example.com"])
    with TestClient(app) as client:
        response = client.get(f"/ingest/{job_id}/status")
    body = response.json()
    assert body["job_id"] == job_id
    assert body["urls"][0]["url"] == "https://example.com"
    assert body["urls"][0]["stage"] == "queued"
