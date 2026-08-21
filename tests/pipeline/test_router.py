import json

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

import app.pipeline.loader as loader_module
from app.config import Settings
from app.pipeline.registry import IndexingCollectionRegistry
from app.pipeline.router import build_pipeline_router


def _settings(tmp_path) -> Settings:
    return Settings(qdrant_path=str(tmp_path / "qdrant"), qdrant_collection="proute", vector_size=2)


def _embed_client() -> httpx.AsyncClient:
    def handler(request):
        body = json.loads(request.read())
        return httpx.Response(200, json={"embeddings": [[0.1, 0.2] for _ in body["input"]]})
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_pipeline_status_before_any_load(tmp_path):
    settings = _settings(tmp_path)
    registry = IndexingCollectionRegistry(settings)
    app = FastAPI()
    app.include_router(build_pipeline_router(registry, _embed_client(), settings))

    with TestClient(app) as client:
        response = client.get("/pipeline/status")

    assert response.status_code == 200
    body = response.json()
    assert body["active"] is None
    assert body["doc_counts"] == {"fixed_window": 0, "semantic": 0, "hierarchical": 0, "hierarchical_summary": 0}
    registry.close_all()


def test_pipeline_load_then_status_reflects_active_config(tmp_path, monkeypatch):
    monkeypatch.setattr(loader_module, "extract_pdf_text", lambda path: "one two three. four five six.")
    settings = _settings(tmp_path)
    registry = IndexingCollectionRegistry(settings)

    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    (archive_dir / "a.pdf").write_bytes(b"a")
    monkeypatch.setattr(loader_module, "DEFAULT_ARCHIVE_DIR", archive_dir)

    app = FastAPI()
    app.include_router(build_pipeline_router(registry, _embed_client(), settings))

    with TestClient(app) as client:
        load_response = client.post("/pipeline/load", json={
            "indexing_strategy": "fixed_window", "retrieval_strategy": "hybrid_rrf", "post_retrieval_strategy": "none",
        })
        status_response = client.get("/pipeline/status")

    assert load_response.status_code == 200
    assert load_response.json()["indexed"]["new_docs"] == 1
    assert status_response.json()["active"] == {
        "indexing_strategy": "fixed_window", "retrieval_strategy": "hybrid_rrf", "post_retrieval_strategy": "none",
    }
    assert status_response.json()["doc_counts"]["fixed_window"] == 1
    registry.close_all()


def test_pipeline_load_with_unimplemented_indexing_strategy_returns_501(tmp_path, monkeypatch):
    monkeypatch.setattr(loader_module, "extract_pdf_text", lambda path: "one two three. four five six.")
    settings = _settings(tmp_path)
    registry = IndexingCollectionRegistry(settings)

    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    (archive_dir / "a.pdf").write_bytes(b"a")
    monkeypatch.setattr(loader_module, "DEFAULT_ARCHIVE_DIR", archive_dir)

    app = FastAPI()
    app.include_router(build_pipeline_router(registry, _embed_client(), settings))

    with TestClient(app) as client:
        response = client.post("/pipeline/load", json={
            "indexing_strategy": "semantic", "retrieval_strategy": "hybrid_rrf", "post_retrieval_strategy": "none",
        })
        status_response = client.get("/pipeline/status")

    assert response.status_code == 501
    assert status_response.json()["active"] is None
    registry.close_all()


def test_pipeline_load_rejects_unknown_strategy_id(tmp_path):
    settings = _settings(tmp_path)
    registry = IndexingCollectionRegistry(settings)
    app = FastAPI()
    app.include_router(build_pipeline_router(registry, _embed_client(), settings))

    with TestClient(app) as client:
        response = client.post("/pipeline/load", json={
            "indexing_strategy": "not_real", "retrieval_strategy": "hybrid_rrf", "post_retrieval_strategy": "none",
        })

    assert response.status_code == 422
    registry.close_all()
