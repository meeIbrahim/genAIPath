import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import Settings
from app.indexing.bm25_index import InMemoryBM25Index
from app.indexing.chunk_store import ChunkStore
from app.indexing.router import build_indexing_router
from app.indexing.vector_index import QdrantVectorIndex
from app.ingestion.models import PageMapEntry


def test_post_index_chunk_returns_result_and_populates_stores(tmp_path):
    settings = Settings(qdrant_path=str(tmp_path / "qdrant"), qdrant_collection="t", vector_size=2)
    bm25 = InMemoryBM25Index()
    vectors = QdrantVectorIndex(settings)
    store = ChunkStore()

    def handler(request):
        return httpx.Response(200, json={"embeddings": [[1.0, 0.0]]})

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = FastAPI()
    app.include_router(build_indexing_router(bm25, vectors, store, http_client, settings))

    body = {
        "source_url": "https://example.com",
        "cleaned_text": "just one short sentence here.",
        "pages_fetched": 1,
        "fetched_at": "2026-07-29T12:00:00+00:00",
        "page_map": [{"page": 1, "char_start": 0, "char_end": 30}],
    }
    with TestClient(app) as client:
        response = client.post("/index/chunk", json=body)

    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "indexed"
    assert result["chunk_count"] == 1
