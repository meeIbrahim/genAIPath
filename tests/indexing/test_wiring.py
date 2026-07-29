import asyncio

import httpx
from fastapi import FastAPI

from app.config import Settings
from app.indexing.bm25_index import InMemoryBM25Index
from app.indexing.chunk_store import ChunkStore
from app.indexing.indexer import index_document
from app.indexing.vector_index import QdrantVectorIndex
from app.ingestion.job_store import JobStore
from app.ingestion.router import build_ingestion_router

GOOD_PAGE = (
    "<html><body><article><p>Enough real content here to clear the extraction "
    "threshold easily. This paragraph is intentionally padded with additional "
    "sentences so that the normalized extracted text comfortably exceeds the "
    "two hundred character minimum length required by the default ingestion "
    "settings.</p></article></body></html>"
)


def _fetch_handler(request: httpx.Request) -> httpx.Response:
    if str(request.url) == "https://example.com/article":
        return httpx.Response(200, text=GOOD_PAGE)
    return httpx.Response(404, text="not found")


async def test_ingest_then_index_populates_chunk_store(tmp_path):
    settings = Settings(qdrant_path=str(tmp_path / "qdrant"), qdrant_collection="wiring", vector_size=2)
    bm25 = InMemoryBM25Index()
    vectors = QdrantVectorIndex(settings)
    store = ChunkStore()

    def embed_handler(request):
        return httpx.Response(200, json={"embeddings": [[1.0, 0.0]]})

    embed_client = httpx.AsyncClient(transport=httpx.MockTransport(embed_handler))

    async def sink(payload):
        await index_document(payload, bm25, vectors, store, embed_client, settings)

    job_store = JobStore()
    app = FastAPI()
    app.include_router(
        build_ingestion_router(
            job_store,
            client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(_fetch_handler)),
            sink=sink,
        )
    )

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/ingest", json={"urls": ["https://example.com/article"]})
        job_id = response.json()["job_id"]

        elapsed = 0.0
        while elapsed < 2.0:
            status = (await client.get(f"/ingest/{job_id}/status")).json()
            if status["urls"][0]["stage"] in ("done", "error"):
                break
            await asyncio.sleep(0.01)
            elapsed += 0.01

    assert status["urls"][0]["stage"] == "done"
    assert len(store._chunks) >= 1
    await embed_client.aclose()
