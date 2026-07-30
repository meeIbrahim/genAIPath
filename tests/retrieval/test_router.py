import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import Settings
from app.indexing.bm25_index import InMemoryBM25Index
from app.indexing.chunk_store import ChunkStore
from app.indexing.models import ChunkMetadata
from app.indexing.vector_index import QdrantVectorIndex
from app.retrieval.router import build_retrieval_router


def test_post_query_returns_answer_with_citations_and_chunks(tmp_path):
    settings = Settings(
        qdrant_path=str(tmp_path / "qdrant"),
        qdrant_collection="t",
        vector_size=2,
        groq_api_key="test-key",
        synthesis_context_budget=5,
    )
    bm25 = InMemoryBM25Index()
    vectors = QdrantVectorIndex(settings)
    store = ChunkStore()

    chunk_id = "11111111-1111-1111-1111-111111111111"
    chunk = ChunkMetadata(
        chunk_id=chunk_id,
        doc_id="d1",
        source_url="https://example.com",
        page_number=1,
        chunk_index=0,
        char_start=0,
        char_end=20,
        overlap_with_prev=0,
        indexed_at="2026-07-29T12:00:00+00:00",
        text="the quick brown fox",
    )
    bm25.add_documents([chunk_id], ["the quick brown fox"])
    vectors.upsert([chunk_id], [[1.0, 0.0]], [chunk.model_dump()])
    store.add([chunk])

    def embed_handler(request):
        return httpx.Response(200, json={"embeddings": [[1.0, 0.0]]})

    def groq_handler(request):
        return httpx.Response(200, json={"choices": [{"message": {"content": "The fox is quick [1]."}}]})

    embedding_client = httpx.AsyncClient(transport=httpx.MockTransport(embed_handler))
    synthesis_client = httpx.AsyncClient(transport=httpx.MockTransport(groq_handler))

    app = FastAPI()
    app.include_router(build_retrieval_router(bm25, vectors, store, embedding_client, synthesis_client, settings))

    with TestClient(app) as client:
        response = client.post("/query", json={"query": "tell me about the fox"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "The fox is quick [1]."
    assert body["citations"] == [{"marker": 1, "chunk_id": chunk_id}]
    assert body["retrieved_chunks"][0]["chunk_id"] == chunk_id
    assert body["retrieved_chunks"][0]["used_in_synthesis"] is True
