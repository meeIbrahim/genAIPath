import uuid

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


def test_post_query_default_settings_marks_overflow_chunks_not_used_in_synthesis(tmp_path):
    # Regression test: with default settings (display_top_k=8, synthesis_context_budget=6)
    # and no explicit top_k override, retrieval can return more chunks than synthesis
    # actually consumes. This exercises the used_in_synthesis=False branch end-to-end.
    settings = Settings(
        qdrant_path=str(tmp_path / "qdrant"),
        qdrant_collection="t",
        vector_size=2,
        groq_api_key="test-key",
    )
    assert settings.display_top_k == 8
    assert settings.synthesis_context_budget == 6

    bm25 = InMemoryBM25Index()
    vectors = QdrantVectorIndex(settings)
    store = ChunkStore()

    chunk_ids = [str(uuid.uuid4()) for _ in range(8)]
    chunks = [
        ChunkMetadata(
            chunk_id=chunk_id,
            doc_id="d1",
            source_url="https://example.com",
            page_number=1,
            chunk_index=index,
            char_start=0,
            char_end=20,
            overlap_with_prev=0,
            indexed_at="2026-07-29T12:00:00+00:00",
            text=f"chunk number {index} about the quick brown fox",
        )
        for index, chunk_id in enumerate(chunk_ids)
    ]
    # No bm25 documents are added, so ranking is driven purely by semantic
    # similarity. Vectors move progressively further from the query vector
    # ([1.0, 0.0]), giving a strictly decreasing, deterministic similarity
    # order that matches chunk_ids order (chunk 0 most similar ... chunk 7 least).
    vector_list = [[1.0, index * 0.1] for index in range(8)]
    vectors.upsert(chunk_ids, vector_list, [chunk.model_dump() for chunk in chunks])
    store.add(chunks)

    def embed_handler(request):
        return httpx.Response(200, json={"embeddings": [[1.0, 0.0]]})

    def groq_handler(request):
        return httpx.Response(200, json={"choices": [{"message": {"content": "A generic answer with no citations."}}]})

    embedding_client = httpx.AsyncClient(transport=httpx.MockTransport(embed_handler))
    synthesis_client = httpx.AsyncClient(transport=httpx.MockTransport(groq_handler))

    app = FastAPI()
    app.include_router(build_retrieval_router(bm25, vectors, store, embedding_client, synthesis_client, settings))

    with TestClient(app) as client:
        response = client.post("/query", json={"query": "tell me about the fox"})

    assert response.status_code == 200
    body = response.json()
    retrieved = body["retrieved_chunks"]

    assert len(retrieved) == 8
    assert [chunk["chunk_id"] for chunk in retrieved] == chunk_ids

    used_flags = [chunk["used_in_synthesis"] for chunk in retrieved]
    assert used_flags[:6] == [True] * 6
    assert used_flags[6:] == [False] * 2
    assert any(flag is True for flag in used_flags)
    assert any(flag is False for flag in used_flags)
