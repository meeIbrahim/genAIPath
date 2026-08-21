import json

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import Settings
from app.indexing.models import ChunkMetadata
from app.pipeline.config import PipelineConfig, set_active
from app.pipeline.registry import IndexingCollectionRegistry
from app.retrieval.router import build_retrieval_router


def _settings(tmp_path, **overrides) -> Settings:
    defaults = dict(qdrant_path=str(tmp_path / "qdrant"), qdrant_collection="t", vector_size=2, groq_api_key="test-key")
    defaults.update(overrides)
    return Settings(**defaults)


def _embed_handler(request):
    return httpx.Response(200, json={"embeddings": [[1.0, 0.0]]})


def test_query_without_active_pipeline_returns_400(tmp_path):
    settings = _settings(tmp_path)
    registry = IndexingCollectionRegistry(settings)
    embedding_client = httpx.AsyncClient(transport=httpx.MockTransport(_embed_handler))
    synthesis_client = httpx.AsyncClient(transport=httpx.MockTransport(_embed_handler))
    app = FastAPI()
    app.include_router(build_retrieval_router(registry, embedding_client, synthesis_client, settings))

    with TestClient(app) as client:
        response = client.post("/query", json={"query": "anything"})

    assert response.status_code == 400
    registry.close_all()


def test_query_uses_active_pipeline_returns_answer_with_citations_and_chunks(tmp_path):
    settings = _settings(tmp_path)
    registry = IndexingCollectionRegistry(settings)
    collection = registry.get("fixed_window")

    chunk_id = "11111111-1111-1111-1111-111111111111"
    chunk = ChunkMetadata(
        chunk_id=chunk_id, doc_id="d1", doc_id_hash="h1", source_url="paper.pdf", page_number=1,
        chunk_index=0, char_start=0, char_end=20, overlap_with_prev=0,
        indexed_at="2026-07-29T12:00:00+00:00", text="the quick brown fox",
    )
    collection.bm25_index.add_documents([chunk_id], ["the quick brown fox"])
    collection.vector_index.upsert([chunk_id], [[1.0, 0.0]], [chunk.model_dump()])
    collection.chunk_store.add([chunk])

    set_active(PipelineConfig(indexing_strategy="fixed_window", retrieval_strategy="hybrid_rrf", post_retrieval_strategy="none"))

    def groq_handler(request):
        return httpx.Response(200, json={"choices": [{"message": {"content": "The fox is quick [1]."}}]})

    embedding_client = httpx.AsyncClient(transport=httpx.MockTransport(_embed_handler))
    synthesis_client = httpx.AsyncClient(transport=httpx.MockTransport(groq_handler))
    app = FastAPI()
    app.include_router(build_retrieval_router(registry, embedding_client, synthesis_client, settings))

    with TestClient(app) as client:
        response = client.post("/query", json={"query": "tell me about the fox"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "The fox is quick [1]."
    assert body["citations"] == [{"marker": 1, "chunk_id": chunk_id}]
    assert body["retrieved_chunks"][0]["chunk_id"] == chunk_id
    assert body["retrieved_chunks"][0]["used_in_synthesis"] is True
    assert "judge_attempts" not in body
    registry.close_all()


def test_query_with_unimplemented_post_retrieval_strategy_returns_501(tmp_path):
    settings = _settings(tmp_path, qdrant_collection="t_cer")
    registry = IndexingCollectionRegistry(settings)
    collection = registry.get("fixed_window")

    chunk_id = "55555555-5555-5555-5555-555555555555"
    chunk = ChunkMetadata(
        chunk_id=chunk_id, doc_id="d1", doc_id_hash="h1", source_url="paper.pdf", page_number=1,
        chunk_index=0, char_start=0, char_end=20, overlap_with_prev=0,
        indexed_at="2026-07-29T12:00:00+00:00", text="the quick brown fox",
    )
    collection.bm25_index.add_documents([chunk_id], ["the quick brown fox"])
    collection.vector_index.upsert([chunk_id], [[1.0, 0.0]], [chunk.model_dump()])
    collection.chunk_store.add([chunk])

    set_active(PipelineConfig(indexing_strategy="fixed_window", retrieval_strategy="hybrid_rrf", post_retrieval_strategy="cross_encoder_rerank"))

    embedding_client = httpx.AsyncClient(transport=httpx.MockTransport(_embed_handler))
    synthesis_client = httpx.AsyncClient(transport=httpx.MockTransport(_embed_handler))
    app = FastAPI()
    app.include_router(build_retrieval_router(registry, embedding_client, synthesis_client, settings))

    with TestClient(app) as client:
        response = client.post("/query", json={"query": "tell me about the fox"})

    assert response.status_code == 501
    registry.close_all()


def test_query_includes_preferences_and_filtered_out_count_with_metadata_filter_strategy(tmp_path):
    settings = _settings(tmp_path, qdrant_collection="t2")
    registry = IndexingCollectionRegistry(settings)
    collection = registry.get("fixed_window")

    kept_id = "22222222-2222-2222-2222-222222222222"
    excluded_id = "33333333-3333-3333-3333-333333333333"
    kept_chunk = ChunkMetadata(
        chunk_id=kept_id, doc_id="d1", doc_id_hash="h1", source_url="paper.pdf", page_number=1,
        chunk_index=0, char_start=0, char_end=10, overlap_with_prev=0,
        indexed_at="2026-07-29T12:00:00+00:00", text="a hotel in lahore", city="lahore", price=None,
    )
    excluded_chunk = ChunkMetadata(
        chunk_id=excluded_id, doc_id="d1", doc_id_hash="h1", source_url="paper.pdf", page_number=1,
        chunk_index=1, char_start=10, char_end=20, overlap_with_prev=0,
        indexed_at="2026-07-29T12:00:00+00:00", text="a hotel in paris", city="paris", price=None,
    )
    collection.bm25_index.add_documents([kept_id, excluded_id], ["a hotel in lahore", "a hotel in paris"])
    collection.vector_index.upsert(
        [kept_id, excluded_id], [[1.0, 0.0], [0.9, 0.1]],
        [kept_chunk.model_dump(), excluded_chunk.model_dump()],
    )
    collection.chunk_store.add([kept_chunk, excluded_chunk])

    set_active(PipelineConfig(indexing_strategy="fixed_window", retrieval_strategy="hybrid_rrf", post_retrieval_strategy="metadata_filter"))

    def groq_handler(request):
        return httpx.Response(200, json={"choices": [{"message": {"content": "A hotel in Lahore [1]."}}]})

    embedding_client = httpx.AsyncClient(transport=httpx.MockTransport(_embed_handler))
    synthesis_client = httpx.AsyncClient(transport=httpx.MockTransport(groq_handler))
    app = FastAPI()
    app.include_router(build_retrieval_router(registry, embedding_client, synthesis_client, settings))

    with TestClient(app) as client:
        response = client.post("/query", json={"query": "hotel in Lahore"})

    assert response.status_code == 200
    body = response.json()
    assert body["preferences"]["city"] == "lahore"
    assert body["filtered_out_count"] == 1
    assert [c["chunk_id"] for c in body["retrieved_chunks"]] == [kept_id]
    registry.close_all()


def test_query_default_settings_marks_overflow_chunks_not_used_in_synthesis(tmp_path):
    settings = _settings(tmp_path, qdrant_collection="t3")
    assert settings.display_top_k == 8
    assert settings.synthesis_context_budget == 6

    registry = IndexingCollectionRegistry(settings)
    collection = registry.get("fixed_window")

    chunk_ids = [f"4444444{i}-4444-4444-4444-444444444444" for i in range(8)]
    chunks = [
        ChunkMetadata(
            chunk_id=chunk_id, doc_id="d1", doc_id_hash="h1", source_url="paper.pdf", page_number=1,
            chunk_index=index, char_start=0, char_end=20, overlap_with_prev=0,
            indexed_at="2026-07-29T12:00:00+00:00", text=f"chunk number {index} about the quick brown fox",
        )
        for index, chunk_id in enumerate(chunk_ids)
    ]
    vector_list = [[1.0, index * 0.1] for index in range(8)]
    collection.vector_index.upsert(chunk_ids, vector_list, [chunk.model_dump() for chunk in chunks])
    collection.chunk_store.add(chunks)

    set_active(PipelineConfig(indexing_strategy="fixed_window", retrieval_strategy="semantic_only", post_retrieval_strategy="none"))

    def groq_handler(request):
        return httpx.Response(200, json={"choices": [{"message": {"content": "A generic answer with no citations."}}]})

    embedding_client = httpx.AsyncClient(transport=httpx.MockTransport(_embed_handler))
    synthesis_client = httpx.AsyncClient(transport=httpx.MockTransport(groq_handler))
    app = FastAPI()
    app.include_router(build_retrieval_router(registry, embedding_client, synthesis_client, settings))

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
    registry.close_all()
