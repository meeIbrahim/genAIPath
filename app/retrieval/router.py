from __future__ import annotations

import httpx
from fastapi import APIRouter

from app.config import Settings, settings as default_settings
from app.indexing.bm25_index import InMemoryBM25Index
from app.indexing.chunk_store import ChunkStore
from app.indexing.vector_index import QdrantVectorIndex
from app.retrieval.models import QueryRequest, QueryResponse
from app.retrieval.retriever import retrieve
from app.retrieval.synthesis import synthesize_answer


def build_retrieval_router(
    bm25_index: InMemoryBM25Index,
    vector_index: QdrantVectorIndex,
    chunk_store: ChunkStore,
    embedding_client: httpx.AsyncClient,
    synthesis_client: httpx.AsyncClient,
    settings: Settings = default_settings,
) -> APIRouter:
    router = APIRouter()

    @router.post("/query", response_model=QueryResponse)
    async def query(request: QueryRequest) -> QueryResponse:
        fused_chunks = await retrieve(
            request.query, bm25_index, vector_index, chunk_store, embedding_client, settings, request.top_k
        )
        answer, citations, used_chunk_ids = await synthesize_answer(
            request.query, fused_chunks, synthesis_client, settings
        )
        for chunk in fused_chunks:
            chunk.used_in_synthesis = chunk.chunk_id in used_chunk_ids

        return QueryResponse(query=request.query, answer=answer, citations=citations, retrieved_chunks=fused_chunks)

    return router
