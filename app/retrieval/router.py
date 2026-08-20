from __future__ import annotations

import httpx
from fastapi import APIRouter, HTTPException

from app.config import Settings, settings as default_settings
from app.extraction.preferences import extract_preferences
from app.pipeline.config import get_active
from app.pipeline.registry import IndexingCollectionRegistry
from app.postretrieval.strategies import POST_RETRIEVAL_STRATEGIES
from app.retrieval.models import QueryRequest, QueryResponse
from app.retrieval.strategies import RETRIEVAL_STRATEGIES
from app.retrieval.synthesis import synthesize_answer


def build_retrieval_router(
    registry: IndexingCollectionRegistry,
    embedding_client: httpx.AsyncClient,
    synthesis_client: httpx.AsyncClient,
    settings: Settings = default_settings,
) -> APIRouter:
    router = APIRouter()

    @router.post("/query", response_model=QueryResponse)
    async def query(request: QueryRequest) -> QueryResponse:
        active = get_active()
        if active is None:
            raise HTTPException(status_code=400, detail="no pipeline loaded — call POST /pipeline/load first")

        collection = registry.get(active.indexing_strategy)
        retrieval_fn = RETRIEVAL_STRATEGIES[active.retrieval_strategy]
        post_retrieval_fn = POST_RETRIEVAL_STRATEGIES[active.post_retrieval_strategy]

        preferences = extract_preferences(request.query)

        fused_chunks = await retrieval_fn(
            request.query, collection.bm25_index, collection.vector_index, collection.chunk_store,
            embedding_client, settings, request.top_k,
        )
        kept_chunks, filtered_out_count = post_retrieval_fn(fused_chunks, preferences)

        answer, citations, used_chunk_ids = await synthesize_answer(request.query, kept_chunks, synthesis_client, settings)
        for chunk in kept_chunks:
            chunk.used_in_synthesis = chunk.chunk_id in used_chunk_ids

        return QueryResponse(
            query=request.query, answer=answer, citations=citations, retrieved_chunks=kept_chunks,
            preferences=preferences, filtered_out_count=filtered_out_count,
        )

    return router
