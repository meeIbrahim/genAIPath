from __future__ import annotations

import dataclasses

import httpx
from fastapi import APIRouter

from app.config import Settings, settings as default_settings
from app.extraction.preferences import extract_preferences
from app.indexing.bm25_index import InMemoryBM25Index
from app.indexing.chunk_store import ChunkStore
from app.indexing.vector_index import QdrantVectorIndex
from app.retrieval.filtering import filter_chunks
from app.retrieval.judge import JudgeError, JudgeVerdict, judge_context
from app.retrieval.models import FusedChunk, JudgeAttempt, QueryRequest, QueryResponse
from app.retrieval.retriever import retrieve
from app.retrieval.synthesis import synthesize_answer

FALLBACK_ANSWER = (
    "I don't have enough reliable information in the indexed content to answer this question confidently."
)


async def _judge_safely(
    query: str, chunks: list[FusedChunk], http_client: httpx.AsyncClient, settings: Settings
) -> JudgeVerdict:
    try:
        return await judge_context(query, chunks, http_client, settings)
    except JudgeError as exc:
        return JudgeVerdict(verdict="context_insufficient", raw_response=f"(judge error: {exc})")


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
        preferences = extract_preferences(request.query)

        fused_chunks = await retrieve(
            request.query, bm25_index, vector_index, chunk_store, embedding_client, settings, request.top_k
        )
        kept_chunks, filtered_out_count = filter_chunks(fused_chunks, preferences)
        verdict = await _judge_safely(request.query, kept_chunks, synthesis_client, settings)
        judge_attempts = [JudgeAttempt(attempt=1, verdict=verdict.verdict, raw_response=verdict.raw_response)]

        if verdict.verdict == "context_insufficient":
            retry_settings = dataclasses.replace(
                settings,
                retrieval_top_k=settings.retrieval_top_k * settings.judge_retry_top_k_multiplier,
                display_top_k=settings.display_top_k * settings.judge_retry_top_k_multiplier,
            )
            fused_chunks = await retrieve(
                request.query, bm25_index, vector_index, chunk_store, embedding_client, retry_settings, request.top_k
            )
            kept_chunks, filtered_out_count = filter_chunks(fused_chunks, preferences)
            verdict = await _judge_safely(request.query, kept_chunks, synthesis_client, retry_settings)
            judge_attempts.append(
                JudgeAttempt(attempt=2, verdict=verdict.verdict, raw_response=verdict.raw_response)
            )

        if verdict.verdict == "context_insufficient":
            for chunk in kept_chunks:
                chunk.used_in_synthesis = False
            return QueryResponse(
                query=request.query,
                answer=FALLBACK_ANSWER,
                citations=[],
                retrieved_chunks=kept_chunks,
                preferences=preferences,
                filtered_out_count=filtered_out_count,
                judge_attempts=judge_attempts,
            )

        answer, citations, used_chunk_ids = await synthesize_answer(
            request.query, kept_chunks, synthesis_client, settings
        )
        for chunk in kept_chunks:
            chunk.used_in_synthesis = chunk.chunk_id in used_chunk_ids

        return QueryResponse(
            query=request.query,
            answer=answer,
            citations=citations,
            retrieved_chunks=kept_chunks,
            preferences=preferences,
            filtered_out_count=filtered_out_count,
            judge_attempts=judge_attempts,
        )

    return router
