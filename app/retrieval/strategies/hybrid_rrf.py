from __future__ import annotations

import asyncio

import httpx

from app.config import Settings, settings as default_settings
from app.indexing.bm25_index import InMemoryBM25Index
from app.indexing.chunk_store import ChunkStore
from app.indexing.embeddings import embed_texts
from app.indexing.vector_index import QdrantVectorIndex
from app.retrieval.fusion import RankedHit, reciprocal_rank_fusion
from app.retrieval.models import FusedChunk


def assemble(
    bm25_hits: list[RankedHit],
    semantic_hits: list[RankedHit],
    chunk_store: ChunkStore,
    settings: Settings,
    display_top_k: int,
) -> list[FusedChunk]:
    fused_hits = reciprocal_rank_fusion(bm25_hits, semantic_hits, settings.rrf_k, display_top_k)
    metadata_by_id = {
        chunk.chunk_id: chunk for chunk in chunk_store.get_many([hit.chunk_id for hit in fused_hits])
    }
    return [
        FusedChunk(
            chunk_id=hit.chunk_id,
            text=metadata_by_id[hit.chunk_id].text,
            source_url=metadata_by_id[hit.chunk_id].source_url,
            page_number=metadata_by_id[hit.chunk_id].page_number,
            city=metadata_by_id[hit.chunk_id].city,
            price=metadata_by_id[hit.chunk_id].price,
            bm25_rank=hit.bm25_rank,
            bm25_score=hit.bm25_score,
            semantic_rank=hit.semantic_rank,
            semantic_score=hit.semantic_score,
            fused_rank=hit.fused_rank,
            rrf_score=hit.rrf_score,
            matched_methods=hit.matched_methods,
        )
        for hit in fused_hits
        if hit.chunk_id in metadata_by_id
    ]


async def search(
    query: str,
    bm25_index: InMemoryBM25Index,
    vector_index: QdrantVectorIndex,
    chunk_store: ChunkStore,
    http_client: httpx.AsyncClient,
    settings: Settings = default_settings,
    top_k: int | None = None,
) -> list[FusedChunk]:
    display_top_k = settings.display_top_k if top_k is None else top_k

    async def bm25_search() -> list[RankedHit]:
        results = bm25_index.search(query, settings.retrieval_top_k)
        return [RankedHit(chunk_id, rank + 1, score) for rank, (chunk_id, score) in enumerate(results)]

    async def semantic_search() -> list[RankedHit]:
        vectors = await embed_texts(http_client, [query], settings)
        results = vector_index.search(vectors[0], settings.retrieval_top_k)
        return [RankedHit(chunk_id, rank + 1, score) for rank, (chunk_id, score) in enumerate(results)]

    bm25_hits, semantic_hits = await asyncio.gather(bm25_search(), semantic_search())
    return assemble(bm25_hits, semantic_hits, chunk_store, settings, display_top_k)
