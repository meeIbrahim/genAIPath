from __future__ import annotations

import httpx

from app.config import Settings, settings as default_settings
from app.indexing.bm25_index import InMemoryBM25Index
from app.indexing.chunk_store import ChunkStore
from app.indexing.vector_index import QdrantVectorIndex
from app.retrieval.fusion import RankedHit
from app.retrieval.models import FusedChunk
from app.retrieval.strategies.hybrid_rrf import assemble


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
    results = bm25_index.search(query, settings.retrieval_top_k)
    bm25_hits = [RankedHit(chunk_id, rank + 1, score) for rank, (chunk_id, score) in enumerate(results)]
    return assemble(bm25_hits, [], chunk_store, settings, display_top_k)
