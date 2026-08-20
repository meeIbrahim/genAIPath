from __future__ import annotations

from typing import Awaitable, Callable

import httpx

from app.config import Settings
from app.indexing.bm25_index import InMemoryBM25Index
from app.indexing.chunk_store import ChunkStore
from app.indexing.vector_index import QdrantVectorIndex
from app.retrieval.models import FusedChunk
from app.retrieval.strategies import bm25_only, hybrid_rrf, semantic_only

RetrievalStrategyFn = Callable[
    [str, InMemoryBM25Index, QdrantVectorIndex, ChunkStore, httpx.AsyncClient, Settings, int | None],
    Awaitable[list[FusedChunk]],
]

RETRIEVAL_STRATEGIES: dict[str, RetrievalStrategyFn] = {
    "bm25_only": bm25_only.search,
    "semantic_only": semantic_only.search,
    "hybrid_rrf": hybrid_rrf.search,
}
