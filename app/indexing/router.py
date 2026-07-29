from __future__ import annotations

import httpx
from fastapi import APIRouter

from app.config import Settings, settings as default_settings
from app.indexing.bm25_index import InMemoryBM25Index
from app.indexing.chunk_store import ChunkStore
from app.indexing.indexer import index_document
from app.indexing.models import IndexResult
from app.indexing.vector_index import QdrantVectorIndex
from app.ingestion.models import IngestionPayload


def build_indexing_router(
    bm25_index: InMemoryBM25Index,
    vector_index: QdrantVectorIndex,
    chunk_store: ChunkStore,
    http_client: httpx.AsyncClient,
    settings: Settings = default_settings,
) -> APIRouter:
    router = APIRouter()

    @router.post("/index/chunk", response_model=IndexResult)
    async def index_chunk(payload: IngestionPayload) -> IndexResult:
        return await index_document(payload, bm25_index, vector_index, chunk_store, http_client, settings)

    return router
