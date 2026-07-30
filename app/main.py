from __future__ import annotations

import httpx
from fastapi import FastAPI

from app.config import Settings, settings
from app.indexing.bm25_index import InMemoryBM25Index
from app.indexing.chunk_store import ChunkStore
from app.indexing.indexer import index_document
from app.indexing.router import build_indexing_router
from app.indexing.vector_index import QdrantVectorIndex
from app.ingestion.job_store import JobStore
from app.ingestion.router import build_ingestion_router
from app.retrieval.router import build_retrieval_router


def create_app(app_settings: Settings = settings) -> FastAPI:
    app = FastAPI(title="RAG Ingestion Service")

    job_store = JobStore()
    bm25_index = InMemoryBM25Index()
    vector_index = QdrantVectorIndex(app_settings)
    chunk_store = ChunkStore()
    embedding_client = httpx.AsyncClient()
    synthesis_client = httpx.AsyncClient()

    async def index_sink(payload) -> None:
        await index_document(payload, bm25_index, vector_index, chunk_store, embedding_client, app_settings)

    app.include_router(build_ingestion_router(job_store, sink=index_sink))
    app.include_router(build_indexing_router(bm25_index, vector_index, chunk_store, embedding_client, app_settings))
    app.include_router(
        build_retrieval_router(
            bm25_index, vector_index, chunk_store, embedding_client, synthesis_client, app_settings
        )
    )

    return app


app = create_app()
