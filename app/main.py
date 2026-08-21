from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import Settings, settings
from app.pipeline.registry import IndexingCollectionRegistry
from app.pipeline.router import build_pipeline_router
from app.retrieval.router import build_retrieval_router


def create_app(app_settings: Settings = settings) -> FastAPI:
    registry = IndexingCollectionRegistry(app_settings)
    embedding_client = httpx.AsyncClient()
    synthesis_client = httpx.AsyncClient()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        yield
        registry.close_all()

    app = FastAPI(title="RAG Pipeline Showcase", lifespan=lifespan)

    app.include_router(build_pipeline_router(registry, embedding_client, app_settings))
    app.include_router(build_retrieval_router(registry, embedding_client, synthesis_client, app_settings))

    app.mount("/static", StaticFiles(directory="app/static"), name="static")

    @app.get("/")
    async def serve_ingest_page() -> FileResponse:
        return FileResponse("app/static/ingest.html")

    @app.get("/query-ui")
    async def serve_query_page() -> FileResponse:
        return FileResponse("app/static/query.html")

    return app


app = create_app()
