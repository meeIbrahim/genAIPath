from __future__ import annotations

from fastapi import FastAPI

from app.ingestion.job_store import JobStore
from app.ingestion.router import build_ingestion_router


def create_app() -> FastAPI:
    app = FastAPI(title="RAG Ingestion Service")
    store = JobStore()
    app.include_router(build_ingestion_router(store))
    return app


app = create_app()
