from __future__ import annotations

import asyncio
from typing import Callable

import httpx
from fastapi import APIRouter, HTTPException

from app.ingestion.job_store import JobStore
from app.ingestion.models import IngestRequest, IngestResponse, JobStatusResponse
from app.ingestion.worker import ingest_url


def build_ingestion_router(
    store: JobStore,
    client_factory: Callable[[], httpx.AsyncClient] = httpx.AsyncClient,
) -> APIRouter:
    router = APIRouter()

    @router.post("/ingest", response_model=IngestResponse)
    async def create_ingest_job(request: IngestRequest) -> IngestResponse:
        job_id = store.create_job(request.urls)
        client = client_factory()

        async def run_all() -> None:
            try:
                # return_exceptions=True is defense in depth: ingest_url already
                # isolates per-URL failures internally, but this ensures gather
                # itself never raises and triggers an early client.aclose() while
                # sibling URLs in the same job are still in flight.
                await asyncio.gather(
                    *(ingest_url(job_id, url, store, client) for url in request.urls),
                    return_exceptions=True,
                )
            finally:
                await client.aclose()

        asyncio.create_task(run_all())
        return IngestResponse(job_id=job_id)

    @router.get("/ingest/{job_id}/status", response_model=JobStatusResponse)
    async def get_ingest_status(job_id: str) -> JobStatusResponse:
        if not store.exists(job_id):
            raise HTTPException(status_code=404, detail="job not found")
        return store.get_status(job_id)

    return router
