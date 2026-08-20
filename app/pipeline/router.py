from __future__ import annotations

import httpx
from fastapi import APIRouter

from app.config import Settings, settings as default_settings
from app.pipeline.config import PipelineConfig, get_active
from app.pipeline.loader import load_pipeline
from app.pipeline.models import PipelineLoadResult, PipelineStatus
from app.pipeline.registry import INDEXING_STRATEGY_IDS, IndexingCollectionRegistry


def build_pipeline_router(
    registry: IndexingCollectionRegistry,
    http_client: httpx.AsyncClient,
    settings: Settings = default_settings,
) -> APIRouter:
    router = APIRouter()

    @router.post("/pipeline/load", response_model=PipelineLoadResult)
    async def load(config: PipelineConfig) -> PipelineLoadResult:
        return await load_pipeline(config, registry, http_client, settings)

    @router.get("/pipeline/status", response_model=PipelineStatus)
    async def status() -> PipelineStatus:
        return PipelineStatus(
            active=get_active(),
            doc_counts={strategy_id: registry.doc_count(strategy_id) for strategy_id in INDEXING_STRATEGY_IDS},
        )

    return router
