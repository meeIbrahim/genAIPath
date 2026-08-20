from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

IndexingStrategyId = Literal["fixed_window", "semantic", "hierarchical", "hierarchical_summary"]
RetrievalStrategyId = Literal["bm25_only", "semantic_only", "hybrid_rrf"]
PostRetrievalStrategyId = Literal["none", "metadata_filter", "cross_encoder_rerank"]


class PipelineConfig(BaseModel):
    indexing_strategy: IndexingStrategyId
    retrieval_strategy: RetrievalStrategyId
    post_retrieval_strategy: PostRetrievalStrategyId


def collection_name_for(base_collection: str, indexing_strategy: IndexingStrategyId) -> str:
    return f"{base_collection}__{indexing_strategy}"


_active: PipelineConfig | None = None


def get_active() -> PipelineConfig | None:
    return _active


def set_active(config: PipelineConfig | None) -> None:
    global _active
    _active = config
