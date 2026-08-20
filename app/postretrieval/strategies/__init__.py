from __future__ import annotations

from typing import Callable

from app.extraction.preferences import QueryPreferences
from app.postretrieval.strategies import cross_encoder_rerank, metadata_filter, none
from app.retrieval.models import FusedChunk

PostRetrievalStrategyFn = Callable[[list[FusedChunk], QueryPreferences], tuple[list[FusedChunk], int]]

POST_RETRIEVAL_STRATEGIES: dict[str, PostRetrievalStrategyFn] = {
    "none": none.apply,
    "metadata_filter": metadata_filter.apply,
    "cross_encoder_rerank": cross_encoder_rerank.apply,
}
