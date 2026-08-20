from __future__ import annotations

from app.extraction.preferences import QueryPreferences
from app.retrieval.models import FusedChunk


def apply(fused_chunks: list[FusedChunk], preferences: QueryPreferences) -> tuple[list[FusedChunk], int]:
    return fused_chunks, 0
